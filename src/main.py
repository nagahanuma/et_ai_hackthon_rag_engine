"""
Production FastAPI Application.
Exposes:
1. Document Lifecycle Management (Upload, Status, List, Replace, Delete)
2. Compliance Audit Query with Observability Trace
3. Live Database Inspectors:
   - GET /api/v1/inspect/sqlite (Inspect relational Ground Truth tables)
   - GET /api/v1/inspect/chroma (Inspect ChromaDB vectors and payloads)
4. Static Mount: Serves frontend Single Page App at http://localhost:8000
"""

import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, status, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.config import settings
from src.db import (
    init_db, 
    get_all_documents, 
    create_job,
    update_job_status,
    get_job_status, 
    get_db_connection,
    list_chat_sessions,
    get_session_messages,
    delete_chat_session,
    delete_all_chat_sessions,
    register_user,
    authenticate_user,
    get_user_by_id,
    update_user_api_keys,
    increment_user_usage,
    get_all_users_usage,
    delete_user_and_storage
)
from src.ingestion_service import IngestionService
from src.rag_pipeline import ComplianceRAGPipeline, ComplianceAnswer
from src.logger import get_logger, set_transaction_id
from src.auth import create_access_token, verify_access_token

logger = get_logger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="Enterprise Document Intelligence & Compliance Audit RAG Engine",
    description="Production-grade Agentic RAG for multi-domain document intelligence, compliance verification, and internal audit controls.",
    version="1.0.0"
)

# Initialize Rate Limiter (IP-based)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Deployment & Production Gatekeeper Middleware
# When PROD_MODE == 'Y': Strictly blocks requests originating from local addresses (localhost, 127.0.0.1, ::1)
# When PROD_MODE == 'N': Accepts all requests including local test requests
@app.middleware("http")
async def prod_mode_gatekeeper_middleware(request: Request, call_next):
    # Always allow container health probes (e.g. GCP Cloud Run, Docker healthchecks)
    if request.url.path in ("/health", "/api/v1/health"):
        return await call_next(request)

    prod_flag = getattr(settings, "PROD_MODE", "N").strip().upper()
    if prod_flag == "Y":
        client_host = request.client.host if request.client else ""
        origin = request.headers.get("origin", "").lower()
        host = request.headers.get("host", "").lower()
        
        is_local = (
            client_host in ("127.0.0.1", "localhost", "::1", "0.0.0.0") or
            "localhost" in origin or
            "127.0.0.1" in origin or
            "localhost" in host or
            "127.0.0.1" in host
        )
        if is_local:
            logger.warning(f"Production Gatekeeper: Blocked local request to '{request.url.path}' from host '{client_host}' / Origin '{origin}' (PROD_MODE=Y)")
            return JSONResponse(
                status_code=403,
                content={
                    "status": "FORBIDDEN",
                    "detail": "Production Mode Active (PROD_MODE=Y): Local requests from localhost/127.0.0.1 are disabled. Set PROD_MODE=N in .env to allow local requests."
                }
            )

    response = await call_next(request)
    return response

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Singletons
ingestion_service = IngestionService()
rag_pipeline = ComplianceRAGPipeline()

# Max allowed file size: Configurable via MAX_UPLOAD_SIZE_MB in .env (Default: 50 MB to preserve Supabase quota)
MAX_FILE_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


@app.on_event("startup")
def startup_event():
    """Initializes database tables on application boot and runs inactive account retention purge."""
    init_db()
    try:
        cleanup_res = ingestion_service.cleanup_inactive_users_data(days_inactive=90)
        if cleanup_res.get("purged_users", 0) > 0:
            logger.info(
                f"Data Retention: Purged {cleanup_res['purged_users']} inactive users and "
                f"{cleanup_res['purged_docs']} orphaned documents (>90 days inactive)."
            )
    except Exception as e:
        logger.warning(f"Data retention check encountered error on boot: {e}")
    logger.info("Application startup: Database and ChromaDB vector store ready.")


# ==========================================
# PYDANTIC SCHEMAS
# ==========================================

class UserRegisterRequest(BaseModel):
    username: str = Field(..., example="compliance_analyst")
    email: str = Field(..., example="analyst@bank.com")
    password: str = Field(..., example="Secret123")


class UserLoginRequest(BaseModel):
    login: str = Field(..., example="nagahanuma")
    password: str = Field(..., example="Rag@Prod")


class UserSettingsRequest(BaseModel):
    custom_groq_key: Optional[str] = None
    custom_openai_key: Optional[str] = None
    custom_model: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the minimum Capital Adequacy Ratio?")
    session_id: Optional[str] = Field(None, example="sess_12345678")
    user_id: Optional[str] = Field(None, example="usr_admin_master")
    custom_model: Optional[str] = Field(None, example="llama-3.3-70b-versatile")


class CitationResponse(BaseModel):
    doc_id: str
    filename: str
    page_number: int
    parent_id: str
    content_type: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    session_id: Optional[str] = None
    citations: List[CitationResponse]
    execution_time_ms: float
    total_sources_retrieved: int
    trace: Dict[str, Any] = Field(default_factory=dict)


# ==========================================
# 1. DOCUMENT LIFECYCLE APIS (CRUD)
# ==========================================

def run_background_ingest(
    temp_path_str: str, 
    doc_type: str, 
    user_id: Optional[str] = None,
    original_filename: Optional[str] = None,
    job_id: Optional[str] = None,
    doc_id: Optional[str] = None
):
    """Background task to parse and embed files without blocking HTTP requests."""
    temp_path = Path(temp_path_str)
    try:
        ingestion_service.ingest_document(
            temp_path, 
            doc_type=doc_type, 
            user_id=user_id,
            original_filename=original_filename,
            job_id=job_id,
            doc_id=doc_id
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/api/v1/documents/upload", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("5/minute")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: str = Form("regulatory_circular"),
    user_id: Optional[str] = Form(None)
):
    """Uploads a regulatory document (PDF, TXT, MD) and schedules background ingestion with user ownership."""
    # Strict Authentication Check
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in or register before uploading regulatory files."
        )
    user_record = get_user_by_id(user_id)
    if not user_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user account. Please log in again."
        )

    allowed_exts = [".pdf", ".txt", ".md"]
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported format '{suffix}'. Allowed: {allowed_exts}"
        )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, temp_file)
        file_size = Path(temp_file.name).stat().st_size

        if file_size > MAX_FILE_SIZE_BYTES:
            temp_file.close()
            Path(temp_file.name).unlink()
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB"
            )
    finally:
        temp_file.close()

    # Pre-register job so client can immediately track real-time progress
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    create_job(job_id=job_id, doc_id=doc_id, filename=file.filename)
    update_job_status(job_id, status="PENDING", progress=5.0)

    background_tasks.add_task(
        run_background_ingest, 
        temp_file.name, 
        doc_type, 
        user_id, 
        file.filename, 
        job_id, 
        doc_id
    )

    logger.info(f"Accepted file '{file.filename}' ({file_size} bytes) for user '{user_record['username']}' with job_id: {job_id}.")
    return {
        "status": "ACCEPTED",
        "job_id": job_id,
        "doc_id": doc_id,
        "message": "File accepted for ingestion. Monitor progress via /api/v1/documents/jobs/{job_id}",
        "filename": file.filename,
    }


@app.get("/api/v1/documents/jobs/{job_id}")
def check_job_status(job_id: str):
    """Returns real-time progress for frontend progress bars."""
    job = get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@app.get("/api/v1/documents")
def list_documents(user_id: Optional[str] = None):
    """Returns active documents. Admin sees all; standard users see only their own uploaded documents."""
    is_admin = False
    if user_id:
        user = get_user_by_id(user_id)
        if user and user.get("role") == "admin":
            is_admin = True
    
    # If not admin, strictly filter by user_id
    filter_user_id = None if is_admin else user_id
    docs = get_all_documents(user_id=filter_user_id)
    return {"total_documents": len(docs), "documents": docs}


@app.put("/api/v1/documents/{doc_id}/replace")
async def replace_document(
    doc_id: str,
    file: UploadFile = File(...)
):
    """Replaces an existing document with a revised version atomically."""
    suffix = Path(file.filename).suffix.lower()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, temp_file)
    finally:
        temp_file.close()

    try:
        result = ingestion_service.replace_document_version(
            doc_id, Path(temp_file.name), original_filename=file.filename
        )
        return result
    finally:
        Path(temp_file.name).unlink(missing_ok=True)


@app.delete("/api/v1/documents/{doc_id}")
def delete_document_endpoint(doc_id: str, user_id: Optional[str] = None):
    """Permanently purges or unlinks a document across relational DB and pgvector/ChromaDB."""
    is_admin = False
    if user_id:
        user = get_user_by_id(user_id)
        if user and user.get("role") == "admin":
            is_admin = True

    del_user = None if is_admin else user_id
    success = ingestion_service.delete_document_completely(doc_id, user_id=del_user)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return {
        "status": "DELETED",
        "message": f"Document '{doc_id}' was removed successfully."
    }


# ==========================================
# 2. AUTHENTICATION, ADMIN & USER SETTINGS APIS
# ==========================================

@app.post("/api/v1/auth/register", status_code=status.HTTP_201_CREATED)
def api_register(body: UserRegisterRequest):
    """Registers a new standard user account and issues cryptographic JWT access token."""
    try:
        user = register_user(username=body.username, email=body.email, password=body.password)
        jwt_token = create_access_token(user["user_id"], user["username"], user.get("role", "user"))
        return {
            "status": "SUCCESS",
            "message": "User registered successfully.",
            "user": user,
            "token": jwt_token
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/auth/login")
def api_login(body: UserLoginRequest):
    """Authenticates standard user or Admin and issues signed HMAC-SHA256 JWT."""
    user = authenticate_user(login_id=body.login, password=body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username/email or password.")
    jwt_token = create_access_token(user["user_id"], user["username"], user.get("role", "user"))
    return {
        "status": "SUCCESS",
        "message": "Login successful.",
        "user": user,
        "token": jwt_token
    }


@app.get("/api/v1/auth/me")
def api_me(user_id: str):
    """Returns safe user profile and role."""
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"user": user}


@app.post("/api/v1/user/settings")
def api_update_settings(user_id: str, body: UserSettingsRequest):
    """Configures user BYOK API keys (OpenAI or Groq key) and preferred LLM model."""
    success = update_user_api_keys(
        user_id=user_id,
        custom_groq_key=body.custom_groq_key,
        custom_openai_key=body.custom_openai_key,
        custom_model=body.custom_model
    )
    if not success:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"status": "SUCCESS", "message": "API keys and model preferences updated successfully."}


@app.get("/api/v1/admin/users")
def api_admin_users(admin_user_id: str):
    """
    Admin Only: View how many users registered and how much application resources they used.
    """
    user = get_user_by_id(admin_user_id)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: You are not authorized to view Admin resource analytics.")
    users = get_all_users_usage()
    total_queries = sum(u.get("total_queries", 0) for u in users)
    total_tokens = sum(u.get("total_tokens", 0) for u in users)
    total_storage_bytes = sum(u.get("total_storage_bytes", 0) for u in users)
    total_chunks = sum(u.get("total_chunks", 0) for u in users)
    return {
        "status": "SUCCESS",
        "total_users": len(users),
        "total_queries_executed": total_queries,
        "total_tokens_consumed": total_tokens,
        "total_storage_bytes": total_storage_bytes,
        "total_chunks_stored": total_chunks,
        "users": users
    }


@app.post("/api/v1/admin/maintenance/cleanup")
def api_admin_cleanup(admin_user_id: str, days: int = 90):
    """Admin Only: Manually trigger inactive user & document purge to optimize database storage."""
    admin = get_user_by_id(admin_user_id)
    if not admin or admin.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Unauthorized. Admin privileges required.")
    res = ingestion_service.cleanup_inactive_users_data(days_inactive=days)
    return {
        "status": "SUCCESS",
        "message": f"Purged {res['purged_users']} inactive users and {res['purged_docs']} associated documents (> {days} days inactive).",
        "purged_users": res["purged_users"],
        "purged_docs": res["purged_docs"]
    }


@app.delete("/api/v1/admin/users/{target_user_id}")
def api_admin_delete_user(target_user_id: str, admin_user_id: str):
    """
    Admin Only: Delete a specific user and reclaim all database and vector storage.
    Purges any documents, chunks, and embeddings exclusively owned by this user.
    """
    try:
        res = delete_user_and_storage(admin_user_id=admin_user_id, target_user_id=target_user_id)
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting user {target_user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete user and reclaim storage: {e}")



# ==========================================
# 3. COMPLIANCE AUDIT & QUERY API
# ==========================================

@app.post("/api/v1/compliance/query", response_model=QueryResponse)
@limiter.limit("20/minute")
def query_compliance(request: Request, body: QueryRequest):
    """Compliance query returning answer, exact citations, session_id, and full pipeline trace."""
    set_transaction_id()

    # Strict Authentication Check: Support Bearer JWT or user_id
    auth_header = request.headers.get("authorization")
    token_claims = verify_access_token(auth_header) if auth_header else None
    effective_user_id = token_claims.get("sub") if token_claims else body.user_id

    if not effective_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in or register before querying the compliance engine."
        )
    user_record = get_user_by_id(effective_user_id)
    if not user_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user session. Please log in again."
        )

    is_admin = (user_record.get("role") == "admin")
    custom_groq = user_record.get("custom_groq_key")
    custom_openai = user_record.get("custom_openai_key")
    resolved_model = body.custom_model or user_record.get("custom_model")

    result: ComplianceAnswer = rag_pipeline.query(
        user_question=body.query, 
        session_id=body.session_id,
        custom_groq_key=custom_groq,
        custom_openai_key=custom_openai,
        custom_model=resolved_model,
        user_id=effective_user_id,
        is_admin=is_admin
    )

    # Track resource consumption for the user
    if effective_user_id:
        approx_tokens = int((len(body.query) + len(result.answer)) / 4) + 150
        increment_user_usage(user_id=effective_user_id, queries=1, tokens=approx_tokens)
    
    citations = [
        CitationResponse(
            doc_id=c.doc_id,
            filename=c.filename,
            page_number=c.page_number,
            parent_id=c.parent_id,
            content_type=c.content_type
        )
        for c in result.citations
    ]

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        session_id=result.session_id,
        citations=citations,
        execution_time_ms=result.execution_time_ms,
        total_sources_retrieved=result.total_sources_retrieved,
        trace=result.trace
    )


@app.get("/api/v1/compliance/sessions")
def get_chat_sessions(user_id: Optional[str] = None, limit: int = 50):
    """Lists recent conversation sessions for session history/switching, optionally filtered by user_id."""
    sessions = list_chat_sessions(user_id=user_id, limit=limit)
    return {"sessions": sessions, "total_sessions": len(sessions)}


@app.get("/api/v1/compliance/sessions/{session_id}/messages")
def get_session_history(session_id: str):
    """Retrieves all chat messages and citations for a conversation session."""
    messages = get_session_messages(session_id)
    return {"session_id": session_id, "messages": messages, "total_messages": len(messages)}


@app.delete("/api/v1/compliance/sessions/all")
def purge_all_user_chat_sessions(user_id: str):
    """Deletes all conversation sessions for a specific user to reclaim database storage."""
    purged_count = delete_all_chat_sessions(user_id=user_id)
    return {
        "status": "DELETED",
        "message": f"Purged {purged_count} conversation sessions for user {user_id}.",
        "purged_count": purged_count
    }


@app.delete("/api/v1/compliance/sessions/{session_id}")
def purge_chat_session(session_id: str):
    """Deletes a conversation session and all its messages."""
    success = delete_chat_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"status": "DELETED", "message": f"Session '{session_id}' was purged."}


@app.get("/api/v1/cache/status")
def get_cache_status():
    """Returns semantic cache metrics and total cached queries."""
    if hasattr(rag_pipeline, "semantic_cache") and rag_pipeline.semantic_cache:
        count = rag_pipeline.semantic_cache.collection.count()
        return {
            "status": "active",
            "total_cached_queries": count,
            "similarity_threshold": rag_pipeline.semantic_cache.threshold,
            "collection_name": rag_pipeline.semantic_cache.collection_name
        }
    return {"status": "disabled", "total_cached_queries": 0}


@app.delete("/api/v1/cache/clear")
def clear_semantic_cache():
    """Purges the in-engine semantic cache collection."""
    if hasattr(rag_pipeline, "semantic_cache") and rag_pipeline.semantic_cache:
        rag_pipeline.semantic_cache.clear()
        return {"status": "cleared", "message": "Semantic cache purged successfully."}
    return {"status": "not_active"}


# ==========================================
# 3. LIVE STORAGE INSPECTORS (SQLITE vs CHROMADB)
# ==========================================

@app.get("/api/v1/inspect/sqlite")
def inspect_sqlite():
    """Returns raw rows from relational tables for side-by-side frontend inspection."""
    from src.db import is_postgres
    with get_db_connection() as conn:
        if is_postgres():
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM documents ORDER BY created_at DESC LIMIT 20;")
                docs = cur.fetchall()
                cur.execute("SELECT * FROM parent_chunks ORDER BY created_at DESC LIMIT 20;")
                parents = cur.fetchall()
                cur.execute("SELECT * FROM ingestion_jobs ORDER BY created_at DESC LIMIT 20;")
                jobs = cur.fetchall()
        else:
            docs = conn.execute("SELECT * FROM documents ORDER BY created_at DESC LIMIT 20;").fetchall()
            parents = conn.execute("SELECT * FROM parent_chunks ORDER BY created_at DESC LIMIT 20;").fetchall()
            jobs = conn.execute("SELECT * FROM ingestion_jobs ORDER BY created_at DESC LIMIT 20;").fetchall()

        return {
            "database": "Supabase PostgreSQL (Ground Truth Store)" if is_postgres() else "SQLite (Ground Truth Store)",
            "tables": {
                "documents": [dict(d) for d in docs],
                "parent_chunks": [
                    {
                        "parent_id": p["parent_id"],
                        "doc_id": p["doc_id"],
                        "page_number": p["page_number"],
                        "chunk_index": p["chunk_index"],
                        "content_type": p["content_type"],
                        "content_preview": p["content"][:200] + "..." if len(p["content"]) > 200 else p["content"],
                        "total_chars": len(p["content"])
                    }
                    for p in parents
                ],
                "ingestion_jobs": [dict(j) for j in jobs]
            }
        }


@app.get("/api/v1/inspect/chroma")
def inspect_chroma():
    """Returns raw vector records and metadata payloads from Supabase pgvector or ChromaDB."""
    from src.db import is_postgres
    try:
        vm = rag_pipeline.retriever.vector_manager
        if is_postgres():
            from src.db import get_all_child_chunks_for_inspection
            raw_records = get_all_child_chunks_for_inspection(limit=20)
            sample_records = []
            for r in raw_records:
                sample_records.append({
                    "id": r["child_id"],
                    "document_snippet": r["content"][:180] + "..." if len(r["content"]) > 180 else r["content"],
                    "metadata": {
                        "parent_id": r["parent_id"],
                        "doc_id": r["doc_id"],
                        "page_number": r["page_number"]
                    },
                    "embedding_preview": "[1024-dimensional BGE-M3 vector in Supabase pgvector]",
                    "vector_dimension": 1024
                })
            
            with get_db_connection() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT COUNT(*) as count FROM child_chunks;")
                    total_count = cur.fetchone()["count"]

            return {
                "database": "Supabase pgvector (Cloud Vector Index)",
                "collection_name": "child_chunks (HNSW Cosine)",
                "total_vectors_in_collection": total_count,
                "sample_records": sample_records
            }
        else:
            collection = vm.collection
            data = collection.get(
                include=["documents", "metadatas", "embeddings"],
                limit=20
            )

            sample_records = []
            if data and data.get("ids"):
                ids = data["ids"]
                docs = data.get("documents") if data.get("documents") is not None else []
                metas = data.get("metadatas") if data.get("metadatas") is not None else []
                embeds = data.get("embeddings") if data.get("embeddings") is not None else []

                for idx in range(len(ids)):
                    embed_vector = embeds[idx] if idx < len(embeds) else []
                    if hasattr(embed_vector, "tolist"):
                        embed_vector = embed_vector.tolist()

                    dim_count = len(embed_vector)
                    embed_preview = (
                        f"[{', '.join(str(round(float(x), 4)) for x in embed_vector[:3])}, ... total {dim_count} dimensions]"
                        if dim_count > 0 else "N/A"
                    )

                    sample_records.append({
                        "id": ids[idx],
                        "document_snippet": docs[idx][:180] + "..." if (idx < len(docs) and len(docs[idx]) > 180) else (docs[idx] if idx < len(docs) else ""),
                        "metadata": metas[idx] if idx < len(metas) else {},
                        "embedding_preview": embed_preview,
                        "vector_dimension": dim_count
                    })

            return {
                "database": "ChromaDB (Vector Search Index)",
                "collection_name": collection.name,
                "total_vectors_in_collection": collection.count(),
                "sample_records": sample_records
            }
    except Exception as e:
        logger.error(f"Failed to inspect vector storage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



# ==========================================
# 4. FRONTEND STATIC MOUNT & ROOT
# ==========================================

# Mount frontend directory if it exists
frontend_dir = Path("frontend")
frontend_dir.mkdir(parents=True, exist_ok=True)
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/health")
@app.get("/api/v1/health")
def health_check():
    """Lightweight health check endpoint for monitoring & keep-alive pings."""
    return {
        "status": "HEALTHY",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "Enterprise Document Intelligence & Compliance Audit RAG Engine",
        "prod_mode": getattr(settings, "PROD_MODE", "N")
    }


@app.get("/")
def root_redirect():
    """Root endpoint automatically redirecting user to the dashboard UI."""
    return RedirectResponse(url="/app/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
