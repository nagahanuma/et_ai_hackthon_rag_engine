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
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.db import init_db, get_all_documents, get_job_status, get_db_connection
from src.ingestion_service import IngestionService
from src.rag_pipeline import ComplianceRAGPipeline, ComplianceAnswer
from src.logger import get_logger, set_transaction_id

logger = get_logger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="Banking Regulatory Compliance RAG Engine",
    description="Production-grade Agentic RAG for 50,000-page banking regulations and internal controls.",
    version="1.0.0"
)

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

# Max allowed file size: 100 MB
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024


@app.on_event("startup")
def startup_event():
    """Initializes database tables on application boot."""
    init_db()
    logger.info("Application startup: SQLite ground truth and ChromaDB vector store ready.")


# ==========================================
# PYDANTIC SCHEMAS
# ==========================================

class QueryRequest(BaseModel):
    query: str = Field(..., example="What did we build in step 4?")


class CitationResponse(BaseModel):
    doc_id: str
    filename: str
    page_number: int
    parent_id: str
    content_type: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationResponse]
    execution_time_ms: float
    total_sources_retrieved: int
    trace: Dict[str, Any] = Field(default_factory=dict)


# ==========================================
# 1. DOCUMENT LIFECYCLE APIS (CRUD)
# ==========================================

def run_background_ingest(temp_path_str: str, doc_type: str):
    """Background task to parse and embed files without blocking HTTP requests."""
    temp_path = Path(temp_path_str)
    try:
        ingestion_service.ingest_document(temp_path, doc_type=doc_type)
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/api/v1/documents/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: str = Form("regulatory_circular")
):
    """Uploads a regulatory document (PDF, TXT, MD) and schedules background ingestion."""
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

    background_tasks.add_task(run_background_ingest, temp_file.name, doc_type)

    logger.info(f"Accepted file '{file.filename}' ({file_size} bytes) for background ingestion.")
    return {
        "status": "ACCEPTED",
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
def list_documents():
    """Returns all active documents for the Admin Dashboard."""
    docs = get_all_documents()
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
        result = ingestion_service.replace_document_version(doc_id, Path(temp_file.name))
        return result
    finally:
        Path(temp_file.name).unlink(missing_ok=True)


@app.delete("/api/v1/documents/{doc_id}")
def delete_document_endpoint(doc_id: str):
    """Permanently purges a document across both SQLite and ChromaDB."""
    success = ingestion_service.delete_document_completely(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return {
        "status": "DELETED",
        "message": f"Document '{doc_id}' and all associated vectors and parent chunks were completely purged."
    }


# ==========================================
# 2. COMPLIANCE AUDIT & QUERY API
# ==========================================

@app.post("/api/v1/compliance/query", response_model=QueryResponse)
def query_compliance(request: QueryRequest):
    """Compliance query returning answer, exact citations, and full pipeline trace."""
    set_transaction_id()
    result: ComplianceAnswer = rag_pipeline.query(request.query)
    
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
        citations=citations,
        execution_time_ms=result.execution_time_ms,
        total_sources_retrieved=result.total_sources_retrieved,
        trace=result.trace
    )


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
    """Returns raw rows from SQLite tables for side-by-side frontend inspection."""
    with get_db_connection() as conn:
        docs = conn.execute("SELECT * FROM documents ORDER BY created_at DESC LIMIT 20;").fetchall()
        parents = conn.execute("SELECT * FROM parent_chunks ORDER BY created_at DESC LIMIT 20;").fetchall()
        jobs = conn.execute("SELECT * FROM ingestion_jobs ORDER BY created_at DESC LIMIT 20;").fetchall()

        return {
            "database": "SQLite (Ground Truth Store)",
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
    """Returns raw vector records and metadata payloads from ChromaDB."""
    try:
        collection = rag_pipeline.retriever.vector_manager.collection
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
                # Safely convert numpy array to list if needed
                if hasattr(embed_vector, "tolist"):
                    embed_vector = embed_vector.tolist()

                dim_count = len(embed_vector)
                # Format compact 384-dimensional vector preview
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
        logger.error(f"Failed to inspect ChromaDB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



# ==========================================
# 4. FRONTEND STATIC MOUNT & ROOT
# ==========================================

# Mount frontend directory if it exists
frontend_dir = Path("frontend")
frontend_dir.mkdir(parents=True, exist_ok=True)
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/")
def root_redirect():
    """Root endpoint welcoming user and providing quick links."""
    return {
        "status": "HEALTHY",
        "app": "Banking Regulatory Compliance RAG Engine",
        "dashboard_ui": "http://localhost:8000/app/index.html",
        "swagger_docs": "http://localhost:8000/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
