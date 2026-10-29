"""
Database module for metadata, parent chunks, and background ingestion jobs.
Supports dual backend:
1. Supabase PostgreSQL (with pgvector) for enterprise cloud production.
2. Local SQLite (with ChromaDB) as automatic offline fallback.
"""

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List

import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


def is_postgres() -> bool:
    """Returns True if a PostgreSQL/Supabase connection string is configured."""
    return bool(settings.DATABASE_URL)


def get_clean_pg_url() -> str:
    """Strips query parameters like ?pgbouncer=true which psycopg2 DSN does not accept."""
    if not settings.DATABASE_URL:
        return ""
    return settings.DATABASE_URL.split("?")[0]


@contextmanager
def get_db_connection():
    """
    Context manager yielding a database connection.
    Connects to Supabase PostgreSQL if DATABASE_URL is present, otherwise SQLite.
    """
    if is_postgres():
        conn = psycopg2.connect(get_clean_pg_url())
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        db_path = settings.SQLITE_DB_PATH
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def init_db() -> None:
    """Initializes tables for documents, parent chunks, child_chunks, and ingestion jobs."""
    if is_postgres():
        logger.info("Initializing/Verifying Supabase PostgreSQL schema...")
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Ensure vector extension is active
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Documents Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        doc_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_hash TEXT UNIQUE NOT NULL,
                        file_type TEXT DEFAULT 'pdf',
                        total_pages INTEGER NOT NULL,
                        file_size_bytes BIGINT NOT NULL,
                        version INTEGER DEFAULT 1,
                        is_active INTEGER DEFAULT 1,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                
                # Parent Chunks Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS parent_chunks (
                        parent_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                        page_number INTEGER NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        content_type TEXT DEFAULT 'text',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                
                # Child Chunks Table (Vector Storage)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS child_chunks (
                        child_id TEXT PRIMARY KEY,
                        parent_id TEXT NOT NULL REFERENCES parent_chunks(parent_id) ON DELETE CASCADE,
                        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                        page_number INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding vector(1024) NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                
                # Ingestion Jobs Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ingestion_jobs (
                        job_id TEXT PRIMARY KEY,
                        doc_id TEXT,
                        filename TEXT NOT NULL,
                        status TEXT NOT NULL,
                        progress_percentage REAL DEFAULT 0.0,
                        retry_count INTEGER DEFAULT 0,
                        error_message TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                
                # Chat Sessions & Messages Tables (Conversational Memory)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT DEFAULT 'anonymous',
                        title TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                cursor.execute("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        message_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        citations JSONB DEFAULT '[]'::jsonb,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)

                # Users Table (Authentication, BYOK Keys & Resource Consumption)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        role TEXT DEFAULT 'user',
                        custom_groq_key TEXT,
                        custom_openai_key TEXT,
                        total_queries INTEGER DEFAULT 0,
                        total_tokens INTEGER DEFAULT 0,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        last_active_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)

                # Ensure user_id exists on documents for multi-tenant ownership
                cursor.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE;")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);")
                cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS custom_model TEXT;")

                # Multi-Tenant Document Association Table (Maps users to documents for seamless deduplication & isolation)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_documents (
                        user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                        filename TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (user_id, doc_id)
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_documents_user ON user_documents(user_id);")
                cursor.execute("""
                    INSERT INTO user_documents (user_id, doc_id, filename)
                    SELECT user_id, doc_id, filename FROM documents WHERE user_id IS NOT NULL
                    ON CONFLICT (user_id, doc_id) DO NOTHING;
                """)

                # HNSW Vector Index & Relational Indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_child_vector_hnsw ON child_chunks USING hnsw (embedding vector_cosine_ops);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_parent_doc ON parent_chunks(doc_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(file_hash);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_job_status ON ingestion_jobs(status);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_msg_session ON chat_messages(session_id, created_at ASC);")
                cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username);")
                cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email);")
        logger.info("Supabase PostgreSQL database verified successfully.")
    else:
        logger.info("Initializing SQLite database tables and indexes...")
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL,
                    file_type TEXT DEFAULT 'pdf',
                    total_pages INTEGER NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    version INTEGER DEFAULT 1,
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    parent_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_type TEXT DEFAULT 'text',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress_percentage REAL DEFAULT 0.0,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT DEFAULT 'anonymous',
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id) ON DELETE CASCADE
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    custom_groq_key TEXT,
                    custom_openai_key TEXT,
                    total_queries INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Ensure user_id exists on documents for multi-tenant ownership
            cols = [c[1] for c in cursor.execute("PRAGMA table_info(documents)").fetchall()]
            if "user_id" not in cols:
                cursor.execute("ALTER TABLE documents ADD COLUMN user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE;")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);")

            # Multi-Tenant Document Association Table in SQLite
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_documents (
                    user_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, doc_id),
                    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
                    FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_documents_user ON user_documents(user_id);")
            cursor.execute("""
                INSERT OR IGNORE INTO user_documents (user_id, doc_id, filename)
                SELECT user_id, doc_id, filename FROM documents WHERE user_id IS NOT NULL;
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_parent_doc ON parent_chunks(doc_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(file_hash);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_job_status ON ingestion_jobs(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_msg_session ON chat_messages(session_id, created_at ASC);")
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username);")
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email);")
            user_cols = [c[1] for c in cursor.execute("PRAGMA table_info(users)").fetchall()]
            if "custom_model" not in user_cols:
                cursor.execute("ALTER TABLE users ADD COLUMN custom_model TEXT;")
        logger.info("SQLite database initialized successfully.")

    # Automatically seed/verify single Admin user from .env credentials
    seed_admin_user()


# ==========================================
# DOCUMENT REGISTRY (CRUD OPERATIONS)
# ==========================================

def get_document_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """Checks if a document with the exact SHA-256 hash was already processed."""
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM documents WHERE file_hash = %s;", (file_hash,))
                row = cur.fetchone()
                return dict(row) if row else None
        else:
            row = conn.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,)).fetchone()
            return dict(row) if row else None


def get_document_by_id(doc_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single document by its doc_id."""
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM documents WHERE doc_id = %s;", (doc_id,))
                row = cur.fetchone()
                return dict(row) if row else None
        else:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            return dict(row) if row else None


def check_user_has_document(user_id: str, doc_id: str) -> bool:
    """Checks if a user already owns or is linked to this doc_id."""
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 1 FROM user_documents WHERE user_id = %s AND doc_id = %s
                    UNION
                    SELECT 1 FROM documents WHERE user_id = %s AND doc_id = %s;
                """, (user_id, doc_id, user_id, doc_id))
                return cur.fetchone() is not None
        else:
            row = conn.execute("""
                SELECT 1 FROM user_documents WHERE user_id = ? AND doc_id = ?
                UNION
                SELECT 1 FROM documents WHERE user_id = ? AND doc_id = ?
            """, (user_id, doc_id, user_id, doc_id)).fetchone()
            return row is not None


def link_user_to_document(user_id: str, doc_id: str, filename: str) -> None:
    """Links an existing document to a user via fast SHA-256 deduplication."""
    logger.info(f"Linking user '{user_id}' to document '{doc_id}' ('{filename}')")
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_documents (user_id, doc_id, filename)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, doc_id) DO NOTHING;
                """, (user_id, doc_id, filename))
        else:
            conn.execute("""
                INSERT OR IGNORE INTO user_documents (user_id, doc_id, filename)
                VALUES (?, ?, ?)
            """, (user_id, doc_id, filename))


def get_all_documents(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves active documents. If user_id is provided, filters strictly for that user."""
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if user_id:
                    query = """
                        SELECT 
                            d.doc_id,
                            COALESCE(ud.filename, d.filename) AS filename,
                            d.file_hash,
                            d.file_type,
                            d.total_pages,
                            d.file_size_bytes,
                            d.version,
                            d.is_active,
                            COALESCE(ud.created_at, d.created_at) AS created_at,
                            d.updated_at,
                            COALESCE(ud.user_id, d.user_id) AS user_id,
                            COUNT(p.parent_id) AS total_parent_chunks
                        FROM documents d
                        LEFT JOIN user_documents ud ON d.doc_id = ud.doc_id AND ud.user_id = %s
                        LEFT JOIN parent_chunks p ON d.doc_id = p.doc_id
                        WHERE d.is_active = 1 AND (ud.user_id = %s OR d.user_id = %s)
                        GROUP BY d.doc_id, ud.filename, ud.created_at, ud.user_id
                        ORDER BY created_at DESC;
                    """
                    cur.execute(query, (user_id, user_id, user_id))
                else:
                    query = """
                        SELECT d.*, COUNT(p.parent_id) AS total_parent_chunks
                        FROM documents d
                        LEFT JOIN parent_chunks p ON d.doc_id = p.doc_id
                        WHERE d.is_active = 1
                        GROUP BY d.doc_id
                        ORDER BY d.created_at DESC;
                    """
                    cur.execute(query)
                rows = cur.fetchall()
                docs = []
                for r in rows:
                    d = dict(r)
                    if d.get("created_at"):
                        d["created_at"] = d["created_at"].isoformat()
                    if d.get("updated_at"):
                        d["updated_at"] = d["updated_at"].isoformat()
                    docs.append(d)
                return docs
        else:
            if user_id:
                query = """
                    SELECT 
                        d.doc_id,
                        COALESCE(ud.filename, d.filename) AS filename,
                        d.file_hash,
                        d.file_type,
                        d.total_pages,
                        d.file_size_bytes,
                        d.version,
                        d.is_active,
                        COALESCE(ud.created_at, d.created_at) AS created_at,
                        d.updated_at,
                        COALESCE(ud.user_id, d.user_id) AS user_id,
                        COUNT(p.parent_id) AS total_parent_chunks
                    FROM documents d
                    LEFT JOIN user_documents ud ON d.doc_id = ud.doc_id AND ud.user_id = ?
                    LEFT JOIN parent_chunks p ON d.doc_id = p.doc_id
                    WHERE d.is_active = 1 AND (ud.user_id = ? OR d.user_id = ?)
                    GROUP BY d.doc_id, ud.filename, ud.created_at, ud.user_id
                    ORDER BY created_at DESC;
                """
                rows = conn.execute(query, (user_id, user_id, user_id)).fetchall()
            else:
                query = """
                    SELECT d.*, COUNT(p.parent_id) AS total_parent_chunks
                    FROM documents d
                    LEFT JOIN parent_chunks p ON d.doc_id = p.doc_id
                    WHERE d.is_active = 1
                    GROUP BY d.doc_id
                    ORDER BY d.created_at DESC;
                """
                rows = conn.execute(query).fetchall()
            docs = []
            for r in rows:
                d = dict(r)
                if d.get("created_at") and not str(d["created_at"]).endswith("Z"):
                    d["created_at"] = str(d["created_at"]).replace(" ", "T") + "Z"
                if d.get("updated_at") and not str(d["updated_at"]).endswith("Z"):
                    d["updated_at"] = str(d["updated_at"]).replace(" ", "T") + "Z"
                docs.append(d)
            return docs


def get_user_doc_ids(user_id: str) -> List[str]:
    """Retrieves all active doc_ids owned by or linked to a user."""
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT doc_id FROM user_documents WHERE user_id = %s
                    UNION
                    SELECT doc_id FROM documents WHERE user_id = %s AND is_active = 1;
                """, (user_id, user_id))
                return [r[0] for r in cur.fetchall()]
        else:
            rows = conn.execute("""
                SELECT doc_id FROM user_documents WHERE user_id = ?
                UNION
                SELECT doc_id FROM documents WHERE user_id = ? AND is_active = 1
            """, (user_id, user_id)).fetchall()
            return [r[0] for r in rows]


def insert_document(doc_id: str, filename: str, file_hash: str, total_pages: int, file_size: int, file_type: str = "pdf", user_id: Optional[str] = None) -> None:
    """Registers a new document with user ownership."""
    logger.info(f"Registering document {doc_id} ('{filename}', {total_pages} pages, {file_size} bytes, user: {user_id})")
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO documents (doc_id, filename, file_hash, total_pages, file_size_bytes, file_type, user_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                """, (doc_id, filename, file_hash, total_pages, file_size, file_type, user_id))
                if user_id:
                    cur.execute("""
                        INSERT INTO user_documents (user_id, doc_id, filename)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (user_id, doc_id) DO NOTHING;
                    """, (user_id, doc_id, filename))
        else:
            conn.execute("""
                INSERT INTO documents (doc_id, filename, file_hash, total_pages, file_size_bytes, file_type, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (doc_id, filename, file_hash, total_pages, file_size, file_type, user_id))
            if user_id:
                conn.execute("""
                    INSERT OR IGNORE INTO user_documents (user_id, doc_id, filename)
                    VALUES (?, ?, ?)
                """, (user_id, doc_id, filename))


def replace_document(doc_id: str, new_filename: str, new_hash: str, new_total_pages: int, new_file_size: int, new_file_type: str = "pdf") -> int:
    """Replaces an existing document with a new version atomically."""
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT version FROM documents WHERE doc_id = %s;", (doc_id,))
                row = cur.fetchone()
                current_version = row["version"] if row else 1
                new_version = current_version + 1

                # Delete old parent chunks (cascades to child_chunks)
                cur.execute("DELETE FROM parent_chunks WHERE doc_id = %s;", (doc_id,))
                cur.execute("""
                    UPDATE documents
                    SET filename = %s, file_hash = %s, total_pages = %s, file_size_bytes = %s, 
                        file_type = %s, version = %s, updated_at = NOW()
                    WHERE doc_id = %s;
                """, (new_filename, new_hash, new_total_pages, new_file_size, new_file_type, new_version, doc_id))
                return new_version
        else:
            row = conn.execute("SELECT version FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            current_version = row["version"] if row else 1
            new_version = current_version + 1

            conn.execute("DELETE FROM parent_chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("""
                UPDATE documents
                SET filename = ?, file_hash = ?, total_pages = ?, file_size_bytes = ?, 
                    file_type = ?, version = ?, updated_at = CURRENT_TIMESTAMP
                WHERE doc_id = ?
            """, (new_filename, new_hash, new_total_pages, new_file_size, new_file_type, new_version, doc_id))
            return new_version


def delete_document(doc_id: str, user_id: Optional[str] = None) -> bool:
    """Purges document or unlinks user. Cascades when no other users reference it."""
    logger.info(f"Purging document {doc_id} (requested by user: {user_id or 'admin/global'})")
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor() as cur:
                if user_id:
                    # Unlink from user_documents
                    cur.execute("DELETE FROM user_documents WHERE doc_id = %s AND user_id = %s;", (doc_id, user_id))
                    # Check if any other user or owner still references this doc
                    cur.execute("SELECT COUNT(*) FROM user_documents WHERE doc_id = %s;", (doc_id,))
                    count = cur.fetchone()[0]
                    cur.execute("SELECT user_id FROM documents WHERE doc_id = %s;", (doc_id,))
                    owner_row = cur.fetchone()
                    owner_id = owner_row[0] if owner_row else None
                    if count == 0 and (owner_id is None or owner_id == user_id):
                        cur.execute("DELETE FROM documents WHERE doc_id = %s;", (doc_id,))
                    return True
                else:
                    cur.execute("DELETE FROM documents WHERE doc_id = %s;", (doc_id,))
                    return cur.rowcount > 0
        else:
            if user_id:
                conn.execute("DELETE FROM user_documents WHERE doc_id = ? AND user_id = ?", (doc_id, user_id))
                row = conn.execute("SELECT COUNT(*) FROM user_documents WHERE doc_id = ?", (doc_id,)).fetchone()
                count = row[0] if row else 0
                owner_row = conn.execute("SELECT user_id FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
                owner_id = owner_row[0] if owner_row else None
                if count == 0 and (owner_id is None or owner_id == user_id):
                    conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
                return True
            else:
                cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
                return cursor.rowcount > 0


# ==========================================
# PARENT CHUNKS OPERATIONS
# ==========================================

def insert_parent_chunks_batch(chunks: List[Dict[str, Any]]) -> None:
    """Inserts parent chunks in a batch transaction for maximum speed."""
    if not chunks:
        return
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor() as cur:
                execute_batch(cur, """
                    INSERT INTO parent_chunks (parent_id, doc_id, page_number, chunk_index, content, content_type)
                    VALUES (%(parent_id)s, %(doc_id)s, %(page_number)s, %(chunk_index)s, %(content)s, %(content_type)s);
                """, chunks)
        else:
            conn.executemany("""
                INSERT INTO parent_chunks (parent_id, doc_id, page_number, chunk_index, content, content_type)
                VALUES (:parent_id, :doc_id, :page_number, :chunk_index, :content, :content_type)
            """, chunks)


def get_parent_chunk_by_id(parent_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves full parent context when a child vector matches."""
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM parent_chunks WHERE parent_id = %s;", (parent_id,))
                row = cur.fetchone()
                return dict(row) if row else None
        else:
            row = conn.execute("SELECT * FROM parent_chunks WHERE parent_id = ?", (parent_id,)).fetchone()
            return dict(row) if row else None


# ==========================================
# SUPABASE PGVECTOR (CHILD CHUNKS) OPERATIONS
# ==========================================

def insert_child_chunks_pgvector(chunks: List[Dict[str, Any]]) -> None:
    """
    Inserts child chunks with 1024-d vector embeddings directly into Supabase pgvector.
    Expected chunk keys: child_id, parent_id, doc_id, page_number, content, embedding (list[float]).
    """
    if not chunks or not is_postgres():
        return
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            execute_batch(cur, """
                INSERT INTO child_chunks (child_id, parent_id, doc_id, page_number, content, embedding)
                VALUES (%(child_id)s, %(parent_id)s, %(doc_id)s, %(page_number)s, %(content)s, %(embedding)s::vector)
                ON CONFLICT (child_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding;
            """, chunks)


def search_child_chunks_pgvector(
    query_vector: List[float], 
    top_k: int = 20, 
    allowed_doc_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Performs Cosine Similarity vector search (<=> operator) against Supabase pgvector with HNSW index.
    Restricts search to allowed_doc_ids if provided (multi-tenant isolation).
    Returns matching child chunks ordered by cosine score descending.
    """
    if not is_postgres():
        return []
    if allowed_doc_ids is not None and len(allowed_doc_ids) == 0:
        return []

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if allowed_doc_ids is not None:
                cur.execute("""
                    SELECT 
                        child_id, 
                        parent_id, 
                        doc_id, 
                        page_number, 
                        content,
                        (1 - (embedding <=> %s::vector)) AS score
                    FROM child_chunks
                    WHERE doc_id = ANY(%s)
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (query_vector, allowed_doc_ids, query_vector, top_k))
            else:
                cur.execute("""
                    SELECT 
                        child_id, 
                        parent_id, 
                        doc_id, 
                        page_number, 
                        content,
                        (1 - (embedding <=> %s::vector)) AS score
                    FROM child_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (query_vector, query_vector, top_k))
            rows = cur.fetchall()
            return [dict(r) for r in rows]


def get_all_child_chunks_for_inspection(limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieves sample child vector records for frontend Storage Inspector."""
    if not is_postgres():
        return []
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT child_id, parent_id, doc_id, page_number, content, embedding::text
                FROM child_chunks
                LIMIT %s;
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]


# ==========================================
# INGESTION JOBS & BACKGROUND MONITORING
# ==========================================

def create_job(job_id: str, doc_id: str, filename: str) -> None:
    """Creates a new background ingestion job with status PENDING."""
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ingestion_jobs (job_id, doc_id, filename, status, progress_percentage)
                    VALUES (%s, %s, %s, 'PENDING', 0.0);
                """, (job_id, doc_id, filename))
        else:
            conn.execute("""
                INSERT INTO ingestion_jobs (job_id, doc_id, filename, status, progress_percentage)
                VALUES (?, ?, ?, 'PENDING', 0.0)
            """, (job_id, doc_id, filename))


def update_job_status(
    job_id: str, 
    status: str, 
    progress: float, 
    error_message: Optional[str] = None, 
    increment_retry: bool = False
) -> None:
    """Updates job progress and logs retries/errors."""
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor() as cur:
                if increment_retry:
                    cur.execute("""
                        UPDATE ingestion_jobs
                        SET status = %s, progress_percentage = %s, error_message = %s, 
                            retry_count = retry_count + 1, updated_at = NOW()
                        WHERE job_id = %s;
                    """, (status, progress, error_message, job_id))
                else:
                    cur.execute("""
                        UPDATE ingestion_jobs
                        SET status = %s, progress_percentage = %s, error_message = %s, updated_at = NOW()
                        WHERE job_id = %s;
                    """, (status, progress, error_message, job_id))
        else:
            if increment_retry:
                conn.execute("""
                    UPDATE ingestion_jobs
                    SET status = ?, progress_percentage = ?, error_message = ?, 
                        retry_count = retry_count + 1, updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ?
                """, (status, progress, error_message, job_id))
            else:
                conn.execute("""
                    UPDATE ingestion_jobs
                    SET status = ?, progress_percentage = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ?
                """, (status, progress, error_message, job_id))


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetches real-time status of an ingestion job for the frontend polling API."""
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM ingestion_jobs WHERE job_id = %s;", (job_id,))
                row = cur.fetchone()
                return dict(row) if row else None
        else:
            row = conn.execute("SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,)).fetchone()
            return dict(row) if row else None


# ==========================================
# CONVERSATIONAL CHAT HISTORY (MULTI-TURN)
# ==========================================

def create_or_get_chat_session(
    session_id: str, 
    title: Optional[str] = None, 
    user_id: str = "anonymous"
) -> str:
    """
    Ensures a chat session exists in the database.
    If it doesn't exist, creates it. Returns session_id.
    """
    session_title = title or f"Session {session_id[:8]}"
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_sessions (session_id, user_id, title)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE SET 
                        user_id = CASE WHEN chat_sessions.user_id = 'anonymous' AND EXCLUDED.user_id != 'anonymous' THEN EXCLUDED.user_id ELSE chat_sessions.user_id END,
                        title = CASE WHEN LEFT(chat_sessions.title, 8) = 'Session ' AND LEFT(EXCLUDED.title, 8) != 'Session ' THEN EXCLUDED.title ELSE chat_sessions.title END,
                        updated_at = NOW();
                """, (session_id, user_id, session_title))
        else:
            conn.execute("""
                INSERT INTO chat_sessions (session_id, user_id, title)
                VALUES (?, ?, ?)
                ON CONFLICT (session_id) DO UPDATE SET 
                    user_id = CASE WHEN chat_sessions.user_id = 'anonymous' AND excluded.user_id != 'anonymous' THEN excluded.user_id ELSE chat_sessions.user_id END,
                    title = CASE WHEN chat_sessions.title LIKE 'Session %' AND excluded.title NOT LIKE 'Session %' THEN excluded.title ELSE chat_sessions.title END,
                    updated_at = CURRENT_TIMESTAMP;
            """, (session_id, user_id, session_title))
    return session_id


def save_chat_message(
    session_id: str, 
    role: str, 
    content: str, 
    citations: Optional[List[Dict[str, Any]]] = None,
    message_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> str:
    """
    Saves a user query or final LLM answer to the database.
    Stores ONLY the user query and the final generated answer (no supporting points/citations).
    Automatically trims conversation history so only the 30 most recent messages are kept.
    Returns the message_id.
    """
    chat_title = (content[:50] + "...") if role == "user" else None
    create_or_get_chat_session(session_id, title=chat_title, user_id=user_id or "anonymous")
    msg_id = message_id or f"msg_{uuid.uuid4().hex[:12]}"
    # User requirement: Store user query and final answer only, without supporting citations
    citations_json = "[]"

    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_messages (message_id, session_id, role, content, citations)
                    VALUES (%s, %s, %s, %s, %s::jsonb);
                """, (msg_id, session_id, role, content, citations_json))
                cur.execute("""
                    UPDATE chat_sessions
                    SET updated_at = NOW()
                    WHERE session_id = %s;
                """, (session_id,))
                # Automatically trim conversation history to max 30 messages (delete oldest)
                cur.execute("""
                    DELETE FROM chat_messages
                    WHERE session_id = %s
                      AND message_id NOT IN (
                          SELECT message_id
                          FROM chat_messages
                          WHERE session_id = %s
                          ORDER BY created_at DESC, message_id DESC
                          LIMIT 30
                      );
                """, (session_id, session_id))
        else:
            conn.execute("""
                INSERT INTO chat_messages (message_id, session_id, role, content, citations)
                VALUES (?, ?, ?, ?, ?)
            """, (msg_id, session_id, role, content, citations_json))
            conn.execute("""
                UPDATE chat_sessions
                SET updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """, (session_id,))
            # Automatically trim conversation history in SQLite to max 30 messages
            conn.execute("""
                DELETE FROM chat_messages
                WHERE session_id = ?
                  AND message_id NOT IN (
                      SELECT message_id
                      FROM chat_messages
                      WHERE session_id = ?
                      ORDER BY created_at DESC, message_id DESC
                      LIMIT 30
                  );
            """, (session_id, session_id))
    return msg_id


def get_recent_chat_history(session_id: str, limit: int = 6) -> List[Dict[str, Any]]:
    """
    Retrieves the most recent N messages for context injection into RAG prompt.
    Returns in chronological order (oldest to newest among the recent window).
    """
    query = """
        SELECT role, content
        FROM (
            SELECT role, content, created_at, message_id
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY created_at DESC, message_id DESC
            LIMIT %s
        ) sub
        ORDER BY created_at ASC, message_id ASC;
    """
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (session_id, limit))
                return [dict(r) for r in cur.fetchall()]
        else:
            sqlite_query = """
                SELECT role, content
                FROM (
                    SELECT role, content, created_at, message_id
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC, message_id DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC, message_id ASC;
            """
            rows = conn.execute(sqlite_query, (session_id, limit)).fetchall()
            return [dict(r) for r in rows]


def get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    """
    Fetches all messages for a specific chat session with parsed citations.
    """
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT message_id, session_id, role, content, citations, created_at
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY created_at ASC, message_id ASC;
                """, (session_id,))
                rows = cur.fetchall()
                messages = []
                for r in rows:
                    m = dict(r)
                    if m.get("created_at"):
                        m["created_at"] = m["created_at"].isoformat()
                    if isinstance(m.get("citations"), str):
                        try:
                            m["citations"] = json.loads(m["citations"])
                        except Exception:
                            m["citations"] = []
                    messages.append(m)
                return messages
        else:
            rows = conn.execute("""
                SELECT message_id, session_id, role, content, citations, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY created_at ASC, message_id ASC;
            """, (session_id,)).fetchall()
            messages = []
            for r in rows:
                m = dict(r)
                if isinstance(m.get("citations"), str):
                    try:
                        m["citations"] = json.loads(m["citations"])
                    except Exception:
                        m["citations"] = []
                messages.append(m)
            return messages


def list_chat_sessions(user_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Lists recent chat sessions for session switching or history UI."""
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if user_id:
                    cur.execute("""
                        SELECT s.*, COUNT(m.message_id) AS total_messages
                        FROM chat_sessions s
                        LEFT JOIN chat_messages m ON s.session_id = m.session_id
                        WHERE s.user_id = %s
                        GROUP BY s.session_id
                        ORDER BY s.updated_at DESC
                        LIMIT %s;
                    """, (user_id, limit))
                else:
                    cur.execute("""
                        SELECT s.*, COUNT(m.message_id) AS total_messages
                        FROM chat_sessions s
                        LEFT JOIN chat_messages m ON s.session_id = m.session_id
                        GROUP BY s.session_id
                        ORDER BY s.updated_at DESC
                        LIMIT %s;
                    """, (limit,))
                rows = cur.fetchall()
                sessions = []
                for r in rows:
                    s = dict(r)
                    if s.get("created_at"):
                        s["created_at"] = s["created_at"].isoformat()
                    if s.get("updated_at"):
                        s["updated_at"] = s["updated_at"].isoformat()
                    sessions.append(s)
                return sessions
        else:
            if user_id:
                rows = conn.execute("""
                    SELECT s.*, COUNT(m.message_id) AS total_messages
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON s.session_id = m.session_id
                    WHERE s.user_id = ?
                    GROUP BY s.session_id
                    ORDER BY s.updated_at DESC
                    LIMIT ?
                """, (user_id, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT s.*, COUNT(m.message_id) AS total_messages
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON s.session_id = m.session_id
                    GROUP BY s.session_id
                    ORDER BY s.updated_at DESC
                    LIMIT ?
                """, (limit,)).fetchall()
            return [dict(r) for r in rows]


def delete_chat_session(session_id: str) -> bool:
    """Deletes a chat session and cascades all messages."""
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_sessions WHERE session_id = %s;", (session_id,))
                return cur.rowcount > 0
        else:
            cursor = conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
            return cursor.rowcount > 0


def delete_all_chat_sessions(user_id: str) -> int:
    """Purges all chat sessions and messages for a specific user to reclaim database storage."""
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_sessions WHERE user_id = %s;", (user_id,))
                return cur.rowcount
        else:
            cursor = conn.execute("DELETE FROM chat_sessions WHERE user_id = ?", (user_id,))
            return cursor.rowcount


# ==========================================
# AUTHENTICATION, SINGLE ADMIN & BYOK KEYS
# ==========================================

def hash_password(password: str) -> str:
    """Computes SHA-256 hash using application SECRET_KEY as salt."""
    salt = getattr(settings, "SECRET_KEY", "compliance_salt_2026")
    return hashlib.sha256(f"{password}:{salt}".encode()).hexdigest()


def seed_admin_user() -> None:
    """
    Ensures the single primary administrator account exists.
    Configured via settings.ADMIN_USERNAME / settings.ADMIN_PASSWORD in .env.
    """
    admin_user = settings.ADMIN_USERNAME.strip().lower()
    admin_email = settings.ADMIN_EMAIL.strip().lower()
    pwd_hash = hash_password(settings.ADMIN_PASSWORD)

    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO users (user_id, username, email, password_hash, role)
                    VALUES ('usr_admin_master', %s, %s, %s, 'admin')
                    ON CONFLICT (username) DO UPDATE SET
                        password_hash = EXCLUDED.password_hash,
                        email = EXCLUDED.email,
                        role = 'admin';
                """, (admin_user, admin_email, pwd_hash))
        else:
            row = conn.execute("SELECT user_id FROM users WHERE username = ? OR email = ?", (admin_user, admin_email)).fetchone()
            if not row:
                conn.execute("""
                    INSERT INTO users (user_id, username, email, password_hash, role)
                    VALUES ('usr_admin_master', ?, ?, ?, 'admin')
                """, (admin_user, admin_email, pwd_hash))
            else:
                conn.execute("""
                    UPDATE users
                    SET password_hash = ?, email = ?, role = 'admin'
                    WHERE username = ?;
                """, (pwd_hash, admin_email, admin_user))
    logger.info(f"Verified Primary Admin account: '{admin_user}' (Role: admin)")


def register_user(username: str, email: str, password: str) -> Dict[str, Any]:
    """
    Registers a new standard user (no email verification required for local testing).
    """
    user_name = username.strip().lower()
    user_email = email.strip().lower()
    if not user_name or not password:
        raise ValueError("Username and password are required.")

    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT user_id FROM users WHERE username = %s OR email = %s;", (user_name, user_email))
                if cur.fetchone():
                    raise ValueError(f"User with username '{user_name}' or email '{user_email}' already exists.")
        else:
            row = conn.execute("SELECT user_id FROM users WHERE username = ? OR email = ?", (user_name, user_email)).fetchone()
            if row:
                raise ValueError(f"User with username '{user_name}' or email '{user_email}' already exists.")

    new_id = f"usr_{uuid.uuid4().hex[:12]}"
    pwd_hash = hash_password(password)

    try:
        with get_db_connection() as conn:
            if is_postgres():
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO users (user_id, username, email, password_hash, role)
                        VALUES (%s, %s, %s, %s, 'user');
                    """, (new_id, user_name, user_email, pwd_hash))
            else:
                conn.execute("""
                    INSERT INTO users (user_id, username, email, password_hash, role)
                    VALUES (?, ?, ?, ?, 'user')
                """, (new_id, user_name, user_email, pwd_hash))
    except Exception as e:
        err_str = str(e).lower()
        if "unique" in err_str or "duplicate" in err_str:
            raise ValueError(f"User with username '{user_name}' or email '{user_email}' already exists.")
        raise

    return {
        "user_id": new_id,
        "username": user_name,
        "email": user_email,
        "role": "user",
        "custom_groq_key": None,
        "custom_openai_key": None
    }


def authenticate_user(login_id: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticates a user via username or email and password.
    Returns safe user dict on success, None on failure.
    """
    clean_login = login_id.strip().lower()
    pwd_hash = hash_password(password)

    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT user_id, username, email, role, custom_groq_key, custom_openai_key,
                           total_queries, total_tokens, created_at, last_active_at
                    FROM users
                    WHERE (LOWER(username) = %s OR LOWER(email) = %s)
                      AND password_hash = %s;
                """, (clean_login, clean_login, pwd_hash))
                row = cur.fetchone()
                if row:
                    cur.execute("UPDATE users SET last_active_at = NOW() WHERE user_id = %s;", (row["user_id"],))
                    u = dict(row)
                    if u.get("created_at"): u["created_at"] = u["created_at"].isoformat()
                    if u.get("last_active_at"): u["last_active_at"] = u["last_active_at"].isoformat()
                    return u
        else:
            row = conn.execute("""
                SELECT user_id, username, email, role, custom_groq_key, custom_openai_key,
                       total_queries, total_tokens, created_at, last_active_at
                FROM users
                WHERE (LOWER(username) = ? OR LOWER(email) = ?)
                  AND password_hash = ?
            """, (clean_login, clean_login, pwd_hash)).fetchone()
            if row:
                conn.execute("UPDATE users SET last_active_at = CURRENT_TIMESTAMP WHERE user_id = ?", (row["user_id"],))
                return dict(row)
    return None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves safe user profile by user_id."""
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT user_id, username, email, role, custom_groq_key, custom_openai_key, custom_model,
                           total_queries, total_tokens, created_at, last_active_at
                    FROM users WHERE user_id = %s;
                """, (user_id,))
                row = cur.fetchone()
                if row:
                    u = dict(row)
                    if u.get("created_at"): u["created_at"] = u["created_at"].isoformat()
                    if u.get("last_active_at"): u["last_active_at"] = u["last_active_at"].isoformat()
                    return u
        else:
            row = conn.execute("""
                SELECT user_id, username, email, role, custom_groq_key, custom_openai_key, custom_model,
                       total_queries, total_tokens, created_at, last_active_at
                FROM users WHERE user_id = ?
            """, (user_id,)).fetchone()
            if row:
                return dict(row)
    return None


def update_user_api_keys(
    user_id: str, 
    custom_groq_key: Optional[str] = None, 
    custom_openai_key: Optional[str] = None,
    custom_model: Optional[str] = None
) -> bool:
    """Updates user BYOK API keys and preferred LLM model."""
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE users
                    SET custom_groq_key = %s, custom_openai_key = %s, custom_model = %s, last_active_at = NOW()
                    WHERE user_id = %s;
                """, (custom_groq_key, custom_openai_key, custom_model, user_id))
                return cur.rowcount > 0
        else:
            cur = conn.execute("""
                UPDATE users
                SET custom_groq_key = ?, custom_openai_key = ?, custom_model = ?, last_active_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (custom_groq_key, custom_openai_key, custom_model, user_id))
            return cur.rowcount > 0


def increment_user_usage(user_id: str, queries: int = 1, tokens: int = 0) -> None:
    """Increments a user's total query count and consumed token resource metrics."""
    if not user_id or user_id in ("anonymous", "guest"):
        return
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE users
                    SET total_queries = total_queries + %s,
                        total_tokens = total_tokens + %s,
                        last_active_at = NOW()
                    WHERE user_id = %s;
                """, (queries, tokens, user_id))
        else:
            conn.execute("""
                UPDATE users
                SET total_queries = total_queries + ?,
                    total_tokens = total_tokens + ?,
                    last_active_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (queries, tokens, user_id))


def get_all_users_usage() -> List[Dict[str, Any]]:
    """
    Admin Only: Retrieves all registered users and their exact storage and resource consumption.
    Includes document count, storage bytes (for Supabase Free Tier tracking), vector chunks, and chat messages.
    """
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT u.user_id, u.username, u.email, u.role,
                           (u.custom_groq_key IS NOT NULL AND u.custom_groq_key != '') AS has_custom_groq,
                           (u.custom_openai_key IS NOT NULL AND u.custom_openai_key != '') AS has_custom_openai,
                           u.total_queries, u.total_tokens, u.created_at, u.last_active_at,
                           COALESCE(doc_stats.doc_count, 0) AS total_documents,
                           COALESCE(doc_stats.storage_bytes, 0) AS total_storage_bytes,
                           COALESCE(doc_stats.chunk_count, 0) AS total_chunks,
                           COALESCE(chat_stats.msg_count, 0) AS total_chat_messages
                    FROM users u
                    LEFT JOIN (
                        SELECT ud.user_id,
                               COUNT(DISTINCT ud.doc_id) AS doc_count,
                               COALESCE(SUM(d.file_size_bytes), 0) AS storage_bytes,
                               COUNT(DISTINCT cc.child_id) AS chunk_count
                        FROM user_documents ud
                        JOIN documents d ON ud.doc_id = d.doc_id
                        LEFT JOIN child_chunks cc ON d.doc_id = cc.doc_id
                        GROUP BY ud.user_id
                    ) doc_stats ON u.user_id = doc_stats.user_id
                    LEFT JOIN (
                        SELECT cs.user_id,
                               COUNT(cm.message_id) AS msg_count
                        FROM chat_sessions cs
                        JOIN chat_messages cm ON cs.session_id = cm.session_id
                        GROUP BY cs.user_id
                    ) chat_stats ON u.user_id = chat_stats.user_id
                    ORDER BY total_storage_bytes DESC, u.total_queries DESC, u.created_at DESC;
                """)
                rows = cur.fetchall()
                users = []
                for r in rows:
                    u = dict(r)
                    if u.get("created_at"): u["created_at"] = u["created_at"].isoformat()
                    if u.get("last_active_at"): u["last_active_at"] = u["last_active_at"].isoformat()
                    users.append(u)
                return users
        else:
            rows = conn.execute("""
                SELECT u.user_id, u.username, u.email, u.role,
                       (u.custom_groq_key IS NOT NULL AND u.custom_groq_key != '') AS has_custom_groq,
                       (u.custom_openai_key IS NOT NULL AND u.custom_openai_key != '') AS has_custom_openai,
                       u.total_queries, u.total_tokens, u.created_at, u.last_active_at,
                       COALESCE(doc_stats.doc_count, 0) AS total_documents,
                       COALESCE(doc_stats.storage_bytes, 0) AS total_storage_bytes,
                       COALESCE(doc_stats.chunk_count, 0) AS total_chunks,
                       COALESCE(chat_stats.msg_count, 0) AS total_chat_messages
                FROM users u
                LEFT JOIN (
                    SELECT ud.user_id,
                           COUNT(DISTINCT ud.doc_id) AS doc_count,
                           COALESCE(SUM(d.file_size_bytes), 0) AS storage_bytes,
                           COUNT(DISTINCT cc.child_id) AS chunk_count
                    FROM user_documents ud
                    JOIN documents d ON ud.doc_id = d.doc_id
                    LEFT JOIN child_chunks cc ON d.doc_id = cc.doc_id
                    GROUP BY ud.user_id
                ) doc_stats ON u.user_id = doc_stats.user_id
                LEFT JOIN (
                    SELECT cs.user_id,
                           COUNT(cm.message_id) AS msg_count
                    FROM chat_sessions cs
                    JOIN chat_messages cm ON cs.session_id = cm.session_id
                    GROUP BY cs.user_id
                ) chat_stats ON u.user_id = chat_stats.user_id
                ORDER BY total_storage_bytes DESC, u.total_queries DESC, u.created_at DESC;
            """).fetchall()
            return [dict(r) for r in rows]


def delete_user_and_storage(admin_user_id: str, target_user_id: str) -> Dict[str, Any]:
    """
    Admin Only: Deletes a specific user and reclaims all their database storage.
    Purges any documents exclusively owned by this user (cascading chunks and vector embeddings),
    removes their chat sessions/messages, and deletes the user record.
    The primary Administrator account cannot be deleted.
    """
    admin = get_user_by_id(admin_user_id)
    if not admin or admin.get("role") != "admin":
        raise ValueError("Unauthorized: Only an Administrator can delete user accounts.")
    
    target = get_user_by_id(target_user_id)
    if not target:
        raise ValueError("User not found.")
    
    if target_user_id == "usr_admin_master" or target.get("role") == "admin":
        raise ValueError("Protected Account: Administrator accounts cannot be deleted.")

    # 1. Fetch all documents associated with this user
    doc_ids = get_user_doc_ids(target_user_id)
    purged_docs = 0
    reclaimed_bytes = 0

    from src.ingestion_service import ingestion_service
    for doc_id in doc_ids:
        doc = get_document_by_id(doc_id)
        if doc:
            reclaimed_bytes += doc.get("file_size_bytes", 0)
        # delete_document_completely unlinks user and drops doc + vectors if no other user references it
        ingestion_service.delete_document_completely(doc_id, user_id=target_user_id)
        purged_docs += 1

    # 2. Delete user and their chat sessions
    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_sessions WHERE user_id = %s;", (target_user_id,))
                cur.execute("DELETE FROM users WHERE user_id = %s;", (target_user_id,))
        else:
            conn.execute("DELETE FROM chat_sessions WHERE user_id = ?", (target_user_id,))
            conn.execute("DELETE FROM users WHERE user_id = ?", (target_user_id,))

    logger.info(f"Admin '{admin.get('username')}' purged user '{target.get('username')}' (Reclaimed ~{reclaimed_bytes} bytes).")
    return {
        "status": "SUCCESS",
        "deleted_user_id": target_user_id,
        "username": target.get("username"),
        "purged_documents": purged_docs,
        "reclaimed_bytes": reclaimed_bytes
    }


def cleanup_inactive_users(days_inactive: int = 90) -> Dict[str, Any]:
    """
    Automated Data Retention & Storage Optimization:
    Purges non-admin users who have been inactive for more than `days_inactive` days.
    Returns the count of purged users and the list of document IDs to prune from vector store.
    The primary Administrator account ('nagahanuma') is permanently exempt.
    """
    logger.info(f"Running automated cleanup for users inactive for >{days_inactive} days...")
    purged_user_ids = []
    purged_doc_ids = []

    with get_db_connection() as conn:
        if is_postgres():
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT user_id, username FROM users
                    WHERE role != 'admin'
                      AND last_active_at < NOW() - (%s || ' days')::INTERVAL;
                """, (days_inactive,))
                inactive_users = cur.fetchall()
                if not inactive_users:
                    logger.info("No inactive users found exceeding retention threshold.")
                    return {"purged_users": 0, "purged_docs": 0, "doc_ids": []}

                purged_user_ids = [u["user_id"] for u in inactive_users]

                # Find all documents owned by these users
                cur.execute("SELECT doc_id FROM documents WHERE user_id = ANY(%s);", (purged_user_ids,))
                purged_doc_ids = [r["doc_id"] for r in cur.fetchall()]

                # Delete users (Cascades to documents, parent_chunks, child_chunks, chat_messages)
                cur.execute("DELETE FROM users WHERE user_id = ANY(%s);", (purged_user_ids,))
        else:
            cur = conn.cursor()
            query = f"""
                SELECT user_id, username FROM users
                WHERE role != 'admin'
                  AND last_active_at < datetime('now', '-{int(days_inactive)} days')
            """
            inactive_users = cur.execute(query).fetchall()
            if not inactive_users:
                logger.info("No inactive users found exceeding retention threshold.")
                return {"purged_users": 0, "purged_docs": 0, "doc_ids": []}

            purged_user_ids = [u["user_id"] for u in inactive_users]
            placeholders = ",".join("?" for _ in purged_user_ids)
            doc_rows = cur.execute(f"SELECT doc_id FROM documents WHERE user_id IN ({placeholders})", purged_user_ids).fetchall()
            purged_doc_ids = [r["doc_id"] for r in doc_rows]

            # Delete users (Cascades through SQLite ON DELETE CASCADE)
            cur.execute(f"DELETE FROM users WHERE user_id IN ({placeholders})", purged_user_ids)

    logger.info(f"Storage Cleanup: Purged {len(purged_user_ids)} inactive users and {len(purged_doc_ids)} associated documents.")
    return {
        "purged_users": len(purged_user_ids),
        "purged_docs": len(purged_doc_ids),
        "doc_ids": purged_doc_ids
    }



