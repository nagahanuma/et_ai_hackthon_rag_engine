"""
Database module for metadata, parent chunks, and background ingestion jobs.
Supports full CRUD lifecycle: Ingest, Track Status, List, Replace, and Cascade Delete.
"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
from src.config import settings

from src.logger import get_logger

logger = get_logger(__name__)

def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to SQLite and ensures foreign keys are enabled."""
    db_path = settings.SQLITE_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row  # Returns rows as dictionary-like objects
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    logger.info("Initializing SQLite database tables and indexes...")
    """Initializes tables for documents, parent chunks, and ingestion jobs."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Documents Table: Stores unique files, versions, and SHA-256 hashes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                file_type TEXT DEFAULT 'pdf', -- 'pdf', 'docx', 'txt', 'md'
                total_pages INTEGER NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                version INTEGER DEFAULT 1,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 2. Parent Chunks Table: Full legal context (referenced by child chunks in Vector DB)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parent_chunks (
                parent_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_type TEXT DEFAULT 'text', -- 'text' or 'table'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
            );
        """)

        # 3. Ingestion Jobs Table: Tracks async upload tasks, timeouts, retries, and errors
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_jobs (
                job_id TEXT PRIMARY KEY,
                doc_id TEXT,
                filename TEXT NOT NULL,
                status TEXT NOT NULL, -- 'PENDING', 'PARSING', 'CHUNKING', 'EMBEDDING', 'COMPLETED', 'FAILED'
                progress_percentage REAL DEFAULT 0.0,
                retry_count INTEGER DEFAULT 0,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Performance Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_parent_doc ON parent_chunks(doc_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(file_hash);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_job_status ON ingestion_jobs(status);")
        conn.commit()
    logger.info("SQLite database initialized successfully.")    


# ==========================================
# DOCUMENT REGISTRY (CRUD OPERATIONS)
# ==========================================

def get_document_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """Checks if a document with the exact SHA-256 hash was already processed."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,)).fetchone()
        return dict(row) if row else None


def get_document_by_id(doc_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single document by its doc_id."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None


def get_all_documents() -> List[Dict[str, Any]]:
    """Retrieves all active documents for the Admin Dashboard."""
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT d.*, COUNT(p.parent_id) AS total_parent_chunks
            FROM documents d
            LEFT JOIN parent_chunks p ON d.doc_id = p.doc_id
            WHERE d.is_active = 1
            GROUP BY d.doc_id
            ORDER BY d.created_at DESC;
        """).fetchall()
        return [dict(r) for r in rows]


def insert_document(doc_id: str, filename: str, file_hash: str, total_pages: int, file_size: int, file_type: str = "pdf") -> None:
    """Registers a new document."""
    logger.info(f"Registering document {doc_id} ('{filename}', {total_pages} pages, {file_size} bytes)")
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO documents (doc_id, filename, file_hash, total_pages, file_size_bytes, file_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_id, filename, file_hash, total_pages, file_size, file_type))
        conn.commit()


def replace_document(doc_id: str, new_filename: str, new_hash: str, new_total_pages: int, new_file_size: int, new_file_type: str = "pdf") -> int:
    """
    Replaces an existing document with a new version.
    Purges old parent chunks and increments the version counter atomically.
    Returns the new version number.
    """
    with get_db_connection() as conn:
        # 1. Fetch current version
        row = conn.execute("SELECT version FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        current_version = row["version"] if row else 1
        new_version = current_version + 1

        # 2. Delete old parent chunks (old vectors in Chroma are purged by the service)
        conn.execute("DELETE FROM parent_chunks WHERE doc_id = ?", (doc_id,))

        # 3. Update document metadata
        conn.execute("""
            UPDATE documents
            SET filename = ?, file_hash = ?, total_pages = ?, file_size_bytes = ?, 
                file_type = ?, version = ?, updated_at = CURRENT_TIMESTAMP
            WHERE doc_id = ?
        """, (new_filename, new_hash, new_total_pages, new_file_size, new_file_type, new_version, doc_id))
        
        conn.commit()
        return new_version


def delete_document(doc_id: str) -> bool:
    logger.info(f"Purging document {doc_id} and cascading associated parent chunks")
    with get_db_connection() as conn:
        cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        conn.commit()
        success = cursor.rowcount > 0
        if success:
            logger.info(f"Successfully deleted document {doc_id}")
        else:
            logger.warning(f"Document {doc_id} not found for deletion")
        return success


# ==========================================
# PARENT CHUNKS OPERATIONS
# ==========================================

def insert_parent_chunks_batch(chunks: List[Dict[str, Any]]) -> None:
    """Inserts parent chunks in a single transaction for maximum speed."""
    with get_db_connection() as conn:
        conn.executemany("""
            INSERT INTO parent_chunks (parent_id, doc_id, page_number, chunk_index, content, content_type)
            VALUES (:parent_id, :doc_id, :page_number, :chunk_index, :content, :content_type)
        """, chunks)
        conn.commit()


def get_parent_chunk_by_id(parent_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves full parent context when a child vector matches."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM parent_chunks WHERE parent_id = ?", (parent_id,)).fetchone()
        return dict(row) if row else None


# ==========================================
# INGESTION JOBS & BACKGROUND MONITORING
# ==========================================

def create_job(job_id: str, doc_id: str, filename: str) -> None:
    """Creates a new background ingestion job with status PENDING."""
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO ingestion_jobs (job_id, doc_id, filename, status, progress_percentage)
            VALUES (?, ?, ?, 'PENDING', 0.0)
        """, (job_id, doc_id, filename))
        conn.commit()


def update_job_status(
    job_id: str, 
    status: str, 
    progress: float, 
    error_message: Optional[str] = None, 
    increment_retry: bool = False
) -> None:
    """Updates job progress and logs retries/errors."""
    with get_db_connection() as conn:
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
        conn.commit()


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetches real-time status of an ingestion job for the frontend polling API."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None
