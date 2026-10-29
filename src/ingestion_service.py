"""
Document Ingestion & Lifecycle Service.
Coordinates:
- Document Parser (PyMuPDF with streaming and Markdown tables)
- Hierarchical Chunker (Parent-Child)
- SQLite Ground Truth Store (Metadata & Parent Chunks)
- ChromaDB Vector Store (Child Chunk Embeddings)
- Full CRUD Lifecycle: Upload, Track Status, Replace, and Cascade Delete.
"""

import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from src.parser import DocumentParser, ParsedDocumentMetadata
from src.chunker import HierarchicalChunker
from src.vector_store import VectorStoreManager
from src.db import (
    get_document_by_hash,
    get_document_by_id,
    insert_document,
    insert_parent_chunks_batch,
    create_job,
    update_job_status,
    replace_document,
    delete_document,
    cleanup_inactive_users,
    check_user_has_document,
    link_user_to_document,
)
from src.logger import get_logger, set_transaction_id

logger = get_logger(__name__)


class IngestionService:
    """Enterprise service managing document ingestion and lifecycle workflows."""

    def __init__(self, vector_manager: Optional[VectorStoreManager] = None):
        self.vector_manager = vector_manager or VectorStoreManager()
        self.chunker = HierarchicalChunker()
        logger.info("Ingestion Service initialized.")

    def ingest_document(
        self,
        file_path: Path,
        doc_type: str = "regulatory_circular",
        user_id: Optional[str] = None,
        original_filename: Optional[str] = None,
        job_id: Optional[str] = None,
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end ingestion pipeline:
        1. Hash verification & deduplication
        2. Job creation in SQLite
        3. Streaming parse of pages and Markdown tables
        4. Hierarchical parent-child chunking
        5. Atomic persistence:
           - Insert into 'documents' table (Parent row with user_id)
           - Insert into 'parent_chunks' table (Child rows referencing doc_id)
           - Insert into ChromaDB (Vector embeddings)
        """
        path = Path(file_path)
        display_name = original_filename or path.name
        txn_id = set_transaction_id(f"ingest_{uuid.uuid4().hex[:6]}")
        logger.info(f"Initiating ingestion workflow for file: '{display_name}' (User: {user_id})")

        # 1. Inspect Document & Compute Hash
        doc_info: ParsedDocumentMetadata = DocumentParser.get_document_info(path)
        effective_filename = original_filename or doc_info.filename
        
        # Check Deduplication: Has this exact document already been processed?
        existing_doc = get_document_by_hash(doc_info.file_hash)
        if existing_doc:
            user_has_doc = check_user_has_document(user_id, existing_doc["doc_id"]) if user_id else False
            if user_has_doc:
                logger.info(f"Duplicate document detected: '{display_name}' already owned by user '{user_id}'.")
                if job_id:
                    update_job_status(job_id, status="COMPLETED", progress=100.0, error_message="Document already in your library.")
                return {
                    "status": "ALREADY_EXISTS",
                    "message": "You have already uploaded this document.",
                    "doc_id": existing_doc["doc_id"],
                    "filename": existing_doc["filename"],
                    "version": existing_doc["version"],
                }
            else:
                logger.info(f"Deduplication match: Fast-linking existing doc '{existing_doc['doc_id']}' to user '{user_id}'.")
                if user_id:
                    link_user_to_document(user_id=user_id, doc_id=existing_doc["doc_id"], filename=effective_filename)
                if job_id:
                    update_job_status(job_id, status="COMPLETED", progress=100.0)
                return {
                    "status": "COMPLETED",
                    "message": f"Document '{effective_filename}' instantly added to your library via SHA-256 deduplication.",
                    "doc_id": existing_doc["doc_id"],
                    "filename": effective_filename,
                    "version": existing_doc["version"],
                }

        # 2. Register Ingestion Job if not already provided
        if not doc_id:
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        if not job_id:
            job_id = f"job_{uuid.uuid4().hex[:8]}"
            create_job(job_id=job_id, doc_id=doc_id, filename=effective_filename)

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                # Stage 1: PARSING
                update_job_status(job_id, status="PARSING", progress=20.0)
                logger.info(f"Parsing '{effective_filename}' (Attempt {attempt}/{max_retries})...")
                blocks_generator = DocumentParser.parse_stream(path)

                # Stage 2: CHUNKING
                update_job_status(job_id, status="CHUNKING", progress=50.0)
                logger.info(f"Generating hierarchical chunks for '{doc_id}'...")
                parent_chunks, child_chunks = self.chunker.chunk_document(doc_id, blocks_generator)

                # Stage 3: INSERT DOCUMENT RECORD FIRST (Satisfies Foreign Key with user_id)
                logger.info(f"Registering document record {doc_id} with user_id '{user_id}'...")
                insert_document(
                    doc_id=doc_id,
                    filename=effective_filename,
                    file_hash=doc_info.file_hash,
                    total_pages=doc_info.total_pages,
                    file_size=doc_info.file_size_bytes,
                    file_type=doc_info.file_type,
                    user_id=user_id
                )

                # Stage 4: PERSIST PARENT CHUNKS IN SQLITE
                logger.info(f"Writing {len(parent_chunks)} parent chunks to SQLite...")
                parent_records = [
                    {
                        "parent_id": p.parent_id,
                        "doc_id": p.doc_id,
                        "page_number": p.page_number,
                        "chunk_index": p.chunk_index,
                        "content": p.content,
                        "content_type": p.content_type,
                    }
                    for p in parent_chunks
                ]
                insert_parent_chunks_batch(parent_records)

                # Stage 5: EMBEDDING & VECTOR INDEXING (SUPABASE PGVECTOR / CHROMADB)
                update_job_status(job_id, status="EMBEDDING", progress=70.0)
                logger.info(f"Embedding and indexing {len(child_chunks)} child vectors...")
                def on_embed_progress(pct: float):
                    update_job_status(job_id, status="EMBEDDING", progress=round(pct, 1))

                self.vector_manager.add_child_chunks(child_chunks, progress_callback=on_embed_progress)

                # Stage 6: COMPLETED
                update_job_status(job_id, status="COMPLETED", progress=100.0)
                logger.info(f"Document '{effective_filename}' successfully indexed under doc_id: {doc_id}")

                return {
                    "status": "COMPLETED",
                    "doc_id": doc_id,
                    "job_id": job_id,
                    "filename": effective_filename,
                    "total_pages": doc_info.total_pages,
                    "parent_chunks_created": len(parent_chunks),
                    "child_vectors_indexed": len(child_chunks),
                }

            except Exception as e:
                logger.error(f"Ingestion attempt {attempt} failed for '{doc_info.filename}': {str(e)}", exc_info=True)
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    update_job_status(job_id, status="PARSING", progress=0.0, error_message=str(e), increment_retry=True)
                else:
                    update_job_status(job_id, status="FAILED", progress=0.0, error_message=str(e))
                    raise RuntimeError(f"Ingestion permanently failed after {max_retries} attempts: {str(e)}")

    def replace_document_version(self, doc_id: str, new_file_path: Path, original_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Replaces an existing document with a revised version:
        1. Purges obsolete vector embeddings from ChromaDB.
        2. Purges obsolete parent chunks and updates version in SQLite.
        3. Parses and indexes new document contents under the same doc_id.
        """
        path = Path(new_file_path)
        display_name = original_filename or path.name
        logger.info(f"Replacing document {doc_id} with revised file: '{display_name}'")

        doc_record = get_document_by_id(doc_id)
        if not doc_record:
            raise ValueError(f"Document with doc_id '{doc_id}' not found.")

        # 1. Purge old vector embeddings from ChromaDB
        self.vector_manager.delete_by_doc_id(doc_id)

        # 2. Inspect new file
        new_doc_info = DocumentParser.get_document_info(path)
        effective_filename = original_filename or new_doc_info.filename

        # 3. Atomically update SQLite document record & delete old parent chunks
        new_version = replace_document(
            doc_id=doc_id,
            new_filename=effective_filename,
            new_hash=new_doc_info.file_hash,
            new_total_pages=new_doc_info.total_pages,
            new_file_size=new_doc_info.file_size_bytes,
            new_file_type=new_doc_info.file_type
        )

        # 4. Parse & Chunk new document
        blocks_generator = DocumentParser.parse_stream(path)
        parent_chunks, child_chunks = self.chunker.chunk_document(doc_id, blocks_generator)

        # 5. Insert new parent chunks and child vectors
        parent_records = [
            {
                "parent_id": p.parent_id,
                "doc_id": p.doc_id,
                "page_number": p.page_number,
                "chunk_index": p.chunk_index,
                "content": p.content,
                "content_type": p.content_type,
            }
            for p in parent_chunks
        ]
        insert_parent_chunks_batch(parent_records)
        self.vector_manager.add_child_chunks(child_chunks)

        logger.info(f"Document {doc_id} successfully upgraded to Version {new_version}!")
        return {
            "status": "REPLACED",
            "doc_id": doc_id,
            "filename": effective_filename,
            "version": new_version,
            "parent_chunks": len(parent_chunks),
            "child_vectors": len(child_chunks),
        }

    def delete_document_completely(self, doc_id: str, user_id: Optional[str] = None) -> bool:
        """
        Cascading complete purge of a document across both ChromaDB and relational DB.
        If user_id is provided, unlinks the user and only purges vectors if no other user owns the document.
        """
        logger.info(f"Initiating deletion of doc_id: {doc_id} (User: {user_id or 'admin/global'})")

        success = delete_document(doc_id, user_id=user_id)
        if success:
            # Check if document still exists (e.g. other users still reference it)
            remaining_doc = get_document_by_id(doc_id)
            if not remaining_doc:
                self.vector_manager.delete_by_doc_id(doc_id)
                logger.info(f"Document {doc_id} completely purged from database and pgvector/ChromaDB.")
            else:
                logger.info(f"Document {doc_id} unlinked from user {user_id} (retained in storage for other users).")
        return success

    def cleanup_inactive_users_data(self, days_inactive: int = 90) -> Dict[str, Any]:
        """
        Deletes users inactive for > days_inactive and purges all their vectors from ChromaDB.
        """
        res = cleanup_inactive_users(days_inactive=days_inactive)
        purged_docs = res.get("doc_ids", [])
        for doc_id in purged_docs:
            try:
                self.vector_manager.delete_by_doc_id(doc_id)
            except Exception as e:
                logger.warning(f"Error purging vectors for inactive doc {doc_id}: {e}")
        return res

