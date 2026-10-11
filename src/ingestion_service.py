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
)
from src.logger import get_logger, set_transaction_id

logger = get_logger(__name__)


class IngestionService:
    """Enterprise service managing document ingestion and lifecycle workflows."""

    def __init__(self, vector_manager: Optional[VectorStoreManager] = None):
        self.vector_manager = vector_manager or VectorStoreManager()
        self.chunker = HierarchicalChunker()
        logger.info("Ingestion Service initialized.")

    def ingest_document(self, file_path: Path, doc_type: str = "regulatory_circular") -> Dict[str, Any]:
        """
        Executes end-to-end ingestion pipeline:
        1. Hash verification & deduplication
        2. Job creation in SQLite
        3. Streaming parse of pages and Markdown tables
        4. Hierarchical parent-child chunking
        5. Atomic persistence:
           - Insert into 'documents' table (Parent row)
           - Insert into 'parent_chunks' table (Child rows referencing doc_id)
           - Insert into ChromaDB (Vector embeddings)
        """
        path = Path(file_path)
        txn_id = set_transaction_id(f"ingest_{uuid.uuid4().hex[:6]}")
        logger.info(f"Initiating ingestion workflow for file: '{path.name}'")

        # 1. Inspect Document & Compute Hash
        doc_info: ParsedDocumentMetadata = DocumentParser.get_document_info(path)
        
        # Check Deduplication: Has this exact document already been processed?
        existing_doc = get_document_by_hash(doc_info.file_hash)
        if existing_doc:
            logger.info(f"Duplicate document detected: '{path.name}' already indexed with doc_id: {existing_doc['doc_id']}.")
            return {
                "status": "ALREADY_EXISTS",
                "message": "Document with identical SHA-256 hash is already indexed.",
                "doc_id": existing_doc["doc_id"],
                "filename": existing_doc["filename"],
                "version": existing_doc["version"],
            }

        # 2. Register Ingestion Job in SQLite
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        create_job(job_id=job_id, doc_id=doc_id, filename=doc_info.filename)

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                # Stage 1: PARSING
                update_job_status(job_id, status="PARSING", progress=20.0)
                logger.info(f"Parsing '{doc_info.filename}' (Attempt {attempt}/{max_retries})...")
                blocks_generator = DocumentParser.parse_stream(path)

                # Stage 2: CHUNKING
                update_job_status(job_id, status="CHUNKING", progress=50.0)
                logger.info(f"Generating hierarchical chunks for '{doc_id}'...")
                parent_chunks, child_chunks = self.chunker.chunk_document(doc_id, blocks_generator)

                # Stage 3: INSERT DOCUMENT RECORD FIRST (Satisfies SQLite Foreign Key)
                logger.info(f"Registering document record {doc_id} in SQLite...")
                insert_document(
                    doc_id=doc_id,
                    filename=doc_info.filename,
                    file_hash=doc_info.file_hash,
                    total_pages=doc_info.total_pages,
                    file_size=doc_info.file_size_bytes,
                    file_type=doc_info.file_type
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

                # Stage 5: EMBEDDING & VECTOR INDEXING IN CHROMADB
                update_job_status(job_id, status="EMBEDDING", progress=75.0)
                logger.info(f"Embedding and indexing {len(child_chunks)} child vectors in ChromaDB...")
                self.vector_manager.add_child_chunks(child_chunks)

                # Stage 6: COMPLETED
                update_job_status(job_id, status="COMPLETED", progress=100.0)
                logger.info(f"Document '{doc_info.filename}' successfully indexed under doc_id: {doc_id}")

                return {
                    "status": "COMPLETED",
                    "doc_id": doc_id,
                    "job_id": job_id,
                    "filename": doc_info.filename,
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

    def replace_document_version(self, doc_id: str, new_file_path: Path) -> Dict[str, Any]:
        """
        Replaces an existing document with a revised version:
        1. Purges obsolete vector embeddings from ChromaDB.
        2. Purges obsolete parent chunks and updates version in SQLite.
        3. Parses and indexes new document contents under the same doc_id.
        """
        path = Path(new_file_path)
        logger.info(f"Replacing document {doc_id} with revised file: '{path.name}'")

        doc_record = get_document_by_id(doc_id)
        if not doc_record:
            raise ValueError(f"Document with doc_id '{doc_id}' not found.")

        # 1. Purge old vector embeddings from ChromaDB
        self.vector_manager.delete_by_doc_id(doc_id)

        # 2. Inspect new file
        new_doc_info = DocumentParser.get_document_info(path)

        # 3. Atomically update SQLite document record & delete old parent chunks
        new_version = replace_document(
            doc_id=doc_id,
            new_filename=new_doc_info.filename,
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
            "filename": new_doc_info.filename,
            "version": new_version,
            "parent_chunks": len(parent_chunks),
            "child_vectors": len(child_chunks),
        }

    def delete_document_completely(self, doc_id: str) -> bool:
        """
        Cascading complete purge of a document across both ChromaDB and SQLite.
        """
        logger.info(f"Initiating complete deletion of doc_id: {doc_id}")

        # 1. Delete all vector embeddings from ChromaDB
        self.vector_manager.delete_by_doc_id(doc_id)

        # 2. Delete document from SQLite (foreign keys cascade to parent_chunks and jobs)
        success = delete_document(doc_id)
        if success:
            logger.info(f"Document {doc_id} completely purged from SQLite and ChromaDB.")
        return success
