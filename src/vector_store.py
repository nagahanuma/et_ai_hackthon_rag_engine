"""
Vector Store module managing vector embeddings and search.
Supports:
1. Supabase pgvector with HNSW indexing for cloud production.
2. ChromaDB as automatic local offline fallback.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import torch

from src.config import settings
from src.chunker import ChildChunk
from src.logger import get_logger
from src.db import is_postgres, insert_child_chunks_pgvector, search_child_chunks_pgvector

# Production CPU optimization: Prevent PyTorch from monopolizing CPU cores, ensuring GIL-free multi-worker concurrency
if getattr(settings, "EMBEDDING_DEVICE", "cpu").lower() == "cpu":
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

logger = get_logger(__name__)


@dataclass
class VectorSearchResult:
    """Represents a matched child chunk from vector search."""
    child_id: str
    parent_id: str
    doc_id: str
    page_number: int
    content: str
    score: float  # Cosine similarity score
    metadata: Dict[str, Any]


class VectorStoreManager:
    """Manages vector storage (Supabase pgvector or ChromaDB) and embedding inference."""

    def __init__(self):
        self.use_postgres = is_postgres()
        self.collection_name = settings.VECTOR_COLLECTION_NAME
        self.client = None
        self.collection = None

        if self.use_postgres:
            logger.info(f"VectorStoreManager: Operating in CLOUD MODE (Supabase pgvector enabled)")
        else:
            db_path = settings.CHROMA_DB_PATH
            db_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(db_path),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Regulatory Compliance Child Chunks"}
            )
            logger.info(f"VectorStoreManager: Operating in LOCAL MODE (ChromaDB collection: '{self.collection_name}')")

        # Load Embedding Model
        self.embedding_model = None
        self._init_embedding_model()

    def _init_embedding_model(self):
        """Loads the embedding model based on settings."""
        if settings.EMBEDDING_PROVIDER == "local":
            logger.info(f"Loading local embedding model: '{settings.EMBEDDING_MODEL}' on device '{settings.EMBEDDING_DEVICE}' (sentence-transformers)...")
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE)
            logger.info(f"Local embedding model '{settings.EMBEDDING_MODEL}' loaded successfully on '{settings.EMBEDDING_DEVICE}'.")
        else:
            logger.warning(f"Configured provider '{settings.EMBEDDING_PROVIDER}'. Ensure API key is configured.")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a list of text strings."""
        if not texts:
            return []
        if settings.EMBEDDING_PROVIDER == "local":
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True  # Enables pure cosine similarity
            )
            return embeddings.tolist()
        else:
            raise NotImplementedError("External embedding provider requires OpenAI adapter.")

    def add_child_chunks(self, chunks: List[ChildChunk], progress_callback: Optional[Any] = None) -> int:
        """
        Batches and inserts child chunks into Supabase pgvector or ChromaDB.
        Returns total chunks added.
        """
        if not chunks:
            return 0

        total_chunks = len(chunks)
        batch_size = settings.EMBEDDING_BATCH_SIZE
        target_store = "Supabase pgvector" if self.use_postgres else f"ChromaDB collection '{self.collection_name}'"
        logger.info(f"Indexing {total_chunks} child chunks into {target_store} (batch size: {batch_size})...")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            documents = [c.content for c in batch]
            embeddings = self.embed_texts(documents)

            if self.use_postgres:
                pg_records = [
                    {
                        "child_id": c.child_id,
                        "parent_id": c.parent_id,
                        "doc_id": c.doc_id,
                        "page_number": c.page_number,
                        "content": c.content,
                        "embedding": emb,
                    }
                    for c, emb in zip(batch, embeddings)
                ]
                insert_child_chunks_pgvector(pg_records)
            else:
                ids = [c.child_id for c in batch]
                metadatas = [
                    {
                        "parent_id": c.parent_id,
                        "doc_id": c.doc_id,
                        "page_number": c.page_number,
                        "content_type": c.metadata.get("content_type", "text"),
                    }
                    for c in batch
                ]
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )

            processed_count = min(i + batch_size, total_chunks)
            logger.debug(f"Indexed {processed_count}/{total_chunks} chunks")
            if progress_callback and total_chunks > 0:
                try:
                    pct = 70.0 + (processed_count / total_chunks) * 28.0
                    progress_callback(pct)
                except Exception:
                    pass

        logger.info(f"Successfully indexed {total_chunks} vectors into {target_store}")
        return total_chunks

    def search(
        self,
        query: str,
        top_k: int = 20,
        where_filter: Optional[Dict[str, Any]] = None,
        allowed_doc_ids: Optional[List[str]] = None
    ) -> List[VectorSearchResult]:
        """
        Performs dense vector similarity search in Supabase pgvector or ChromaDB.
        Returns top_k matching child chunks. Supports multi-tenant isolation via allowed_doc_ids.
        """
        if allowed_doc_ids is not None and len(allowed_doc_ids) == 0:
            return []

        query_vector = self.embed_texts([query])[0]

        if self.use_postgres:
            pg_matches = search_child_chunks_pgvector(
                query_vector, 
                top_k=top_k, 
                allowed_doc_ids=allowed_doc_ids
            )
            return [
                VectorSearchResult(
                    child_id=r["child_id"],
                    parent_id=r["parent_id"],
                    doc_id=r["doc_id"],
                    page_number=r["page_number"],
                    content=r["content"],
                    score=float(r["score"]),
                    metadata={
                        "parent_id": r["parent_id"],
                        "doc_id": r["doc_id"],
                        "page_number": r["page_number"],
                        "content_type": "text",
                    }
                )
                for r in pg_matches
            ]

        # ChromaDB fallback
        combined_filter = dict(where_filter) if where_filter else {}
        if allowed_doc_ids is not None:
            if len(allowed_doc_ids) == 1:
                combined_filter["doc_id"] = allowed_doc_ids[0]
            else:
                combined_filter["doc_id"] = {"$in": allowed_doc_ids}

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=combined_filter if combined_filter else None,
            include=["documents", "metadatas", "distances"]
        )

        search_results: List[VectorSearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metadatas, distances):
            score = 1.0 - (dist / 2.0) if dist is not None else 0.0
            search_results.append(VectorSearchResult(
                child_id=chunk_id,
                parent_id=meta.get("parent_id", ""),
                doc_id=meta.get("doc_id", ""),
                page_number=meta.get("page_number", 1),
                content=doc,
                score=score,
                metadata=meta
            ))

        return search_results

    def delete_by_doc_id(self, doc_id: str) -> bool:
        """
        Deletes vector embeddings associated with a doc_id.
        In Postgres, this is handled automatically via CASCADE on delete_document().
        """
        if self.use_postgres:
            logger.info(f"Supabase pgvector: Document {doc_id} vector deletion handled via FOREIGN KEY CASCADE")
            return True

        logger.info(f"Purging vector embeddings for doc_id: {doc_id} from ChromaDB")
        try:
            self.collection.delete(where={"doc_id": doc_id})
            return True
        except Exception as e:
            logger.error(f"Error purging ChromaDB vectors for {doc_id}: {str(e)}")
            return False

    def count(self) -> int:
        """Returns total count of indexed vectors across Supabase pgvector or ChromaDB."""
        if self.use_postgres:
            try:
                from src.db import get_db_connection
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) FROM child_chunks;")
                        row = cur.fetchone()
                        return row[0] if row else 0
            except Exception as e:
                logger.error(f"Error querying Postgres vector count: {e}")
                return 0
        elif self.collection:
            return self.collection.count()
        return 0

