"""
Vector Store module managing ChromaDB and embedding generation.
Supports local open-source BAAI/bge-m3 embeddings and collection versioning.
Provides batch ingestion, vector search, and metadata-filtered cascade deletion.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.chunker import ChildChunk
from src.logger import get_logger

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
    """Manages persistent ChromaDB vector storage and embedding inference."""

    def __init__(self):
        # 1. Initialize persistent ChromaDB client
        db_path = settings.CHROMA_DB_PATH
        db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # 2. Connect to the versioned collection
        self.collection_name = settings.VECTOR_COLLECTION_NAME
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Regulatory Compliance Child Chunks"}
        )
        logger.info(f"Connected to ChromaDB collection: '{self.collection_name}' at {db_path}")

        # 3. Load Embedding Model (Local BAAI/bge-m3 or configured model)
        self.embedding_model = None
        self._init_embedding_model()

    def _init_embedding_model(self):
        """Loads the embedding model based on settings."""
        if settings.EMBEDDING_PROVIDER == "local":
            logger.info(f"Loading local embedding model: '{settings.EMBEDDING_MODEL}' (sentence-transformers)...")
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("Local embedding model loaded successfully.")
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
                normalize_embeddings=True  # Enables pure cosine similarity via dot-product
            )
            return embeddings.tolist()
        else:
            raise NotImplementedError("External embedding provider requires OpenAI adapter.")

    def add_child_chunks(self, chunks: List[ChildChunk]) -> int:
        """
        Batches and inserts child chunks into ChromaDB with metadata payloads.
        Returns total chunks added.
        """
        if not chunks:
            return 0

        total_chunks = len(chunks)
        batch_size = settings.EMBEDDING_BATCH_SIZE
        logger.info(f"Adding {total_chunks} child chunks to ChromaDB (batch size: {batch_size})...")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            
            ids = [c.child_id for c in batch]
            documents = [c.content for c in batch]
            metadatas = [
                {
                    "parent_id": c.parent_id,
                    "doc_id": c.doc_id,
                    "page_number": c.page_number,
                    "content_type": c.metadata.get("content_type", "text"),
                }
                for c in batch
            ]

            embeddings = self.embed_texts(documents)

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.debug(f"Upserted {min(i + batch_size, total_chunks)}/{total_chunks} vectors into ChromaDB")

        logger.info(f"Successfully indexed {total_chunks} child vectors into '{self.collection_name}'")
        return total_chunks

    def search(
        self,
        query: str,
        top_k: int = 20,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Performs dense vector similarity search in ChromaDB.
        Returns top_k matching child chunks.
        """
        query_vector = self.embed_texts([query])[0]

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
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
            # Chroma returns L2 or cosine distance; convert to similarity score
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

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Deletes all vector embeddings associated with a doc_id.
        Used for cascade delete and document replacement.
        """
        logger.info(f"Purging all vector embeddings for doc_id: {doc_id} from collection '{self.collection_name}'")
        try:
            self.collection.delete(where={"doc_id": doc_id})
            logger.info(f"Vector embeddings for doc_id: {doc_id} successfully purged.")
            return True
        except Exception as e:
            logger.error(f"Error purging vectors for doc_id {doc_id}: {str(e)}")
            return False
