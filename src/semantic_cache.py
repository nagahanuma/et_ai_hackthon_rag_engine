"""
In-Engine Semantic Caching for Banking Compliance RAG.
Features:
- 100% Free: Uses existing local ChromaDB instance & BGE-small embedding model.
- Sub-10ms Latency: Returns cached answers for semantically identical questions.
- Self-Contained: Zero external services (no Redis, no GPTCache, no paid APIs).
"""

import time
import json
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.logger import get_logger

logger = get_logger("semantic_cache")


@dataclass
class CacheResult:
    is_hit: bool
    answer: str
    citations: List[Dict[str, Any]]
    similarity_score: float
    latency_ms: float
    matched_query: str


class SemanticCache:
    """
    Semantic Vector Cache storing previous queries and verified compliance answers.
    Uses Cosine Similarity threshold (default 0.90) to match differently phrased queries.
    """

    def __init__(
        self,
        embedding_model: Optional[SentenceTransformer] = None,
        similarity_threshold: float = 0.90,
        collection_name: str = "compliance_semantic_cache_v1"
    ):
        self.threshold = similarity_threshold
        self.collection_name = collection_name

        # 1. Reuse existing persistent ChromaDB storage
        self.chroma_client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DB_PATH),
            settings=Settings(anonymized_telemetry=False)
        )

        # 2. Dedicated cache collection with Cosine distance metric
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # 3. Reuse loaded embedding model or load local BGE model
        if embedding_model:
            self.model = embedding_model
        else:
            logger.info("Initializing embedding model for Semantic Cache...")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

        logger.info(f"Semantic Cache ready (Threshold: {self.threshold}, Total Cached: {self.collection.count()})")

    def lookup(self, query: str) -> Optional[CacheResult]:
        """
        Checks if a semantically equivalent query was answered before.
        Returns CacheResult if similarity >= threshold, else None.
        """
        if self.collection.count() == 0:
            return None

        t0 = time.perf_counter()
        query_vec = self.model.encode([query], normalize_embeddings=True).tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if results and results["distances"] and len(results["distances"][0]) > 0:
            # In ChromaDB cosine space: distance = 1 - cosine_similarity
            cosine_distance = results["distances"][0][0]
            similarity = round(1.0 - cosine_distance, 4)

            if similarity >= self.threshold:
                metadata = results["metadatas"][0][0]
                matched_orig_query = results["documents"][0][0]

                logger.info(
                    f"⚡ [CACHE HIT] Similarity: {similarity:.4f} >= {self.threshold} | "
                    f"Matched: '{matched_orig_query}' | Retrieved in {elapsed_ms:.1f}ms"
                )

                citations = []
                if "citations_json" in metadata and metadata["citations_json"]:
                    try:
                        citations = json.loads(metadata["citations_json"])
                    except Exception:
                        citations = []

                return CacheResult(
                    is_hit=True,
                    answer=metadata["answer"],
                    citations=citations,
                    similarity_score=similarity,
                    latency_ms=elapsed_ms,
                    matched_query=matched_orig_query
                )

        logger.debug(f"[CACHE MISS] No semantically similar query found in cache ({elapsed_ms:.1f}ms).")
        return None

    def store(self, query: str, answer: str, citations: List[Any]):
        """Saves a verified Q&A pair and citations into the semantic cache."""
        try:
            query_vec = self.model.encode([query], normalize_embeddings=True).tolist()[0]
            cache_id = f"cache_{uuid.uuid4().hex[:8]}"

            # Serialize citations cleanly
            serializable_citations = []
            for c in citations:
                if hasattr(c, "__dict__"):
                    serializable_citations.append(c.__dict__)
                elif isinstance(c, dict):
                    serializable_citations.append(c)

            self.collection.add(
                ids=[cache_id],
                embeddings=[query_vec],
                documents=[query],
                metadatas=[{
                    "answer": answer,
                    "citations_json": json.dumps(serializable_citations),
                    "stored_at": time.time()
                }]
            )
            logger.info(f"Cached query '{query}' under id '{cache_id}'.")
        except Exception as e:
            logger.warning(f"Failed to store entry in semantic cache: {e}")

    def clear(self):
        """Purges cache entries (useful when documents are updated or deleted)."""
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Semantic Cache cleared.")


# ==========================================
# INTERACTIVE DEMONSTRATION & TEST RUNNER
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧠 DEMONSTRATING SEMANTIC CACHE (ZERO EXTRA SOFTWARE)")
    print("=" * 70)

    cache = SemanticCache(similarity_threshold=0.88)

    # Simulated Q&A pair
    sample_query = "What is the formula to convert CGPA to percentage?"
    sample_answer = "Percentage of Marks = (10 × CGPA) – 7.5 as specified in Section 5.4(e)."
    sample_citations = [{"filename": "regulations.pdf", "page_number": 38}]

    print(f"\n1. Storing prime query: '{sample_query}'")
    cache.store(sample_query, sample_answer, sample_citations)

    # Test Case A: Semantically identical query with completely different wording
    test_query_hit = "how to calculate percentage from cgpa?"
    print(f"\n2. Querying different wording: '{test_query_hit}'")
    hit_result = cache.lookup(test_query_hit)

    if hit_result:
        print(f"   >>> 🎉 CACHE HIT in {hit_result.latency_ms:.1f}ms!")
        print(f"   >>> Similarity: {hit_result.similarity_score * 100:.1f}%")
        print(f"   >>> Matched Original: '{hit_result.matched_query}'")
        print(f"   >>> Answer: '{hit_result.answer}'")
    else:
        print("   >>> CACHE MISS")

    # Test Case B: Completely different query
    test_query_miss = "What are the capital adequacy requirements for credit risk?"
    print(f"\n3. Querying unrelated question: '{test_query_miss}'")
    miss_result = cache.lookup(test_query_miss)

    if miss_result:
        print(f"   >>> CACHE HIT: {miss_result.answer}")
    else:
        print(f"   >>> ❌ CACHE MISS! (Correct behavior: routes to Hybrid RAG + LLM)")

    print("\n" + "=" * 70)
