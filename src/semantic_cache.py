"""
In-Engine Semantic Caching for Document Intelligence & Compliance RAG.
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
        collection_name: Optional[str] = None
    ):
        self.threshold = similarity_threshold
        # Derive collection name from embedding model to prevent dimension collisions
        model_slug = settings.EMBEDDING_MODEL.replace("/", "_").replace("-", "_")
        self.collection_name = collection_name or f"cache_{model_slug}"

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
            logger.info(f"Initializing embedding model for Semantic Cache on device '{settings.EMBEDDING_DEVICE}'...")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE)

        logger.info(f"Semantic Cache ready (Collection: {self.collection_name}, Threshold: {self.threshold}, Total Cached: {self.collection.count()})")

    def lookup(
        self, 
        query: str, 
        allowed_doc_ids: Optional[List[str]] = None, 
        is_admin: bool = False
    ) -> Optional[CacheResult]:
        """
        Checks cache for a semantically equivalent query using cosine similarity.
        Enforces strict multi-tenant isolation: standard users cannot hit cached answers 
        derived from documents they do not own.
        """
        # If standard user has 0 documents, cache lookup must not return documents
        if not is_admin and allowed_doc_ids is not None and len(allowed_doc_ids) == 0:
            logger.debug("[CACHE BYPASSED] User has 0 indexed documents.")
            return None

        if self.collection.count() == 0:
            return None

        t0 = time.perf_counter()
        query_vec = self.model.encode([query], normalize_embeddings=True).tolist()[0]

        try:
            results = self.collection.query(
                query_embeddings=[query_vec],
                n_results=1,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning(f"Cache dimension mismatch ({e}). Resetting cache collection...")
                try:
                    self.chroma_client.delete_collection(self.collection_name)
                except Exception:
                    pass
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                return None
            raise e

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if results and results["distances"] and len(results["distances"][0]) > 0:
            # In ChromaDB cosine space: distance = 1 - cosine_similarity
            cosine_distance = results["distances"][0][0]
            similarity = round(1.0 - cosine_distance, 4)

            if similarity >= self.threshold:
                metadata = results["metadatas"][0][0]
                matched_orig_query = results["documents"][0][0]

                citations = []
                if "citations_json" in metadata and metadata["citations_json"]:
                    try:
                        citations = json.loads(metadata["citations_json"])
                    except Exception:
                        citations = []

                # Multi-tenant isolation verification
                if not is_admin and allowed_doc_ids is not None:
                    cached_doc_ids = set()
                    for c in citations:
                        d_id = c.get("doc_id")
                        if d_id and d_id != "cached":
                            cached_doc_ids.add(d_id)
                    
                    raw_meta_doc_ids = metadata.get("doc_ids", "")
                    if raw_meta_doc_ids:
                        for d_id in raw_meta_doc_ids.split(","):
                            if d_id.strip():
                                cached_doc_ids.add(d_id.strip())

                    # If cached entry references documents not owned by this user, REJECT IT!
                    if cached_doc_ids and not cached_doc_ids.issubset(set(allowed_doc_ids)):
                        logger.info(
                            f"⚡ [CACHE TENANT ISOLATION REJECTION] Cache match '{matched_orig_query}' (Similarity: {similarity}) "
                            f"rejected: cached docs {cached_doc_ids} not in allowed user docs {allowed_doc_ids}."
                        )
                        return None

                logger.info(
                    f"⚡ [CACHE HIT] Similarity: {similarity:.4f} >= {self.threshold} | "
                    f"Matched: '{matched_orig_query}' | Retrieved in {elapsed_ms:.1f}ms"
                )

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

    def store(self, query: str, answer: str, citations: List[Any], user_id: Optional[str] = None):
        """Saves a verified Q&A pair and citations into the semantic cache with tenant metadata."""
        try:
            query_vec = self.model.encode([query], normalize_embeddings=True).tolist()[0]
            cache_id = f"cache_{uuid.uuid4().hex[:8]}"

            # Serialize citations cleanly
            serializable_citations = []
            doc_ids = set()
            for c in citations:
                if hasattr(c, "__dict__"):
                    d = c.__dict__
                    serializable_citations.append(d)
                    if d.get("doc_id"):
                        doc_ids.add(str(d["doc_id"]))
                elif isinstance(c, dict):
                    serializable_citations.append(c)
                    if c.get("doc_id"):
                        doc_ids.add(str(c["doc_id"]))

            self.collection.add(
                ids=[cache_id],
                embeddings=[query_vec],
                documents=[query],
                metadatas=[{
                    "answer": answer,
                    "citations_json": json.dumps(serializable_citations),
                    "doc_ids": ",".join(sorted(doc_ids)),
                    "user_id": user_id or "system",
                    "stored_at": time.time()
                }]
            )
            logger.info(f"Cached query '{query}' under id '{cache_id}' (Doc IDs: {list(doc_ids)}).")
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

    # Simulated Banking Q&A pair
    sample_query = "What is the minimum Common Equity Tier 1 (CET1) capital ratio requirement under Basel III circulars?"
    sample_answer = "Under Basel III statutory norms, the minimum CET1 capital ratio is 8.00% of risk-weighted assets, supported by a 2.50% Capital Conservation Buffer (CCB)."
    sample_citations = [{"filename": "RBI_Prudential_Norms_Basel_III_2024.pdf", "page_number": 1}]

    print(f"\n1. Storing prime query: '{sample_query}'")
    cache.store(sample_query, sample_answer, sample_citations)

    # Test Case A: Semantically identical query with completely different wording
    test_query_hit = "What are the minimum capital adequacy ratios prescribed by RBI under Basel III?"
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
    test_query_miss = "What is the prescribed limit for commercial real estate exposure?"
    print(f"\n3. Querying unrelated question: '{test_query_miss}'")
    miss_result = cache.lookup(test_query_miss)

    if miss_result:
        print(f"   >>> CACHE HIT: {miss_result.answer}")
    else:
        print("   >>> ❌ CACHE MISS! (Correct behavior: routes to Hybrid RAG + LLM)")

    print("\n" + "=" * 70)

