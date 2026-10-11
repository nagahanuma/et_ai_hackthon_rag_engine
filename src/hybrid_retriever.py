"""
Hybrid Retrieval Engine combining Dense Vector Search and BM25 Keyword Search.
Uses Reciprocal Rank Fusion (RRF) to merge rankings and resolves full Parent Chunks from SQLite.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.vector_store import VectorStoreManager, VectorSearchResult
from src.db import get_parent_chunk_by_id
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedContext:
    """Represents a final retrieved parent context with citation metadata ready for the LLM."""
    parent_id: str
    doc_id: str
    page_number: int
    content: str  # Full Parent paragraph or Markdown table
    content_type: str
    rrf_score: float
    matched_child_snippets: List[str]


class HybridRetriever:
    """Orchestrates BM25 + Dense Vector Search and resolves Parent Chunks."""

    def __init__(self, vector_manager: Optional[VectorStoreManager] = None):
        self.vector_manager = vector_manager or VectorStoreManager()
        self.bm25_index: Optional[BM25Okapi] = None
        self.corpus_chunks: List[Dict[str, Any]] = []

    def build_bm25_index(self, child_chunks: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Builds in-memory BM25 index from active child chunks in ChromaDB.
        Tokenizes text for fast exact-keyword matching.
        """
        if child_chunks is None:
            # Fetch all documents currently stored in the ChromaDB collection
            collection_data = self.vector_manager.collection.get(
                include=["documents", "metadatas"]
            )
            if not collection_data or not collection_data["ids"]:
                logger.warning("No documents found in ChromaDB to build BM25 index.")
                return

            self.corpus_chunks = []
            for cid, doc, meta in zip(
                collection_data["ids"],
                collection_data["documents"],
                collection_data["metadatas"]
            ):
                self.corpus_chunks.append({
                    "child_id": cid,
                    "content": doc,
                    "parent_id": meta.get("parent_id", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "page_number": meta.get("page_number", 1),
                    "content_type": meta.get("content_type", "text"),
                })
        else:
            self.corpus_chunks = child_chunks

        # Tokenize corpus for BM25
        tokenized_corpus = [self._tokenize(chunk["content"]) for chunk in self.corpus_chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        logger.info(f"Built BM25 index over {len(self.corpus_chunks)} child chunks.")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple lowercase word tokenizer for BM25."""
        return re.findall(r'\w+', text.lower())

    def search_bm25(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Executes sparse BM25 keyword search."""
        if not self.bm25_index or not self.corpus_chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices sorted by BM25 score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return chunks with actual keyword matches
                item = self.corpus_chunks[idx].copy()
                item["bm25_score"] = float(scores[idx])
                results.append(item)

        return results

    def retrieve(
        self,
        query: str,
        top_k_dense: int = 20,
        top_k_bm25: int = 20,
        final_top_parents: int = 5,
        rrf_k: int = 60
    ) -> List[RetrievedContext]:
        """
        Executes Hybrid Search:
        1. Dense Vector Search (ChromaDB)
        2. Sparse Keyword Search (BM25)
        3. Reciprocal Rank Fusion (RRF)
        4. Resolves unique full Parent Chunks from SQLite
        """
        logger.info(f"[START QUERY PIPELINE] '{query}'")

        # 1. Dense Vector Search
        dense_results = self.vector_manager.search(query=query, top_k=top_k_dense)
        top_dense_info = (
            f"Top match: {dense_results[0].child_id} (Score: {dense_results[0].score:.4f}, Page {dense_results[0].page_number})"
            if dense_results else "No matches"
        )
        logger.info(f"[STAGE 1/4] DENSE VECTOR SEARCH (ChromaDB) -> Retrieved {len(dense_results)} candidates. {top_dense_info}")
        
        # 2. Sparse BM25 Search
        total_vectors = self.vector_manager.collection.count()
        if not self.bm25_index or len(self.corpus_chunks) != total_vectors:
            logger.info(f"[BM25 SYNC] Synchronizing BM25 sparse index ({len(self.corpus_chunks)} -> {total_vectors} vectors)...")
            self.build_bm25_index()
        bm25_results = self.search_bm25(query=query, top_k=top_k_bm25)
        top_bm25_info = (
            f"Top match: {bm25_results[0]['child_id']} (Score: {bm25_results[0]['bm25_score']:.4f}, Page {bm25_results[0]['page_number']})"
            if bm25_results else "No matches"
        )
        logger.info(f"[STAGE 2/4] SPARSE KEYWORD SEARCH (BM25 Okapi) -> Retrieved {len(bm25_results)} candidates. {top_bm25_info}")

        # 3. Reciprocal Rank Fusion (RRF)
        # Score formula: 1 / (rrf_k + rank)
        rrf_scores: Dict[str, float] = {}
        child_lookup: Dict[str, Dict[str, Any]] = {}

        # Rank Dense results
        for rank, res in enumerate(dense_results, start=1):
            cid = res.child_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
            child_lookup[cid] = {
                "child_id": cid,
                "parent_id": res.parent_id,
                "doc_id": res.doc_id,
                "page_number": res.page_number,
                "content": res.content,
            }

        # Rank BM25 results
        for rank, res in enumerate(bm25_results, start=1):
            cid = res["child_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
            if cid not in child_lookup:
                child_lookup[cid] = res

        # Sort all matched children by combined RRF score descending
        sorted_children = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_rrf_score = sorted_children[0][1] if sorted_children else 0.0
        logger.info(f"[STAGE 3/4] RECIPROCAL RANK FUSION (RRF) -> Merged {len(sorted_children)} unique child candidates (Top RRF: {top_rrf_score:.5f})")

        # 4. Resolve Parent Chunks from SQLite & Deduplicate
        resolved_parents: Dict[str, RetrievedContext] = {}

        for cid, score in sorted_children:
            meta = child_lookup[cid]
            parent_id = meta["parent_id"]

            if not parent_id:
                continue

            if parent_id not in resolved_parents:
                parent_data = get_parent_chunk_by_id(parent_id)
                if parent_data:
                    resolved_parents[parent_id] = RetrievedContext(
                        parent_id=parent_id,
                        doc_id=parent_data["doc_id"],
                        page_number=parent_data["page_number"],
                        content=parent_data["content"],
                        content_type=parent_data.get("content_type", "text"),
                        rrf_score=score,
                        matched_child_snippets=[meta["content"]]
                    )
            else:
                # Add child snippet to already resolved parent for complete traceability
                resolved_parents[parent_id].matched_child_snippets.append(meta["content"])

            if len(resolved_parents) >= final_top_parents:
                break

        final_contexts = list(resolved_parents.values())
        pages_resolved = sorted(list(set(c.page_number for c in final_contexts)))
        total_chars = sum(len(c.content) for c in final_contexts)
        logger.info(f"[STAGE 4/4] RESOLVED PARENTS (SQLite) -> {len(final_contexts)} parent legal sections retrieved from Pages {pages_resolved} (Total: {total_chars} chars)")
        return final_contexts
