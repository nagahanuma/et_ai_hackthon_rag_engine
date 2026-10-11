"""
End-to-End Regulatory Compliance RAG Pipeline.
Integrates Hybrid Retrieval, Parent Context Resolution, and LLM Generation.
Captures comprehensive observability traces (Dense, BM25, RRF, Parent Chunks, LLM Prompt).
"""

import time
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass, field

from src.hybrid_retriever import HybridRetriever, RetrievedContext
from src.models import LLMFactory, BaseLLMClient
from src.db import get_document_by_id
from src.semantic_cache import SemanticCache, CacheResult
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Citation:
    """Audit citation linking a factual claim to its exact source page."""
    doc_id: str
    filename: str
    page_number: int
    parent_id: str
    content_type: str


@dataclass
class ComplianceAnswer:
    """Complete response returned to the client/UI with full observability trace."""
    query: str
    answer: str
    citations: List[Citation]
    execution_time_ms: float
    total_sources_retrieved: int
    trace: Dict[str, Any] = field(default_factory=dict)


class ComplianceRAGPipeline:
    """The master orchestrator for banking regulatory query answering."""

    SYSTEM_PROMPT = (
        "You are an expert Banking and Financial Regulatory Compliance Officer. "
        "Your task is to answer questions strictly using the provided regulatory contexts below. "
        "RULES:\n"
        "1. Only use facts directly stated in the contexts. Do NOT extrapolate or assume.\n"
        "2. If the contexts do not contain enough information to answer, state clearly: "
        "'This topic is not covered in the current active regulatory documents.'\n"
        "3. When stating rules, percentages, limits, or requirements, cite the specific document and page number.\n"
        "4. If markdown tables are provided, preserve their numerical accuracy."
    )

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        llm_client: Optional[BaseLLMClient] = None,
        enable_cache: bool = True
    ):
        self.retriever = retriever or HybridRetriever()
        self.llm = llm_client or LLMFactory.get_client()
        self.enable_cache = enable_cache
        if self.enable_cache:
            # Share loaded SentenceTransformer model from retriever to avoid extra RAM
            self.semantic_cache = SemanticCache(
                embedding_model=self.retriever.vector_manager.embedding_model,
                similarity_threshold=0.88
            )
        else:
            self.semantic_cache = None
        logger.info("Compliance RAG Pipeline successfully initialized with Semantic Cache.")

    def _build_context_prompt(self, contexts: List[RetrievedContext]) -> Tuple[str, List[Citation]]:
        """Formats retrieved parent contexts into an annotated prompt block with citations."""
        formatted_blocks = []
        citations: List[Citation] = []
        doc_cache: Dict[str, str] = {}

        for idx, ctx in enumerate(contexts, start=1):
            if ctx.doc_id not in doc_cache:
                doc_record = get_document_by_id(ctx.doc_id)
                doc_cache[ctx.doc_id] = doc_record["filename"] if doc_record else "Unknown Document"
            
            filename = doc_cache[ctx.doc_id]

            citation = Citation(
                doc_id=ctx.doc_id,
                filename=filename,
                page_number=ctx.page_number,
                parent_id=ctx.parent_id,
                content_type=ctx.content_type
            )
            citations.append(citation)

            block = (
                f"--- SOURCE EXCERPT [{idx}] ---\n"
                f"Document: {filename}\n"
                f"Page: {ctx.page_number} | Type: {ctx.content_type}\n"
                f"Content:\n{ctx.content}\n"
            )
            formatted_blocks.append(block)

        context_string = "\n".join(formatted_blocks)
        return context_string, citations

    def query(self, user_question: str) -> ComplianceAnswer:
        """
        Executes complete RAG workflow and collects end-to-end observability trace.
        Checks Semantic Vector Cache first. If hit, bypasses retrieval and LLM.
        """
        start_time = time.perf_counter()
        logger.info(f"Processing compliance query: '{user_question}'")

        # 0. Check In-Engine Semantic Vector Cache
        if self.semantic_cache:
            cache_hit = self.semantic_cache.lookup(user_question)
            if cache_hit:
                logger.info(
                    f"⚡ [SEMANTIC CACHE HIT] Bypassing LLM in {cache_hit.latency_ms:.1f}ms! "
                    f"Similarity: {cache_hit.similarity_score:.4f} against '{cache_hit.matched_query}'"
                )
                cached_citations = [
                    Citation(
                        doc_id=c.get("doc_id", "cached"),
                        filename=c.get("filename", "Verified Regulatory Context"),
                        page_number=c.get("page_number", 1),
                        parent_id=c.get("parent_id", "cached"),
                        content_type=c.get("content_type", "text")
                    )
                    for c in cache_hit.citations
                ]
                trace = {
                    "cache_hit": True,
                    "matched_query": cache_hit.matched_query,
                    "similarity_score": cache_hit.similarity_score,
                    "timing_breakdown": {
                        "cache_lookup_ms": round(cache_hit.latency_ms, 2),
                        "dense_search_ms": 0.0,
                        "bm25_search_ms": 0.0,
                        "llm_generation_ms": 0.0,
                        "total_ms": round(cache_hit.latency_ms, 2),
                    },
                    "dense_vector_matches": [],
                    "bm25_keyword_matches": [],
                    "resolved_parent_contexts": [],
                    "full_prompt_sent_to_llm": {
                        "system_prompt": "Bypassed via In-Engine Semantic Cache (0 LLM Tokens)",
                        "user_prompt": f"Matched previous verified query: '{cache_hit.matched_query}' (Similarity: {cache_hit.similarity_score * 100:.1f}%)"
                    }
                }
                return ComplianceAnswer(
                    query=user_question,
                    answer=cache_hit.answer,
                    citations=cached_citations,
                    execution_time_ms=cache_hit.latency_ms,
                    total_sources_retrieved=len(cached_citations),
                    trace=trace
                )
            else:
                logger.info(f"[SEMANTIC CACHE MISS] Query not in cache -> Routing to Dense Vector + BM25 Hybrid Retrieval...")

        # 1. Capture Dense Vector Matches (ChromaDB)
        t_dense_start = time.perf_counter()
        dense_results = self.retriever.vector_manager.search(query=user_question, top_k=10)
        dense_time_ms = (time.perf_counter() - t_dense_start) * 1000

        dense_trace = [
            {
                "child_id": r.child_id,
                "parent_id": r.parent_id,
                "score": round(r.score, 4),
                "page_number": r.page_number,
                "snippet": r.content[:160] + "..." if len(r.content) > 160 else r.content,
            }
            for r in dense_results
        ]

        # 2. Capture BM25 Keyword Matches
        t_bm25_start = time.perf_counter()
        if not self.retriever.bm25_index:
            self.retriever.build_bm25_index()
        bm25_results = self.retriever.search_bm25(query=user_question, top_k=10)
        bm25_time_ms = (time.perf_counter() - t_bm25_start) * 1000

        bm25_trace = [
            {
                "child_id": r["child_id"],
                "parent_id": r["parent_id"],
                "bm25_score": round(r.get("bm25_score", 0.0), 4),
                "page_number": r["page_number"],
                "snippet": r["content"][:160] + "..." if len(r["content"]) > 160 else r["content"],
            }
            for r in bm25_results
        ]

        # 3. Hybrid Search & Parent Context Resolution
        contexts = self.retriever.retrieve(query=user_question, final_top_parents=5)

        if not contexts:
            logger.warning("No relevant regulatory contexts found for query.")
            return ComplianceAnswer(
                query=user_question,
                answer="No relevant regulatory documents were found matching your query.",
                citations=[],
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                total_sources_retrieved=0,
                trace={"cache_hit": False, "dense_matches": dense_trace, "bm25_matches": bm25_trace, "resolved_parents": []}
            )

        # 4. Assembled Parent Contexts
        context_str, citations = self._build_context_prompt(contexts)
        
        user_prompt = (
            f"REGULATORY CONTEXT EXCERPTS:\n{context_str}\n\n"
            f"COMPLIANCE QUESTION: {user_question}\n\n"
            f"Provide a structured, precise compliance answer with document citations:"
        )

        parent_trace = [
            {
                "parent_id": c.parent_id,
                "doc_id": c.doc_id,
                "page_number": c.page_number,
                "content_type": c.content_type,
                "rrf_score": round(c.rrf_score, 5),
                "full_parent_content": c.content,
            }
            for c in contexts
        ]

        # 5. LLM Generation
        model_display = getattr(self.llm, "model", "LLM")
        logger.info(f"[STAGE 5/5] LLM GENERATION ({model_display}) -> Sending zero-hallucination prompt with {len(context_str)} chars of ground-truth context...")
        t_llm_start = time.perf_counter()
        raw_answer = self.llm.generate(
            prompt=user_prompt,
            system_prompt=self.SYSTEM_PROMPT
        )
        llm_time_ms = (time.perf_counter() - t_llm_start) * 1000

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000
        citation_summary = [f"{c.filename} (p.{c.page_number})" for c in citations]
        logger.info(f"[QUERY COMPLETED] in {total_elapsed_ms:.1f}ms (Retrieval: {dense_time_ms+bm25_time_ms:.1f}ms, LLM: {llm_time_ms:.1f}ms) | Citations: {citation_summary}")

        # 6. Store in In-Engine Semantic Cache for future identical / rephrased queries
        if self.semantic_cache and raw_answer and "This topic is not covered" not in raw_answer:
            self.semantic_cache.store(query=user_question, answer=raw_answer, citations=citations)

        # Compile Full Pipeline Trace for Frontend Visualizer
        trace = {
            "cache_hit": False,
            "timing_breakdown": {
                "cache_lookup_ms": 0.0,
                "dense_search_ms": round(dense_time_ms, 2),
                "bm25_search_ms": round(bm25_time_ms, 2),
                "llm_generation_ms": round(llm_time_ms, 2),
                "total_ms": round(total_elapsed_ms, 2),
            },
            "dense_vector_matches": dense_trace,
            "bm25_keyword_matches": bm25_trace,
            "resolved_parent_contexts": parent_trace,
            "full_prompt_sent_to_llm": {
                "system_prompt": self.SYSTEM_PROMPT,
                "user_prompt": user_prompt
            }
        }

        return ComplianceAnswer(
            query=user_question,
            answer=raw_answer,
            citations=citations,
            execution_time_ms=total_elapsed_ms,
            total_sources_retrieved=len(citations),
            trace=trace
        )
