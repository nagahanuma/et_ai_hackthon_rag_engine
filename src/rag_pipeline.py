"""
End-to-End Regulatory Compliance RAG Pipeline.
Integrates Hybrid Retrieval, Parent Context Resolution, and LLM Generation.
Captures comprehensive observability traces (Dense, BM25, RRF, Parent Chunks, LLM Prompt).
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass, field, asdict

from src.hybrid_retriever import HybridRetriever, RetrievedContext
from src.models import LLMFactory, BaseLLMClient
from src.db import get_document_by_id, save_chat_message, get_recent_chat_history
from src.semantic_cache import SemanticCache, CacheResult
from src.config import settings
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
    session_id: Optional[str] = None
    trace: Dict[str, Any] = field(default_factory=dict)


class ComplianceRAGPipeline:
    """The master orchestrator for banking regulatory query answering."""

    SYSTEM_PROMPT = (
        "You are an advanced, domain-agnostic Document Intelligence AI Assistant. "
        "Your purpose is to thoroughly analyze any uploaded documents—including technical specifications, "
        "problem statements, rules, legal contracts, regulatory guidelines, research papers, and operational reports—"
        "and answer questions accurately and strictly based on the provided context excerpts.\n\n"
        "CORE OPERATING PRINCIPLES:\n"
        "1. STRICT FACTUAL GROUNDING: Base your answer solely on the provided document excerpts. "
        "Do NOT extrapolate, hallucinate, or assume facts not present in the text.\n"
        "2. DOCUMENT AWARENESS & INTENT ORIENTATION: Adapt your tone and terminology directly to the domain "
        "of the uploaded document. When answering broad, exploratory, or overview questions (e.g. 'what are key points?', "
        "'summarize this document', 'what is this about?'), always start with a concise 1-2 sentence orientation "
        "stating what the document is, its core topic/objective/problem statement, and its scope before presenting detailed sections or rules.\n"
        "3. ACCURATE SYNTHESIS: Answer the user's specific inquiry using the relevant facts, rules, steps, definitions, "
        "metrics, or guidelines available in the excerpts.\n"
        "4. CLEAN STRUCTURE: Present information clearly using natural paragraphs, structured bullet points, "
        "or clean Markdown tables when comparing criteria or listing structured rules. Do NOT clutter the answer with "
        "repetitive document filenames—the audit engine independently tracks and displays all source citations in the verification panel.\n"
        "5. NUMERICAL & DETAIL ACCURACY: If tabular data, scoring rules, numerical thresholds, dates, or formulas are present, "
        "preserve their exact details and precision.\n"
        "6. OUT-OF-BOUNDS REFUSAL: Only state 'This topic is not covered in the uploaded documents.' if the provided "
        "excerpts contain zero relevant information regarding the inquiry."
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

    def query(
        self, 
        user_question: str, 
        session_id: Optional[str] = None,
        custom_groq_key: Optional[str] = None,
        custom_openai_key: Optional[str] = None,
        custom_model: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False
    ) -> ComplianceAnswer:
        """
        Executes complete RAG workflow, multi-turn conversational memory, and collects end-to-end trace.
        Checks Semantic Vector Cache first. If hit, bypasses retrieval and LLM.
        Supports BYOK (Bring Your Own Key) for Groq or OpenAI.
        Enforces user document isolation (only Admin searches global documents).
        """
        start_time = time.perf_counter()
        active_session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        logger.info(f"Processing compliance query [Session: {active_session_id}, User: {user_id}, Admin: {is_admin}]: '{user_question}'")

        # Resolve document ownership filter (Admin sees all; standard user only sees their own documents)
        allowed_doc_ids = None
        if user_id and not is_admin:
            from src.db import get_user_doc_ids
            allowed_doc_ids = get_user_doc_ids(user_id)

        # Fetch recent conversational context for multi-turn dialogue
        recent_history = get_recent_chat_history(active_session_id, limit=6)

        # 0. Check In-Engine Semantic Vector Cache (Multi-tenant isolated)
        if self.semantic_cache:
            cache_hit = self.semantic_cache.lookup(
                user_question, 
                allowed_doc_ids=allowed_doc_ids, 
                is_admin=is_admin
            )
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
                # Save conversation turns (user query and final answer only)
                save_chat_message(active_session_id, "user", user_question, user_id=user_id)
                save_chat_message(active_session_id, "assistant", cache_hit.answer, user_id=user_id)

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
                    session_id=active_session_id,
                    trace=trace
                )
            else:
                logger.info(f"[SEMANTIC CACHE MISS] Query not in cache -> Routing to Dense Vector + BM25 Hybrid Retrieval...")

        # 1. Capture Dense Vector Matches (ChromaDB or pgvector)
        t_dense_start = time.perf_counter()
        dense_results = self.retriever.vector_manager.search(
            query=user_question, 
            top_k=10, 
            allowed_doc_ids=allowed_doc_ids
        )
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
        if allowed_doc_ids is not None:
            bm25_results = [r for r in bm25_results if r.get("doc_id") in allowed_doc_ids]
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
        top_k_parents = getattr(settings, "TOP_PARENT_CONTEXTS", 3)
        contexts = self.retriever.retrieve(
            query=user_question, 
            final_top_parents=top_k_parents,
            allowed_doc_ids=allowed_doc_ids
        )

        if not contexts:
            logger.warning("No relevant regulatory contexts found for query.")
            if allowed_doc_ids is not None and len(allowed_doc_ids) == 0:
                fallback_answer = "You have not uploaded any regulatory documents yet. Please upload a compliance PDF/document in the Documents tab first."
            else:
                fallback_answer = "No relevant regulatory documents were found matching your query."
            save_chat_message(active_session_id, "user", user_question, user_id=user_id)
            save_chat_message(active_session_id, "assistant", fallback_answer, user_id=user_id)
            return ComplianceAnswer(
                query=user_question,
                answer=fallback_answer,
                citations=[],
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                total_sources_retrieved=0,
                session_id=active_session_id,
                trace={"cache_hit": False, "dense_matches": dense_trace, "bm25_matches": bm25_trace, "resolved_parents": []}
            )

        # 4. Assembled Parent Contexts & Conversation Turns
        context_str, citations = self._build_context_prompt(contexts)
        
        conversation_history_block = ""
        if recent_history:
            history_lines = [f"{m['role'].capitalize()}: {m['content']}" for m in recent_history]
            conversation_history_block = "PRIOR CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n"

        user_prompt = (
            f"DOCUMENT CONTEXT EXCERPTS:\n{context_str}\n\n"
            f"{conversation_history_block}"
            f"USER QUESTION: {user_question}\n\n"
            f"Provide an accurate, well-structured answer based strictly on the excerpts above:"
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

        # 5. LLM Generation (Resolve BYOK Client or Default System Client)
        if (custom_groq_key and custom_groq_key.strip()) or (custom_openai_key and custom_openai_key.strip()) or (custom_model and custom_model.strip()):
            active_llm = LLMFactory.get_client(
                custom_groq_key=custom_groq_key, 
                custom_openai_key=custom_openai_key,
                custom_model=custom_model
            )
        else:
            active_llm = self.llm

        model_display = getattr(active_llm, "model", "LLM")
        logger.info(f"[STAGE 5/5] LLM GENERATION ({model_display}) -> Sending zero-hallucination prompt with {len(context_str)} chars of ground-truth context...")
        t_llm_start = time.perf_counter()
        raw_answer = active_llm.generate(
            prompt=user_prompt,
            system_prompt=self.SYSTEM_PROMPT
        )
        llm_time_ms = (time.perf_counter() - t_llm_start) * 1000

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000
        citation_summary = [f"{c.filename} (p.{c.page_number})" for c in citations]
        logger.info(f"[QUERY COMPLETED] in {total_elapsed_ms:.1f}ms (Retrieval: {dense_time_ms+bm25_time_ms:.1f}ms, LLM: {llm_time_ms:.1f}ms) | Citations: {citation_summary}")

        # 6. Store in In-Engine Semantic Cache for future identical / rephrased queries
        if self.semantic_cache and raw_answer and "This topic is not covered" not in raw_answer:
            self.semantic_cache.store(query=user_question, answer=raw_answer, citations=citations, user_id=user_id)

        # 7. Persist to Conversational Chat History (user query and final answer only)
        save_chat_message(active_session_id, "user", user_question, user_id=user_id)
        save_chat_message(active_session_id, "assistant", raw_answer, user_id=user_id)

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
            session_id=active_session_id,
            trace=trace
        )
