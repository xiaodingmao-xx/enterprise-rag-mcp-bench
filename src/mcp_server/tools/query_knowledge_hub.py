"""MCP Tool: query_knowledge_hub

This tool provides knowledge retrieval capabilities through the MCP protocol.
It combines HybridSearch (Dense + Sparse + RRF Fusion) with optional Reranking
to find relevant documents and return formatted results with citations.

Usage via MCP:
    Tool name: query_knowledge_hub
    Input schema:
        - query (string, required): The search query
        - top_k (integer, optional): Number of results to return (default: 5)
        - collection (string, optional): Limit search to specific collection
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, TYPE_CHECKING

from mcp import types

from src.core.response.answer_generator import AnswerGenerator
from src.core.response.grounded_answer_builder import GroundedAnswerBuilder
from src.core.response.hallucination_guard import HallucinationGuard
from src.core.response.retrieval_status import (
    RetrievalStatus,
    assess_retrieval_status,
    contexts_from_results,
)
from src.core.response.response_builder import ResponseBuilder, MCPToolResponse
from src.core.settings import load_settings, resolve_path, Settings
from src.core.trace import TraceContext, TraceCollector
from src.core.types import RetrievalResult
from src.core.query_engine.query_cache import LruTtlCache

if TYPE_CHECKING:
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.reranker import CoreReranker

logger = logging.getLogger(__name__)


# Tool metadata
TOOL_NAME = "query_knowledge_hub"
TOOL_DESCRIPTION = """Search the knowledge base for relevant documents.

This tool uses hybrid search (semantic + keyword) to find the most relevant 
documents matching your query. Results include source citations for reference.

Parameters:
- query: Your search question or keywords
- top_k: Maximum number of results (default: 5)
- collection: Limit search to a specific document collection
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query or question to find relevant documents for.",
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum number of results to return.",
            "default": 5,
            "minimum": 1,
            "maximum": 20,
        },
        "collection": {
            "type": "string",
            "description": "Optional collection name to limit the search scope.",
        },
        "mode": {
            "type": "string",
            "enum": ["contexts", "answer"],
            "description": "Return retrieved contexts or generate a grounded answer.",
            "default": "contexts",
        },
        "answer_style": {
            "type": "string",
            "enum": ["concise", "detailed", "bullet"],
            "description": "Answer style used when mode='answer'.",
            "default": "concise",
        },
        "language": {
            "type": "string",
            "enum": ["auto", "zh", "en"],
            "description": "Answer language used when mode='answer'.",
            "default": "auto",
        },
        "include_sources": {
            "type": "boolean",
            "description": "Whether to include source metadata in the structured response.",
            "default": True,
        },
        "include_citations": {
            "type": "boolean",
            "description": "Whether to include citation metadata in the structured response.",
            "default": True,
        },
    },
    "required": ["query"],
}


@dataclass
class QueryKnowledgeHubConfig:
    """Configuration for query_knowledge_hub tool.
    
    Attributes:
        default_top_k: Default number of results if not specified
        max_top_k: Maximum allowed top_k value
        default_collection: Default collection if not specified
        enable_rerank: Whether to apply reranking
    """
    default_top_k: int = 5
    max_top_k: int = 20
    default_collection: str = "default"
    enable_rerank: bool = True


class QueryKnowledgeHubTool:
    """MCP Tool for knowledge base queries.
    
    This class encapsulates the query_knowledge_hub tool logic,
    coordinating HybridSearch and Reranker to produce formatted results.
    
    Design Principles:
    - Lazy initialization: Components created on first use
    - Error resilience: Graceful handling of search/rerank failures
    - Configurable: All parameters from settings.yaml
    
    Example:
        >>> tool = QueryKnowledgeHubTool(settings)
        >>> result = await tool.execute(query="Azure 配置", top_k=5)
        >>> print(result.content)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[QueryKnowledgeHubConfig] = None,
        hybrid_search: Optional[HybridSearch] = None,
        reranker: Optional[CoreReranker] = None,
        response_builder: Optional[ResponseBuilder] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        grounded_answer_builder: Optional[GroundedAnswerBuilder] = None,
        hallucination_guard: Optional[HallucinationGuard] = None,
    ) -> None:
        """Initialize QueryKnowledgeHubTool.
        
        Args:
            settings: Application settings. If None, loaded from default path.
            config: Tool configuration. If None, uses defaults.
            hybrid_search: Optional pre-configured HybridSearch instance.
            reranker: Optional pre-configured CoreReranker instance.
            response_builder: Optional pre-configured ResponseBuilder instance.
        """
        self._settings = settings
        self.config = config or QueryKnowledgeHubConfig()
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._embedding_client = None
        self._response_builder = response_builder or ResponseBuilder()
        self._answer_generator = answer_generator
        self._grounded_answer_builder = grounded_answer_builder or GroundedAnswerBuilder()
        self._hallucination_guard = hallucination_guard or HallucinationGuard()
        self._query_cache: Optional[LruTtlCache[tuple[Hashable, ...], MCPToolResponse]] = None
        
        # Track initialization state
        self._initialized = False
        self._current_collection: Optional[str] = None

    def _get_query_cache(self) -> Optional[LruTtlCache[tuple[Hashable, ...], MCPToolResponse]]:
        cache_settings = getattr(getattr(self.settings, "performance", None), "query_cache", None)
        if cache_settings is None or not getattr(cache_settings, "enabled", True):
            return None

        if self._query_cache is None:
            self._query_cache = LruTtlCache(
                max_size=getattr(cache_settings, "max_size", 128),
                ttl_seconds=getattr(cache_settings, "ttl_seconds", 300),
            )
        return self._query_cache

    def clear_query_cache(self) -> None:
        """Clear cached query responses for this tool instance."""
        if self._query_cache is not None:
            self._query_cache.clear()

    def _build_cache_key(
        self,
        query: str,
        top_k: int,
        collection: str,
        mode: str,
        include_sources: bool,
        include_citations: bool,
        answer_style: str,
        language: str,
    ) -> tuple[Hashable, ...]:
        normalized_query = " ".join(query.split())
        return (
            normalized_query,
            top_k,
            collection,
            mode,
            include_sources,
            include_citations,
            answer_style,
            language,
            self.config.enable_rerank,
            getattr(self.settings.embedding, "provider", ""),
            getattr(self.settings.embedding, "model", ""),
            getattr(self.settings.rerank, "enabled", False),
            getattr(self.settings.rerank, "provider", ""),
            getattr(self.settings.rerank, "model", ""),
            getattr(self.settings.rerank, "top_k", 0),
        )
    
    @property
    def settings(self) -> Settings:
        """Get settings, loading if necessary."""
        if self._settings is None:
            self._settings = load_settings()
        return self._settings
    
    def _ensure_initialized(self, collection: str) -> None:
        """Ensure search components are initialized for the given collection.
        
        Caching strategy (balances speed vs freshness):
        - **Fully cached** (stateless, never go stale): embedding client,
          reranker, query processor, settings.
        - **Cached until collection changes**: vector store (ChromaDB
          PersistentClient reads from SQLite — sees data written by other
          processes), dense retriever, hybrid search.
        - **Auto-refreshes on every query**: BM25 sparse index — the
          ``SparseRetriever._ensure_index_loaded()`` always reloads from
          disk, so the cached SparseRetriever object is fine.
        
        Only when *collection* changes do we tear down and rebuild.
        
        Args:
            collection: Target collection name.
        """
        # Always rebuild vector_store and retriever components so that
        # data ingested by other processes (e.g. Dashboard) is visible
        # immediately without requiring an MCP Server restart.
        
        logger.info(f"Initializing query components for collection: {collection}")
        
        # Import here to avoid circular imports and allow lazy loading
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.core.query_engine.reranker import create_core_reranker
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        
        # === Fully cached components (stateless, never go stale) ===
        if self._embedding_client is None:
            self._embedding_client = EmbeddingFactory.create(self.settings)
        
        if self._reranker is None:
            self._reranker = create_core_reranker(settings=self.settings)
        
        # === Rebuild for new collection ===
        # ChromaDB PersistentClient uses SQLite under the hood —
        # concurrent readers see committed writes from other processes
        # (dashboard ingestion), so caching the client is safe.
        vector_store = VectorStoreFactory.create(
            self.settings,
            collection_name=collection,
        )
        
        dense_retriever = create_dense_retriever(
            settings=self.settings,
            embedding_client=self._embedding_client,
            vector_store=vector_store,
        )
        
        # BM25Indexer just holds the index dir path; the SparseRetriever
        # calls _ensure_index_loaded() on every search, which always
        # reloads from disk — so it picks up dashboard-written data.
        bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
        sparse_retriever = create_sparse_retriever(
            settings=self.settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection
        
        query_processor = QueryProcessor()
        self._hybrid_search = create_hybrid_search(
            settings=self.settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )
        
        self._current_collection = collection
        self._initialized = True
        logger.info(f"Query components initialized for collection: {collection}")
    
    async def execute(
        self,
        query: str,
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        mode: Optional[str] = None,
        include_sources: bool = True,
        include_citations: bool = True,
        answer_style: Optional[str] = None,
        language: str = "auto",
    ) -> MCPToolResponse:
        """Execute the query_knowledge_hub tool.
        
        Args:
            query: Search query string.
            top_k: Maximum results to return.
            collection: Target collection name.
            
        Returns:
            MCPToolResponse with formatted content and citations.
            
        Raises:
            ValueError: If query is empty or invalid.
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Apply defaults
        effective_top_k = min(
            top_k or self.config.default_top_k,
            self.config.max_top_k
        )
        effective_collection = collection or self.config.default_collection
        answer_settings = self._answer_generation_settings()
        effective_mode = self._normalize_mode(mode or getattr(answer_settings, "default_mode", "contexts"))
        effective_answer_style = self._normalize_answer_style(
            answer_style or getattr(answer_settings, "default_answer_style", "concise")
        )
        effective_language = self._normalize_language(language)
        
        logger.info(
            f"Executing query_knowledge_hub: query='{query[:50]}...', "
            f"top_k={effective_top_k}, collection={effective_collection}"
        )
        
        trace = TraceContext(trace_type="query")
        trace.metadata["query"] = query[:200]
        trace.metadata["top_k"] = effective_top_k
        trace.metadata["collection"] = effective_collection
        trace.metadata["source"] = "mcp"
        trace.metadata["mode"] = effective_mode
        trace.metadata["answer_generation_enabled"] = bool(getattr(answer_settings, "enabled", True))
        trace.metadata["answer_style"] = effective_answer_style
        trace.metadata["language"] = effective_language

        try:
            cache_key = self._build_cache_key(
                query,
                effective_top_k,
                effective_collection,
                effective_mode,
                include_sources,
                include_citations,
                effective_answer_style,
                effective_language,
            )
            query_cache = self._get_query_cache()
            _cache_t0 = time.monotonic()
            cached_response = query_cache.get(cache_key) if query_cache is not None else None
            _cache_elapsed = (time.monotonic() - _cache_t0) * 1000.0

            if cached_response is not None:
                cached_response.metadata["cache_hit"] = True
                trace.metadata["cache_hit"] = True
                trace.record_stage(
                    "query_cache",
                    {
                        "enabled": True,
                        "hit": True,
                        "key_fields": {
                            "top_k": effective_top_k,
                            "collection": effective_collection,
                            "mode": effective_mode,
                        },
                    },
                    elapsed_ms=_cache_elapsed,
                )
                TraceCollector().collect(trace)
                return cached_response

            trace.metadata["cache_hit"] = False
            trace.record_stage(
                "query_cache",
                {
                    "enabled": query_cache is not None,
                    "hit": False,
                    "key_fields": {
                        "top_k": effective_top_k,
                        "collection": effective_collection,
                        "mode": effective_mode,
                    },
                },
                elapsed_ms=_cache_elapsed,
            )

            # Initialize components for collection
            # Run blocking I/O (embedding API, ChromaDB, BM25) in a thread
            # to avoid blocking the async event loop / MCP stdio transport
            _init_t0 = time.monotonic()
            await asyncio.to_thread(self._ensure_initialized, effective_collection)
            _init_elapsed = (time.monotonic() - _init_t0) * 1000.0
            trace.record_stage("initialization", {
                "collection": effective_collection,
                "cold_start": _init_elapsed > 500,  # >500ms ≈ cold
            }, elapsed_ms=_init_elapsed)
            
            # Perform hybrid search (blocking: embedding API + DB queries)
            results = await asyncio.to_thread(
                self._perform_search, query, effective_top_k, trace,
            )
            
            # Apply reranking if enabled (may call LLM API)
            if self.config.enable_rerank and results:
                results = await asyncio.to_thread(
                    self._apply_rerank, query, results, effective_top_k, trace,
                )
            
            if effective_mode == "answer":
                response = await asyncio.to_thread(
                    self._build_answer_response,
                    query,
                    results,
                    effective_collection,
                    effective_answer_style,
                    effective_language,
                    include_sources,
                    include_citations,
                    trace,
                )
            else:
                response = self._build_contexts_response(
                    results=results,
                    query=query,
                    collection=effective_collection,
                    include_sources=include_sources,
                    include_citations=include_citations,
                    trace=trace,
                )
            response.metadata["cache_hit"] = False
            
            # Store final results in trace for dashboard display
            trace.metadata["final_results"] = [
                {
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "text": r.text or "",
                    "source": r.metadata.get("source_path", r.metadata.get("source", "")),
                    "title": r.metadata.get("title", ""),
                }
                for r in results
            ]

            logger.info(
                f"query_knowledge_hub completed: {len(results)} results, "
                f"is_empty={response.is_empty}"
            )

            if query_cache is not None and "error" not in response.metadata:
                query_cache.set(cache_key, response)
            
            TraceCollector().collect(trace)
            return response
            
        except Exception as e:
            logger.exception(f"query_knowledge_hub failed: {e}")
            TraceCollector().collect(trace)
            # Return error response
            return self._build_error_response(query, effective_collection, str(e))
    
    def _perform_search(
        self,
        query: str,
        top_k: int,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Perform hybrid search.
        
        Args:
            query: Search query.
            top_k: Maximum results.
            trace: Optional TraceContext for observability.
            
        Returns:
            List of RetrievalResult.
        """
        if self._hybrid_search is None:
            raise RuntimeError("HybridSearch not initialized")
        
        # Pull enough hybrid-search candidates for the reranker, then CoreReranker
        # will cap the actual rerank input with candidate_top_k.
        rerank_config = getattr(getattr(self._reranker, "config", None), "candidate_top_k", None)
        candidate_top_k = rerank_config if isinstance(rerank_config, int) and rerank_config > 0 else top_k * 2
        initial_top_k = max(top_k, candidate_top_k) if self.config.enable_rerank else top_k
        
        try:
            results = self._hybrid_search.search(
                query=query,
                top_k=initial_top_k,
                filters=None,
                trace=trace,
                return_details=False,
            )
            return results if isinstance(results, list) else results.results
        except Exception as e:
            logger.warning(f"Hybrid search failed: {e}")
            return []
    
    def _apply_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Apply reranking to search results.
        
        Args:
            query: Original query.
            results: Search results to rerank.
            top_k: Final number of results.
            trace: Optional TraceContext for observability.
            
        Returns:
            Reranked results (or original if reranking fails).
        """
        if self._reranker is None or not self._reranker.is_enabled:
            return results[:top_k]
        
        try:
            rerank_result = self._reranker.rerank(
                query=query,
                results=results,
                top_k=top_k,
                trace=trace,
            )
            
            if rerank_result.used_fallback:
                logger.warning(
                    f"Reranker fallback: {rerank_result.fallback_reason}"
                )
            
            return rerank_result.results
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return results[:top_k]

    def _build_contexts_response(
        self,
        *,
        results: List[RetrievalResult],
        query: str,
        collection: str,
        include_sources: bool,
        include_citations: bool,
        trace: Optional[Any],
    ) -> MCPToolResponse:
        answer_settings = self._answer_generation_settings()
        status = assess_retrieval_status(
            results,
            min_contexts=getattr(answer_settings, "min_contexts", 1),
            min_score=getattr(answer_settings, "min_score", 0.2),
        )
        contexts = contexts_from_results(results)
        response = self._response_builder.build(
            results=results,
            query=query,
            collection=collection,
        )
        if not include_citations:
            response.citations = []

        response.metadata.update({
            "mode": "contexts",
            "query": query,
            "collection": collection,
            "retrieval_status": status.value,
            "trace_id": getattr(trace, "trace_id", None),
            "results": [
                context.to_result_dict(index, include_citation=include_citations)
                for index, context in enumerate(contexts, start=1)
            ],
            "sources": self._source_dicts(contexts) if include_sources else [],
        })

        self._record_mode_trace(
            trace,
            mode="contexts",
            retrieval_status=status.value,
            returned_context_count=len(contexts),
        )
        return response

    def _build_answer_response(
        self,
        query: str,
        results: List[RetrievalResult],
        collection: str,
        answer_style: str,
        language: str,
        include_sources: bool,
        include_citations: bool,
        trace: Optional[Any],
    ) -> MCPToolResponse:
        answer_settings = self._answer_generation_settings()
        status = assess_retrieval_status(
            results,
            min_contexts=getattr(answer_settings, "min_contexts", 1),
            min_score=getattr(answer_settings, "min_score", 0.2),
        )
        contexts = contexts_from_results(
            results,
            max_context_chars=getattr(answer_settings, "max_context_chars", 8000),
        )

        warnings: list[str] = []
        fallback_reason: Optional[str] = None
        llm_latency_ms = 0.0

        if status == RetrievalStatus.NO_RESULTS:
            answer = self._no_results_answer(query, language)
            warnings.append("NO_RETRIEVAL_RESULTS")
            fallback_reason = "no_results"
        elif status == RetrievalStatus.INSUFFICIENT:
            answer = self._insufficient_answer(contexts, query, language)
            warnings.append("INSUFFICIENT_RETRIEVAL_RESULTS")
            fallback_reason = "insufficient_retrieval"
        elif not bool(getattr(answer_settings, "enabled", True)):
            answer = self._answer_generation_disabled_answer(contexts, query, language)
            warnings.append("ANSWER_GENERATION_DISABLED")
            fallback_reason = "answer_generation_disabled"
        else:
            try:
                generated = self._get_answer_generator().generate(
                    query=query,
                    contexts=contexts,
                    answer_style=answer_style,
                    language=language,
                    trace=trace,
                )
                answer = generated.answer
                llm_latency_ms = generated.llm_latency_ms
                warnings.extend(generated.warnings)
                fallback_reason = generated.fallback_reason
            except Exception as exc:
                logger.warning("Answer generation failed, using grounded fallback: %s", exc)
                answer = self._answer_generation_failed_answer(contexts, query, language)
                warnings.append("ANSWER_GENERATION_FAILED")
                fallback_reason = "answer_generation_failed"

        if bool(getattr(getattr(answer_settings, "hallucination_guard", None), "enabled", True)):
            guard_result = self._hallucination_guard.validate(answer, contexts, status)
            answer = guard_result.answer
            warnings.extend(guard_result.warnings)

        warnings = self._dedupe(warnings)
        grounded = self._grounded_answer_builder.build(
            query=query,
            generated_answer=answer,
            contexts=contexts,
            retrieval_status=status,
            collection=collection,
            trace_id=getattr(trace, "trace_id", None),
            warnings=warnings,
            include_sources=include_sources,
            include_citations=include_citations,
        )
        payload = grounded.to_dict()
        payload.update({
            "answer_style": answer_style,
            "language": language,
            "llm_latency_ms": round(llm_latency_ms, 2),
            "fallback_reason": fallback_reason,
        })

        citations = self._response_builder.citation_generator.generate(results)
        response = MCPToolResponse(
            content=self._format_answer_content(payload),
            citations=citations if include_citations else [],
            metadata=payload,
            is_empty=status == RetrievalStatus.NO_RESULTS,
        )

        self._record_mode_trace(
            trace,
            mode="answer",
            retrieval_status=status.value,
            used_chunk_ids=payload["used_chunk_ids"],
            answer_length=len(answer),
            citation_count=len(payload["citations"]),
            hallucination_warnings=warnings,
            llm_latency_ms=round(llm_latency_ms, 2),
            fallback_reason=fallback_reason,
        )
        return response

    def _get_answer_generator(self) -> AnswerGenerator:
        if self._answer_generator is None:
            answer_settings = self._answer_generation_settings()
            self._answer_generator = AnswerGenerator(
                settings=self.settings,
                timeout_seconds=getattr(answer_settings, "timeout_seconds", 20.0),
            )
        return self._answer_generator

    def _answer_generation_settings(self) -> Any:
        response_settings = getattr(self.settings, "response", None)
        return getattr(response_settings, "answer_generation", None) or type(
            "AnswerSettings",
            (),
            {
                "enabled": True,
                "default_mode": "contexts",
                "min_contexts": 1,
                "min_score": 0.2,
                "max_context_chars": 8000,
                "default_answer_style": "concise",
                "timeout_seconds": 20.0,
                "hallucination_guard": type("GuardSettings", (), {"enabled": True})(),
            },
        )()

    def _record_mode_trace(self, trace: Optional[Any], **metadata: Any) -> None:
        if trace is None:
            return
        trace.metadata.update(metadata)
        if hasattr(trace, "record_stage"):
            trace.record_stage("response_mode", metadata)

    def _format_answer_content(self, payload: Dict[str, Any]) -> str:
        lines = ["## Answer", "", payload.get("answer", "")]
        citations = payload.get("citations") or []
        if citations:
            lines.extend(["", "## Citations"])
            for citation in citations:
                page = citation.get("page")
                page_text = "" if page is None else f", page={page}"
                lines.append(
                    f"[{citation['citation_id']}] source={citation['source']}"
                    f"{page_text}, chunk_id={citation['chunk_id']}"
                )
        if payload.get("warnings"):
            lines.extend(["", "## Warnings"])
            lines.extend(f"- {warning}" for warning in payload["warnings"])
        return "\n".join(lines)

    def _no_results_answer(self, query: str, language: str) -> str:
        if self._use_english(query, language):
            return (
                "I could not retrieve enough relevant information from the knowledge base, "
                "so I cannot answer this question based on the current corpus."
            )
        return "未在知识库中检索到足够相关的信息，因此无法基于当前知识库回答该问题。"

    def _insufficient_answer(
        self,
        contexts: list[Any],
        query: str,
        language: str,
    ) -> str:
        if self._use_english(query, language):
            prefix = (
                "The current knowledge base does not contain enough strong evidence. "
                "The following is only a brief summary of possibly relevant context:"
            )
        else:
            prefix = "当前知识库中的相关信息不足，以下仅是可能相关内容的简要概括："
        return self._context_summary(prefix, contexts)

    def _answer_generation_disabled_answer(
        self,
        contexts: list[Any],
        query: str,
        language: str,
    ) -> str:
        prefix = (
            "Answer generation is disabled; returning grounded context summary:"
            if self._use_english(query, language)
            else "服务端答案生成已关闭，返回基于检索上下文的概括："
        )
        return self._context_summary(prefix, contexts)

    def _answer_generation_failed_answer(
        self,
        contexts: list[Any],
        query: str,
        language: str,
    ) -> str:
        prefix = (
            "The answer generation service is unavailable; returning grounded context summary:"
            if self._use_english(query, language)
            else "答案生成服务暂不可用，返回基于检索上下文的概括："
        )
        return self._context_summary(prefix, contexts)

    def _context_summary(self, prefix: str, contexts: list[Any]) -> str:
        lines = [prefix]
        for context in contexts[:3]:
            snippet = " ".join((context.text or "").split())[:220]
            lines.append(f"- {snippet} [{context.citation_id}]")
        return "\n".join(lines)

    def _source_dicts(self, contexts: list[Any]) -> list[dict[str, Any]]:
        seen = set()
        sources = []
        for context in contexts:
            key = (context.source, context.page, context.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            sources.append(context.to_source_dict())
        return sources

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        return mode if mode in {"contexts", "answer"} else "contexts"

    @staticmethod
    def _normalize_answer_style(answer_style: str) -> str:
        return answer_style if answer_style in {"concise", "detailed", "bullet"} else "concise"

    @staticmethod
    def _normalize_language(language: str) -> str:
        return language if language in {"auto", "zh", "en"} else "auto"

    @staticmethod
    def _use_english(query: str, language: str) -> bool:
        if language == "en":
            return True
        if language == "zh":
            return False
        return all(ord(char) < 128 for char in query)

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen = set()
        output = []
        for item in items:
            if item not in seen:
                seen.add(item)
                output.append(item)
        return output
    
    def _build_error_response(
        self,
        query: str,
        collection: str,
        error_message: str,
    ) -> MCPToolResponse:
        """Build error response.
        
        Args:
            query: Original query.
            collection: Target collection.
            error_message: Error description.
            
        Returns:
            MCPToolResponse indicating error.
        """
        content = f"## 查询失败\n\n"
        content += f"查询: **{query}**\n"
        content += f"集合: `{collection}`\n\n"
        content += f"**错误信息:** {error_message}\n\n"
        content += "请检查:\n"
        content += "- 数据库连接是否正常\n"
        content += "- 集合是否已创建并包含数据\n"
        content += "- 配置文件是否正确\n"
        
        return MCPToolResponse(
            content=content,
            citations=[],
            metadata={
                "query": query,
                "collection": collection,
                "error": error_message,
            },
            is_empty=True,
        )


# Module-level tool instance (lazy-initialized)
_tool_instance: Optional[QueryKnowledgeHubTool] = None


def get_tool_instance(settings: Optional[Settings] = None) -> QueryKnowledgeHubTool:
    """Get or create the tool instance.
    
    Args:
        settings: Optional settings to use for initialization.
        
    Returns:
        QueryKnowledgeHubTool instance.
    """
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = QueryKnowledgeHubTool(settings=settings)
    return _tool_instance


async def query_knowledge_hub_handler(
    query: str,
    top_k: int = 5,
    collection: Optional[str] = None,
    mode: str = "contexts",
    answer_style: str = "concise",
    language: str = "auto",
    include_sources: bool = True,
    include_citations: bool = True,
) -> types.CallToolResult:
    """Handler function for MCP tool registration.
    
    This function is registered with the ProtocolHandler and called
    when the MCP client invokes the query_knowledge_hub tool.
    
    Supports multimodal responses - if search results contain images,
    the response will include ImageContent blocks alongside TextContent.
    
    Args:
        query: Search query string.
        top_k: Maximum number of results.
        collection: Optional collection name.
        
    Returns:
        MCP CallToolResult with content blocks (text and optionally images).
    """
    tool = get_tool_instance()
    
    try:
        response = await tool.execute(
            query=query,
            top_k=top_k,
            collection=collection,
            mode=mode,
            answer_style=answer_style,
            language=language,
            include_sources=include_sources,
            include_citations=include_citations,
        )
        
        # Use to_mcp_content() which handles multimodal (text + images)
        content_blocks = response.to_mcp_content()
        
        return types.CallToolResult(
            content=content_blocks,
            isError=response.is_empty and "error" in response.metadata,
        )
        
    except ValueError as e:
        # Invalid parameters
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"参数错误: {e}",
                )
            ],
            isError=True,
        )
    except Exception as e:
        # Internal error
        logger.exception(f"query_knowledge_hub handler error: {e}")
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"内部错误: 查询处理失败",
                )
            ],
            isError=True,
        )


def register_tool(protocol_handler) -> None:
    """Register query_knowledge_hub tool with the protocol handler.
    
    Args:
        protocol_handler: ProtocolHandler instance to register with.
    """
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=query_knowledge_hub_handler,
    )
    logger.info(f"Registered MCP tool: {TOOL_NAME}")
