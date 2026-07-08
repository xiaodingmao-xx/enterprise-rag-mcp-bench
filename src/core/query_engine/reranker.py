"""Core layer Reranker orchestrating libs.reranker backends with fallback support.

This module implements the CoreReranker class that:
1. Integrates with libs.reranker (LLM, CrossEncoder, None) via RerankerFactory
2. Provides graceful fallback when backend fails or times out
3. Converts RetrievalResult to/from reranker input/output format
4. Supports TraceContext for observability

Design Principles:
- Pluggable: Uses RerankerFactory to instantiate configured backend
- Config-Driven: Reads rerank settings from settings.yaml
- Graceful Fallback: Returns original order on backend failure
- Observable: TraceContext integration for debugging
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import BaseReranker, NoneReranker
from src.libs.reranker.reranker_factory import RerankerFactory

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


class RerankError(RuntimeError):
    """Raised when reranking fails."""


class RerankTimeoutError(RerankError):
    """Raised when reranking exceeds the configured timeout."""


@dataclass
class RerankConfig:
    """Configuration for CoreReranker.
    
    Attributes:
        enabled: Whether reranking is enabled
        top_k: Legacy output top-k value kept for compatibility
        candidate_top_k: Maximum number of hybrid-search candidates to rerank
        output_top_k: Number of results to return after reranking
        timeout: Legacy timeout field kept for compatibility
        timeout_seconds: Timeout for reranker backend (seconds)
        fallback_on_error: Whether to return original order on error
        fallback_on_timeout: Whether to fallback on timeout/error/invalid output
    """
    enabled: bool = True
    top_k: int = 5
    candidate_top_k: int = 20
    output_top_k: int = 0
    timeout: float = 30.0
    timeout_seconds: float = 0.0
    fallback_on_error: bool = True
    fallback_on_timeout: bool = True

    def __post_init__(self) -> None:
        self.top_k = max(1, int(self.top_k))
        self.candidate_top_k = max(1, int(self.candidate_top_k))
        self.output_top_k = max(1, int(self.output_top_k or self.top_k))
        self.timeout = max(0.001, float(self.timeout))
        self.timeout_seconds = max(
            0.001,
            float(self.timeout_seconds or self.timeout),
        )


@dataclass
class RerankResult:
    """Result of a rerank operation.
    
    Attributes:
        results: Reranked list of RetrievalResults
        used_fallback: Whether fallback was used due to backend failure
        fallback_reason: Reason for fallback (if applicable)
        reranker_type: Type of reranker used ('llm', 'cross_encoder', 'none')
        original_order: Original results before reranking (for debugging)
    """
    results: List[RetrievalResult] = field(default_factory=list)
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    reranker_type: str = "none"
    original_order: Optional[List[RetrievalResult]] = None


class CoreReranker:
    """Core layer Reranker with fallback support.
    
    This class wraps libs.reranker implementations and provides:
    1. Type conversion between RetrievalResult and reranker dict format
    2. Graceful fallback when backend fails
    3. Configuration-driven backend selection
    4. TraceContext integration
    
    Design Principles Applied:
    - Pluggable: Backend via RerankerFactory
    - Config-Driven: All parameters from settings
    - Fallback: Returns original order on failure
    - Observable: TraceContext support
    
    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings("config/settings.yaml")
        >>> reranker = CoreReranker(settings)
        >>> results = [RetrievalResult(chunk_id="1", score=0.8, text="...", metadata={})]
        >>> reranked = reranker.rerank("query", results)
        >>> print(reranked.results)
    """
    
    def __init__(
        self,
        settings: Settings,
        reranker: Optional[BaseReranker] = None,
        config: Optional[RerankConfig] = None,
    ) -> None:
        """Initialize CoreReranker.
        
        Args:
            settings: Application settings containing rerank configuration.
            reranker: Optional reranker backend. If None, creates via RerankerFactory.
            config: Optional RerankConfig. If None, extracts from settings.
        """
        self.settings = settings
        
        # Extract config from settings or use provided
        if config is not None:
            self.config = config
        else:
            self.config = self._extract_config(settings)
        
        # Initialize reranker backend
        if reranker is not None:
            self._reranker = reranker
        elif not self.config.enabled:
            self._reranker = NoneReranker(settings=settings)
        else:
            try:
                self._reranker = RerankerFactory.create(settings)
            except Exception as e:
                logger.warning(f"Failed to create reranker, using NoneReranker: {e}")
                self._reranker = NoneReranker(settings=settings)
        
        # Determine reranker type for result reporting
        self._reranker_type = self._get_reranker_type()
    
    def _extract_config(self, settings: Settings) -> RerankConfig:
        """Extract RerankConfig from settings.
        
        Args:
            settings: Application settings.
            
        Returns:
            RerankConfig with values from settings.
        """
        try:
            rerank_settings = settings.rerank
            if rerank_settings is None:
                return RerankConfig(enabled=False)

            top_k = self._positive_int(getattr(rerank_settings, "top_k", 5), 5)
            timeout_seconds = self._positive_float(
                getattr(
                    rerank_settings,
                    "timeout_seconds",
                    getattr(rerank_settings, "timeout", 30.0),
                ),
                30.0,
            )
            fallback_on_timeout = bool(
                getattr(
                    rerank_settings,
                    "fallback_on_timeout",
                    getattr(rerank_settings, "fallback_on_error", True),
                )
            )
            return RerankConfig(
                enabled=bool(rerank_settings.enabled),
                top_k=top_k,
                candidate_top_k=self._positive_int(
                    getattr(rerank_settings, "candidate_top_k", max(top_k * 2, top_k)),
                    max(top_k * 2, top_k),
                ),
                output_top_k=self._positive_int(
                    getattr(rerank_settings, "output_top_k", top_k),
                    top_k,
                ),
                timeout=timeout_seconds,
                timeout_seconds=timeout_seconds,
                fallback_on_error=bool(
                    getattr(rerank_settings, "fallback_on_error", fallback_on_timeout)
                ),
                fallback_on_timeout=fallback_on_timeout,
            )
        except AttributeError:
            logger.warning("Missing rerank configuration, using defaults (disabled)")
            return RerankConfig(enabled=False)

    @staticmethod
    def _positive_int(value: Any, default: int) -> int:
        if isinstance(value, bool):
            parsed = default
        elif isinstance(value, int):
            parsed = value
        elif isinstance(value, str) and value.strip():
            try:
                parsed = int(value)
            except ValueError:
                parsed = default
        else:
            parsed = default
        return max(1, parsed)

    @staticmethod
    def _positive_float(value: Any, default: float) -> float:
        if isinstance(value, bool):
            parsed = default
        elif isinstance(value, (int, float)):
            parsed = float(value)
        elif isinstance(value, str) and value.strip():
            try:
                parsed = float(value)
            except ValueError:
                parsed = default
        else:
            parsed = default
        return max(0.001, parsed)
    
    def _get_reranker_type(self) -> str:
        """Get the type name of the current reranker backend.
        
        Returns:
            String identifier for the reranker type.
        """
        class_name = self._reranker.__class__.__name__
        if "LLM" in class_name:
            return "llm"
        elif "CrossEncoder" in class_name:
            return "cross_encoder"
        elif "Bailian" in class_name:
            return "bailian"
        elif "None" in class_name:
            return "none"
        else:
            return class_name.lower()
    
    def _results_to_candidates(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Convert RetrievalResults to reranker candidate format.
        
        Args:
            results: List of RetrievalResult objects.
            
        Returns:
            List of dicts suitable for reranker input.
        """
        candidates = []
        for result in results:
            candidates.append({
                "id": result.chunk_id,
                "text": result.text,
                "score": result.score,
                "metadata": result.metadata.copy(),
            })
        return candidates
    
    def _candidates_to_results(
        self,
        candidates: List[Dict[str, Any]],
        original_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Convert reranked candidates back to RetrievalResults.
        
        Args:
            candidates: Reranked candidates from reranker.
            original_results: Original results for reference.
            
        Returns:
            List of RetrievalResult in reranked order.
        """
        # Build lookup from original results
        id_to_original = {r.chunk_id: r for r in original_results}
        
        results = []
        for candidate in candidates:
            chunk_id = candidate["id"]
            
            # Get original result or build new one
            if chunk_id in id_to_original:
                original = id_to_original[chunk_id]
                # Create new result with updated score
                rerank_score = candidate.get("rerank_score", candidate.get("score", 0.0))
                results.append(RetrievalResult(
                    chunk_id=original.chunk_id,
                    score=rerank_score,
                    text=original.text,
                    metadata={
                        **original.metadata,
                        "original_score": original.score,
                        "rerank_score": rerank_score,
                        "reranked": True,
                    },
                ))
            else:
                # Candidate not in original - build from candidate data
                results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    score=candidate.get("rerank_score", candidate.get("score", 0.0)),
                    text=candidate.get("text", ""),
                    metadata=candidate.get("metadata", {}),
                ))
        
        return results

    def _call_backend_with_timeout(
        self,
        *,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any],
        timeout_seconds: float,
        kwargs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Run the synchronous provider call behind a real timeout."""
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="core-rerank")
        future = executor.submit(
            self._reranker.rerank,
            query=query,
            candidates=candidates,
            trace=trace,
            **kwargs,
        )
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise RerankTimeoutError(
                f"Reranking timed out after {timeout_seconds:.3f}s"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _validate_rerank_output(
        self,
        reranked_candidates: Any,
        input_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Validate provider output before converting it back to results."""
        if not isinstance(reranked_candidates, list):
            raise RerankError(
                f"Reranker returned {type(reranked_candidates).__name__}, expected list"
            )
        if input_candidates and not reranked_candidates:
            raise RerankError("Reranker returned no candidates")

        for index, candidate in enumerate(reranked_candidates):
            if not isinstance(candidate, dict):
                raise RerankError(
                    f"Reranker output at index {index} is "
                    f"{type(candidate).__name__}, expected dict"
                )
            if "id" not in candidate:
                raise RerankError(f"Reranker output at index {index} missing 'id'")

        return reranked_candidates

    def _record_rerank_trace(
        self,
        trace: Optional[Any],
        *,
        data: Dict[str, Any],
        elapsed_ms: Optional[float],
    ) -> None:
        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage("rerank", data, elapsed_ms=elapsed_ms)

    def _record_fallback_metadata(
        self,
        trace: Optional[Any],
        *,
        fallback_reason: str,
        timeout_seconds: float,
        candidate_count: int,
    ) -> None:
        metadata = getattr(trace, "metadata", None)
        if isinstance(metadata, dict):
            metadata["rerank_fallback"] = True
            metadata["fallback_reason"] = fallback_reason
            metadata["timeout_seconds"] = timeout_seconds
            metadata["candidate_count"] = candidate_count

    def _fallback_result(
        self,
        *,
        results: List[RetrievalResult],
        effective_top_k: int,
        reason: str,
        detail: str,
        timeout_seconds: float,
        candidate_count: int,
        elapsed_ms: float,
        trace: Optional[Any],
    ) -> RerankResult:
        """Build a safe fallback result in original hybrid-search order."""
        logger.warning(
            "Reranker fallback: reason=%s timeout_seconds=%.3f "
            "candidate_count=%s detail=%s",
            reason,
            timeout_seconds,
            candidate_count,
            detail,
        )

        self._record_fallback_metadata(
            trace,
            fallback_reason=reason,
            timeout_seconds=timeout_seconds,
            candidate_count=candidate_count,
        )

        fallback_results = []
        for result in results[:effective_top_k]:
            fallback_results.append(RetrievalResult(
                chunk_id=result.chunk_id,
                score=result.score,
                text=result.text,
                metadata={
                    **result.metadata,
                    "reranked": False,
                    "rerank_fallback": True,
                    "fallback_reason": reason,
                    "timeout_seconds": timeout_seconds,
                    "candidate_count": candidate_count,
                },
            ))

        self._record_rerank_trace(
            trace,
            data={
                "method": self._reranker_type,
                "provider": self._reranker_type,
                "input_count": len(results),
                "candidate_count": candidate_count,
                "candidate_top_k": self.config.candidate_top_k,
                "output_top_k": effective_top_k,
                "output_count": len(fallback_results),
                "timeout_seconds": timeout_seconds,
                "used_fallback": True,
                "fallback_reason": reason,
                "fallback_detail": detail,
            },
            elapsed_ms=elapsed_ms,
        )

        return RerankResult(
            results=fallback_results,
            used_fallback=True,
            fallback_reason=f"{reason}: {detail}",
            reranker_type=self._reranker_type,
            original_order=results[:],
        )
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> RerankResult:
        """Rerank retrieval results using configured backend.
        
        Args:
            query: The user query string.
            results: List of RetrievalResult objects to rerank.
            top_k: Number of results to return. If None, uses config.top_k.
            trace: Optional TraceContext for observability.
            **kwargs: Additional parameters passed to reranker backend.
            
        Returns:
            RerankResult containing reranked results and metadata.
        """
        effective_top_k = self._positive_int(
            top_k if top_k is not None else self.config.output_top_k,
            self.config.output_top_k,
        )
        
        # Early return for empty or single results
        if not results:
            return RerankResult(
                results=[],
                used_fallback=False,
                reranker_type=self._reranker_type,
            )
        
        if len(results) == 1:
            return RerankResult(
                results=results[:],
                used_fallback=False,
                reranker_type=self._reranker_type,
            )
        
        # If reranking disabled, return top_k results in original order
        if not self.config.enabled or isinstance(self._reranker, NoneReranker):
            return RerankResult(
                results=results[:effective_top_k],
                used_fallback=False,
                reranker_type="none",
                original_order=results[:],
            )
        
        # Only rerank the first candidate_top_k hybrid-search candidates.
        candidate_results = results[:self.config.candidate_top_k]
        candidates = self._results_to_candidates(candidate_results)
        candidate_count = len(candidates)
        
        # Attempt reranking
        _t0 = time.monotonic()
        try:
            logger.debug(
                "Reranking %s/%s candidates with %s",
                candidate_count,
                len(results),
                self._reranker_type,
            )
            reranked_candidates = self._call_backend_with_timeout(
                query=query,
                candidates=candidates,
                trace=trace,
                timeout_seconds=self.config.timeout_seconds,
                kwargs=dict(kwargs),
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0

            reranked_candidates = self._validate_rerank_output(
                reranked_candidates,
                candidates,
            )
            
            # Convert back to RetrievalResult
            reranked_results = self._candidates_to_results(
                reranked_candidates,
                candidate_results,
            )
            
            # Apply output top-k limit.
            final_results = reranked_results[:effective_top_k]
            
            logger.info(f"Reranking complete: {len(final_results)} results returned")
            
            self._record_rerank_trace(
                trace,
                data={
                    "method": self._reranker_type,
                    "provider": self._reranker_type,
                    "input_count": len(results),
                    "candidate_count": candidate_count,
                    "candidate_top_k": self.config.candidate_top_k,
                    "output_top_k": effective_top_k,
                    "output_count": len(final_results),
                    "timeout_seconds": self.config.timeout_seconds,
                    "used_fallback": False,
                    "chunks": [
                        {
                            "chunk_id": r.chunk_id,
                            "score": round(r.score, 4),
                            "text": r.text or "",
                            "source": r.metadata.get("source_path", r.metadata.get("source", "")),
                        }
                        for r in final_results
                    ],
                },
                elapsed_ms=_elapsed,
            )
            
            return RerankResult(
                results=final_results,
                used_fallback=False,
                reranker_type=self._reranker_type,
                original_order=results[:],
            )
            
        except Exception as e:
            elapsed_ms = (time.monotonic() - _t0) * 1000.0
            if isinstance(e, RerankTimeoutError):
                reason = "timeout"
            elif isinstance(e, RerankError):
                reason = "invalid_rerank_output"
            else:
                reason = "rerank_error"

            if self.config.fallback_on_timeout and self.config.fallback_on_error:
                return self._fallback_result(
                    results=results,
                    effective_top_k=effective_top_k,
                    reason=reason,
                    detail=str(e),
                    timeout_seconds=self.config.timeout_seconds,
                    candidate_count=candidate_count,
                    elapsed_ms=elapsed_ms,
                    trace=trace,
                )

            self._record_rerank_trace(
                trace,
                data={
                    "method": self._reranker_type,
                    "provider": self._reranker_type,
                    "input_count": len(results),
                    "candidate_count": candidate_count,
                    "candidate_top_k": self.config.candidate_top_k,
                    "output_top_k": effective_top_k,
                    "output_count": 0,
                    "timeout_seconds": self.config.timeout_seconds,
                    "used_fallback": False,
                    "fallback_reason": reason,
                    "fallback_detail": str(e),
                },
                elapsed_ms=elapsed_ms,
            )
            raise RerankError(f"Reranking failed and fallback disabled: {e}") from e
    
    @property
    def reranker_type(self) -> str:
        """Get the type of the current reranker backend."""
        return self._reranker_type
    
    @property
    def is_enabled(self) -> bool:
        """Check if reranking is enabled."""
        return self.config.enabled and not isinstance(self._reranker, NoneReranker)


def create_core_reranker(
    settings: Settings,
    reranker: Optional[BaseReranker] = None,
) -> CoreReranker:
    """Factory function to create a CoreReranker instance.
    
    Args:
        settings: Application settings.
        reranker: Optional reranker backend override.
        
    Returns:
        Configured CoreReranker instance.
    """
    return CoreReranker(settings=settings, reranker=reranker)
