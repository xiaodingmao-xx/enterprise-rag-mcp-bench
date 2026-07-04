"""Retrieval sufficiency helpers for grounded RAG responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional

from src.core.types import RetrievalResult


class RetrievalStatus(str, Enum):
    """Unified retrieval status values used by query responses."""

    NO_RESULTS = "no_results"
    INSUFFICIENT = "insufficient"
    SUFFICIENT = "sufficient"


@dataclass(frozen=True)
class RetrievedContext:
    """A retrieval result normalized for answer generation and citations."""

    citation_id: str
    chunk_id: str
    text: str
    score: float
    source: str = "unknown"
    page: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_result_dict(self, rank: int, include_citation: bool = True) -> dict[str, Any]:
        payload = {
            "rank": rank,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(float(self.score), 4),
            "source": self.source,
            "page": self.page,
            "metadata": dict(self.metadata),
        }
        if include_citation:
            payload["citation"] = self.format_citation()
        return payload

    def to_citation_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "snippet": _snippet(self.text, 240),
        }

    def to_source_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
        }

    def format_citation(self) -> str:
        if self.page is not None:
            return f"[{self.source}, p.{self.page}]"
        return f"[{self.source}]"


def assess_retrieval_status(
    results: list[RetrievalResult],
    *,
    min_contexts: int = 1,
    min_score: float = 0.2,
) -> RetrievalStatus:
    """Classify whether retrieval results are sufficient for answering."""
    if not results:
        return RetrievalStatus.NO_RESULTS
    if len(results) < max(1, min_contexts):
        return RetrievalStatus.INSUFFICIENT

    top_score = max(float(result.score or 0.0) for result in results)
    if top_score < min_score:
        return RetrievalStatus.INSUFFICIENT
    return RetrievalStatus.SUFFICIENT


def contexts_from_results(
    results: Iterable[RetrievalResult],
    *,
    max_context_chars: Optional[int] = None,
) -> list[RetrievedContext]:
    """Convert RetrievalResult objects into citation-ready contexts."""
    contexts: list[RetrievedContext] = []
    remaining = max_context_chars if max_context_chars and max_context_chars > 0 else None

    for index, result in enumerate(results, start=1):
        text = result.text or ""
        if remaining is not None:
            if remaining <= 0:
                break
            text = text[:remaining]
            remaining -= len(text)

        metadata = dict(result.metadata or {})
        page = _coerce_page(metadata.get("page", metadata.get("page_num")))
        source = (
            metadata.get("source_path")
            or metadata.get("source")
            or metadata.get("title")
            or "unknown"
        )
        contexts.append(
            RetrievedContext(
                citation_id=f"C{index}",
                chunk_id=result.chunk_id,
                text=text,
                score=float(result.score or 0.0),
                source=str(source),
                page=page,
                metadata=metadata,
            )
        )

    return contexts


def _coerce_page(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _snippet(text: str, max_chars: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0] + "..."
