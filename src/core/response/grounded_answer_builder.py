"""Build structured grounded-answer responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.core.response.retrieval_status import RetrievedContext, RetrievalStatus


@dataclass
class GroundedAnswer:
    """Structured answer payload returned by answer mode."""

    mode: str
    query: str
    collection: str
    answer: str
    retrieval_status: str
    confidence: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    used_chunk_ids: list[str] = field(default_factory=list)
    trace_id: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "query": self.query,
            "collection": self.collection,
            "answer": self.answer,
            "retrieval_status": self.retrieval_status,
            "confidence": self.confidence,
            "citations": self.citations,
            "sources": self.sources,
            "used_chunk_ids": self.used_chunk_ids,
            "trace_id": self.trace_id,
            "warnings": self.warnings,
        }


class GroundedAnswerBuilder:
    """Assemble final structured payloads for answer mode."""

    def build(
        self,
        query: str,
        generated_answer: str,
        contexts: list[RetrievedContext],
        retrieval_status: str | RetrievalStatus,
        collection: str = "default",
        trace_id: Optional[str] = None,
        warnings: Optional[list[str]] = None,
        include_sources: bool = True,
        include_citations: bool = True,
    ) -> GroundedAnswer:
        status = RetrievalStatus(retrieval_status)
        confidence = self._confidence(status)
        citations = [context.to_citation_dict() for context in contexts] if include_citations else []
        sources = self._unique_sources(contexts) if include_sources else []

        return GroundedAnswer(
            mode="answer",
            query=query,
            collection=collection,
            answer=generated_answer,
            retrieval_status=status.value,
            confidence=confidence,
            citations=citations,
            sources=sources,
            used_chunk_ids=[context.chunk_id for context in contexts],
            trace_id=trace_id,
            warnings=warnings or [],
        )

    @staticmethod
    def _confidence(status: RetrievalStatus) -> str:
        if status == RetrievalStatus.SUFFICIENT:
            return "medium"
        return "low"

    @staticmethod
    def _unique_sources(contexts: list[RetrievedContext]) -> list[dict[str, Any]]:
        seen = set()
        sources = []
        for context in contexts:
            key = (context.source, context.page, context.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            sources.append(context.to_source_dict())
        return sources
