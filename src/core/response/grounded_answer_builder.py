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
    refused: bool = False
    refusal_reason: Optional[str] = None
    confidence_score: float = 0.0
    confidence_factors: dict[str, float] = field(default_factory=dict)
    citation_verification: dict[str, Any] = field(default_factory=dict)
    unsupported_claim_count: int = 0
    citation_coverage: float = 0.0
    invalid_citations: list[str] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    source_conflicts: list[dict[str, Any]] = field(default_factory=list)
    rerank_metadata: dict[str, Any] = field(default_factory=dict)

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
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "confidence_score": self.confidence_score,
            "confidence_factors": self.confidence_factors,
            "citation_verification": self.citation_verification,
            "unsupported_claim_count": self.unsupported_claim_count,
            "citation_coverage": self.citation_coverage,
            "invalid_citations": self.invalid_citations,
            "claims": self.claims,
            "source_conflicts": self.source_conflicts,
            "rerank_metadata": self.rerank_metadata,
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
        citation_records: Optional[list[Any]] = None,
        refused: bool = False,
        refusal_reason: Optional[str] = None,
        confidence_result: Optional[Any] = None,
        citation_verification: Optional[Any] = None,
        claims: Optional[list[Any]] = None,
        source_conflicts: Optional[list[dict[str, Any]]] = None,
        rerank_metadata: Optional[dict[str, Any]] = None,
    ) -> GroundedAnswer:
        status = RetrievalStatus(retrieval_status)
        confidence_result = confidence_result
        confidence = getattr(confidence_result, "label", None) or self._confidence(status)
        confidence_score = float(getattr(confidence_result, "score", 0.0) or 0.0)
        confidence_factors = dict(getattr(confidence_result, "factors", {}) or {})
        if citation_records is not None:
            citations = [
                record.to_dict() if hasattr(record, "to_dict") else dict(record)
                for record in citation_records
            ] if include_citations else []
        else:
            citations = [context.to_citation_dict() for context in contexts] if include_citations else []
        sources = self._unique_sources(contexts) if include_sources else []
        verification_dict = (
            citation_verification.to_dict()
            if hasattr(citation_verification, "to_dict")
            else dict(citation_verification or {})
        )
        claim_dicts = [
            claim.to_dict() if hasattr(claim, "to_dict") else dict(claim)
            for claim in (claims or [])
        ]

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
            warnings=_dedupe(warnings or []),
            refused=refused,
            refusal_reason=refusal_reason,
            confidence_score=confidence_score,
            confidence_factors=confidence_factors,
            citation_verification=verification_dict,
            unsupported_claim_count=int(verification_dict.get("unsupported_claim_count", 0) or 0),
            citation_coverage=float(verification_dict.get("citation_coverage", 0.0) or 0.0),
            invalid_citations=list(verification_dict.get("invalid_citations", []) or []),
            claims=claim_dicts,
            source_conflicts=list(source_conflicts or []),
            rerank_metadata=dict(rerank_metadata or {}),
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


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output
