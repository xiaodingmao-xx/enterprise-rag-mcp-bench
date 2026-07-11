"""Multi-factor confidence scoring for grounded answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional

from src.core.response.retrieval_status import RetrievalStatus


@dataclass
class AnswerConfidence:
    label: str
    score: float
    factors: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnswerConfidenceScorer:
    def __init__(self, high_threshold: float = 0.75, medium_threshold: float = 0.45) -> None:
        self.high_threshold = max(0.0, min(1.0, float(high_threshold)))
        self.medium_threshold = max(0.0, min(self.high_threshold, float(medium_threshold)))

    def score(
        self,
        *,
        answer: str,
        results: Iterable[Any],
        contexts: Iterable[Any],
        retrieval_status: str | RetrievalStatus,
        citation_result: Optional[Any] = None,
        source_conflicts: Optional[Iterable[Any]] = None,
        refused: bool = False,
        rerank_result: Optional[Any] = None,
    ) -> AnswerConfidence:
        result_list = list(results)
        context_list = list(contexts)
        status = RetrievalStatus(retrieval_status)
        warnings: list[str] = []
        top_retrieval = max((self._number(getattr(item, "score", 0.0)) for item in result_list), default=0.0)
        average_retrieval = (
            sum(self._number(getattr(item, "score", 0.0)) for item in result_list) / len(result_list)
            if result_list else 0.0
        )
        top_rerank = max(
            (self._number((getattr(item, "metadata", {}) or {}).get("rerank_score", getattr(item, "score", 0.0))) for item in result_list),
            default=0.0,
        )
        if rerank_result is not None:
            ranked = list(getattr(rerank_result, "results", []) or [])
            top_rerank = max((self._number(getattr(item, "score", 0.0)) for item in ranked), default=top_rerank)
        supporting_count = float(getattr(citation_result, "supported_claim_count", 0) or 0)
        citation_coverage = self._number(getattr(citation_result, "citation_coverage", 0.0)) if citation_result else 0.0
        unsupported_ratio = self._number(getattr(citation_result, "unsupported_claim_ratio", 0.0)) if citation_result else 0.0
        conflict_count = len(list(source_conflicts or []))
        source_agreement = 0.0 if conflict_count else (1.0 if len(context_list) > 1 else 0.7 if context_list else 0.0)
        answer_length = self._answer_length_factor(answer, citation_coverage)
        retrieval_factor = 1.0 if status == RetrievalStatus.SUFFICIENT else 0.35 if status == RetrievalStatus.INSUFFICIENT else 0.0
        refusal_factor = 0.0 if refused else 1.0

        factors = {
            "top_retrieval_score": max(0.0, min(1.0, top_retrieval)),
            "average_retrieval_score": max(0.0, min(1.0, average_retrieval)),
            "top_rerank_score": max(0.0, min(1.0, top_rerank)),
            "supporting_chunk_count": min(1.0, supporting_count / 3.0),
            "citation_coverage": max(0.0, min(1.0, citation_coverage)),
            "unsupported_claim_ratio": max(0.0, min(1.0, unsupported_ratio)),
            "source_agreement": source_agreement,
            "answer_length": answer_length,
            "retrieval_status": retrieval_factor,
            "refusal_status": refusal_factor,
        }
        score = (
            factors["top_retrieval_score"] * 0.14
            + factors["average_retrieval_score"] * 0.10
            + factors["top_rerank_score"] * 0.12
            + factors["supporting_chunk_count"] * 0.12
            + factors["citation_coverage"] * 0.18
            + (1.0 - factors["unsupported_claim_ratio"]) * 0.12
            + factors["source_agreement"] * 0.08
            + factors["answer_length"] * 0.04
            + factors["retrieval_status"] * 0.06
            + factors["refusal_status"] * 0.04
        )
        if refused or status == RetrievalStatus.NO_RESULTS or not context_list:
            score = 0.0
        if conflict_count:
            score *= 0.7
            warnings.append("CONFLICTING_SOURCES")
        if citation_result is None:
            warnings.append("CONFIDENCE_CITATION_DATA_MISSING")
        score = max(0.0, min(1.0, score))
        label = "high" if score >= self.high_threshold else "medium" if score >= self.medium_threshold else "low"
        return AnswerConfidence(label=label, score=round(score, 4), factors=factors, warnings=_dedupe(warnings))

    @staticmethod
    def _number(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _answer_length_factor(answer: str, citation_coverage: float) -> float:
        length = len(answer or "")
        if length == 0:
            return 0.0
        if length <= 800:
            return 1.0
        penalty = min(0.7, (length - 800) / 4000)
        return max(0.0, 1.0 - penalty) * (0.6 + 0.4 * citation_coverage)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output

