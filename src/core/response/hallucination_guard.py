"""Lightweight hallucination guard for grounded answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.core.response.retrieval_status import RetrievedContext, RetrievalStatus


@dataclass
class GuardResult:
    """Validation result from HallucinationGuard."""

    answer: str
    retrieval_status: RetrievalStatus
    warnings: list[str] = field(default_factory=list)


class HallucinationGuard:
    """Rule-based guard for citations and unsupported phrasing."""

    MARKER_RE = re.compile(r"\[C(\d+)\]")
    EXTERNAL_PHRASES = (
        "according to my experience",
        "generally speaking",
        "in general",
        "i think",
        "i believe",
        "based on my knowledge",
        "根据我的经验",
        "一般来说",
        "我认为",
        "我觉得",
        "常识上",
    )

    def validate(
        self,
        answer: str,
        contexts: list[RetrievedContext],
        retrieval_status: str | RetrievalStatus,
    ) -> GuardResult:
        status = RetrievalStatus(retrieval_status)
        warnings: list[str] = []
        answer = answer or ""

        if not contexts and answer.strip():
            warnings.append("NO_CONTEXT_FOR_ANSWER")

        valid_markers = {context.citation_id for context in contexts}
        used_markers = {f"C{match}" for match in self.MARKER_RE.findall(answer)}
        invalid_markers = sorted(used_markers - valid_markers)
        if invalid_markers:
            warnings.append("INVALID_CITATION_MARKER:" + ",".join(invalid_markers))

        if contexts and not used_markers and answer.strip():
            warnings.append("MISSING_CITATION_MARKER")

        lowered = answer.lower()
        if any(phrase in lowered or phrase in answer for phrase in self.EXTERNAL_PHRASES):
            warnings.append("POSSIBLE_EXTERNAL_INFERENCE")

        if status == RetrievalStatus.NO_RESULTS:
            warnings.append("NO_RETRIEVAL_RESULTS")
        elif status == RetrievalStatus.INSUFFICIENT:
            warnings.append("INSUFFICIENT_RETRIEVAL_RESULTS")

        return GuardResult(answer=answer, retrieval_status=status, warnings=_dedupe(warnings))


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output
