"""Safe, language-aware refusal decisions for grounded answers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from src.core.response.retrieval_status import RetrievalStatus


@dataclass
class RefusalDecision:
    should_refuse: bool
    reason: Optional[str]
    message: Optional[str]
    warnings: list[str] = field(default_factory=list)
    retryable: bool = False


@dataclass
class RefusalPolicyConfig:
    enabled: bool = True
    refuse_on_no_context: bool = True
    refuse_on_low_score: bool = True
    refuse_on_no_valid_citation: bool = True
    refuse_on_prompt_injection: bool = True
    refuse_on_unsupported_claims: bool = True
    low_score_threshold: float = 0.2
    max_unsupported_claim_ratio: float = 0.35
    min_citation_coverage: float = 0.5


class RefusalPolicy:
    def __init__(self, config: Optional[RefusalPolicyConfig] = None) -> None:
        self.config = config or RefusalPolicyConfig()

    def decide(
        self,
        *,
        query: str,
        results: Iterable[Any] = (),
        retrieval_status: str | RetrievalStatus = RetrievalStatus.SUFFICIENT,
        citation_result: Optional[Any] = None,
        warnings: Iterable[str] = (),
        prompt_injection: bool = False,
        permission_denied: bool = False,
        out_of_scope: bool = False,
        answer: str = "",
    ) -> RefusalDecision:
        current_warnings = _dedupe(list(warnings))
        status = RetrievalStatus(retrieval_status)
        result_list = list(results)
        if not self.config.enabled:
            return RefusalDecision(False, None, None, current_warnings)

        if not result_list or status == RetrievalStatus.NO_RESULTS:
            if self.config.refuse_on_no_context:
                return self._decision(query, "NO_RETRIEVAL_RESULTS", current_warnings)

        top_score = max((float(getattr(item, "score", 0.0) or 0.0) for item in result_list), default=0.0)
        if self.config.refuse_on_low_score and result_list and top_score < self.config.low_score_threshold:
            return self._decision(query, "LOW_RETRIEVAL_SCORE", _append(current_warnings, "LOW_RETRIEVAL_SCORE"))

        if permission_denied or "UNAUTHORIZED_CITATION_REMOVED" in current_warnings:
            return self._decision(query, "PERMISSION_DENIED", current_warnings)
        if out_of_scope:
            return self._decision(query, "OUT_OF_SCOPE", current_warnings)
        if prompt_injection and self.config.refuse_on_prompt_injection:
            return self._decision(query, "PROMPT_INJECTION_DETECTED", _append(current_warnings, "PROMPT_INJECTION_DETECTED"))

        if citation_result is not None:
            valid_citations = list(getattr(citation_result, "valid_citations", []) or [])
            unsupported_ratio = float(getattr(citation_result, "unsupported_claim_ratio", 0.0) or 0.0)
            coverage = float(getattr(citation_result, "citation_coverage", 0.0) or 0.0)
            claim_count = len(getattr(citation_result, "claim_verifications", []) or [])
            if self.config.refuse_on_no_valid_citation and result_list and claim_count and not valid_citations:
                return self._decision(query, "NO_VALID_CITATIONS", _append(current_warnings, "NO_VALID_CITATIONS"))
            if self.config.refuse_on_unsupported_claims and unsupported_ratio > self.config.max_unsupported_claim_ratio:
                return self._decision(query, "UNSUPPORTED_GENERATED_CLAIMS", _append(current_warnings, "UNSUPPORTED_GENERATED_CLAIMS"))
            if (
                self.config.refuse_on_no_valid_citation
                and result_list
                and claim_count
                and coverage < self.config.min_citation_coverage
                and not answer_is_refusal(answer)
            ):
                return self._decision(query, "NO_VALID_CITATIONS", _append(current_warnings, "NO_VALID_CITATIONS"))

        if "CONFLICTING_SOURCES" in current_warnings:
            current_warnings = _append(current_warnings, "CONFLICTING_SOURCES")
        return RefusalDecision(False, None, None, current_warnings)

    def _decision(self, query: str, reason: str, warnings: list[str]) -> RefusalDecision:
        return RefusalDecision(
            should_refuse=True,
            reason=reason,
            message=self.message_for(query, reason),
            warnings=_append(warnings, reason),
            retryable=reason in {"NO_RETRIEVAL_RESULTS", "LOW_RETRIEVAL_SCORE"},
        )

    @staticmethod
    def message_for(query: str, reason: str) -> str:
        english = bool(query) and all(ord(char) < 128 for char in query)
        if english:
            messages = {
                "NO_RETRIEVAL_RESULTS": "I could not find sufficient evidence in the knowledge base to answer this question.",
                "LOW_RETRIEVAL_SCORE": "The retrieved evidence is too weak to answer this question reliably.",
                "NO_VALID_CITATIONS": "I cannot provide a reliable answer because the available evidence could not be cited.",
                "PERMISSION_DENIED": "I cannot provide an answer from documents outside your access scope.",
                "OUT_OF_SCOPE": "This question is outside the scope of the current knowledge base.",
                "PROMPT_INJECTION_DETECTED": "I cannot follow instructions that conflict with the knowledge-base safety rules.",
                "UNSUPPORTED_GENERATED_CLAIMS": "I cannot verify all factual parts of a reliable answer from the available evidence.",
            }
        else:
            messages = {
                "NO_RETRIEVAL_RESULTS": "未在知识库中找到足够证据，无法可靠回答该问题。",
                "LOW_RETRIEVAL_SCORE": "当前检索证据强度不足，无法可靠回答该问题。",
                "NO_VALID_CITATIONS": "现有证据无法形成有效引用，因此无法提供可靠答案。",
                "PERMISSION_DENIED": "相关文档不在当前用户权限范围内，无法提供答案。",
                "OUT_OF_SCOPE": "该问题超出当前知识库的范围。",
                "PROMPT_INJECTION_DETECTED": "检测到与知识库安全规则冲突的指令，无法按该指令继续。",
                "UNSUPPORTED_GENERATED_CLAIMS": "答案中存在无法由当前证据支持的事实，无法提供可靠答案。",
            }
        return messages.get(reason, messages["NO_VALID_CITATIONS"])


def answer_is_refusal(answer: str) -> bool:
    text = (answer or "").lower()
    return any(token in text for token in ("无法", "不能", "cannot", "not enough evidence", "outside the scope"))


def _append(items: list[str], value: str) -> list[str]:
    return _dedupe(items + [value])


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output

