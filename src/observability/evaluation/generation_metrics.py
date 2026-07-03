"""LLM-as-Judge generation metrics for MMDocRAG evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from src.libs.llm.base_llm import Message


@dataclass(frozen=True)
class JudgeMetricResult:
    """Structured result for an LLM-judged metric."""

    score: float | None
    reason: str = ""
    skipped: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result for JSON output."""

        return {
            "score": self.score,
            "reason": self.reason,
            "skipped": self.skipped,
            "error": self.error,
        }


def _clamp_score(value: Any) -> float:
    score = float(value)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("judge response does not contain a JSON object")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("judge response JSON is not an object")
    return parsed


def _normalise_judge_response(response: Any) -> JudgeMetricResult:
    if isinstance(response, JudgeMetricResult):
        return response

    if isinstance(response, (int, float)):
        return JudgeMetricResult(score=_clamp_score(response))

    if isinstance(response, dict):
        if "score" not in response:
            raise ValueError("judge response missing 'score'")
        return JudgeMetricResult(
            score=_clamp_score(response["score"]),
            reason=str(response.get("reason", "")),
        )

    content = getattr(response, "content", response)
    parsed = _extract_json_object(str(content))
    if "score" not in parsed:
        raise ValueError("judge response missing 'score'")

    return JudgeMetricResult(
        score=_clamp_score(parsed["score"]),
        reason=str(parsed.get("reason", "")),
    )


def _call_judge(
    judge: Any,
    system_prompt: str,
    payload: dict[str, Any],
) -> JudgeMetricResult:
    messages = [
        Message(role="system", content=system_prompt),
        Message(
            role="user",
            content=(
                "Evaluate the following MMDocRAG sample. Return only JSON with "
                'keys "score" and "reason".\n\n'
                + json.dumps(payload, ensure_ascii=False, indent=2)
            ),
        ),
    ]

    if hasattr(judge, "chat"):
        response = judge.chat(messages, temperature=0.0, max_tokens=512)
    elif callable(judge):
        response = judge(messages)
    else:
        raise TypeError("judge must be a BaseLLM-like object or callable")

    return _normalise_judge_response(response)


def answer_correctness(
    *,
    query: str,
    generated_answer: str | None,
    reference_answer: str | None,
    retrieved_contexts: Sequence[str] | None = None,
    judge: Any = None,
) -> JudgeMetricResult:
    """Judge semantic consistency between generated and reference answers."""

    if not reference_answer or not reference_answer.strip():
        return JudgeMetricResult(
            score=None,
            skipped=True,
            reason="missing_reference_answer",
        )
    if judge is None:
        return JudgeMetricResult(score=None, skipped=True, reason="judge_disabled")
    if not generated_answer or not generated_answer.strip():
        return JudgeMetricResult(score=0.0, reason="empty_generated_answer")

    system_prompt = (
        "You are a strict RAG answer evaluator. Score whether generated_answer "
        "is semantically correct against reference_answer for the query. "
        "Use 1 for fully correct, 0 for incorrect, partial credit allowed. "
        'Return JSON: {"score": 0.0, "reason": "brief explanation"}.'
    )
    payload = {
        "metric": "answer_correctness",
        "query": query,
        "generated_answer": generated_answer,
        "reference_answer": reference_answer,
        "retrieved_contexts": list(retrieved_contexts or []),
    }

    try:
        return _call_judge(judge, system_prompt, payload)
    except Exception as exc:
        return JudgeMetricResult(
            score=None,
            skipped=True,
            reason="judge_failed",
            error=str(exc),
        )


def faithfulness(
    *,
    query: str,
    generated_answer: str | None,
    retrieved_contexts: Sequence[str] | None,
    judge: Any = None,
) -> JudgeMetricResult:
    """Judge whether answer facts are supported by retrieved contexts."""

    if judge is None:
        return JudgeMetricResult(score=None, skipped=True, reason="judge_disabled")
    if not generated_answer or not generated_answer.strip():
        return JudgeMetricResult(score=0.0, reason="empty_generated_answer")

    contexts = [str(item) for item in retrieved_contexts or [] if str(item).strip()]
    if not contexts:
        return JudgeMetricResult(score=0.0, reason="missing_retrieved_contexts")

    system_prompt = (
        "You are a strict RAG faithfulness evaluator. Score whether every "
        "factual claim in generated_answer is supported by retrieved_contexts. "
        "Use 1 for fully supported, 0 for unsupported, partial credit allowed. "
        'Return JSON: {"score": 0.0, "reason": "brief explanation"}.'
    )
    payload = {
        "metric": "faithfulness",
        "query": query,
        "generated_answer": generated_answer,
        "retrieved_contexts": contexts,
    }

    try:
        return _call_judge(judge, system_prompt, payload)
    except Exception as exc:
        return JudgeMetricResult(
            score=None,
            skipped=True,
            reason="judge_failed",
            error=str(exc),
        )
