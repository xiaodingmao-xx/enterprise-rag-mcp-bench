"""Unit tests for MMDocRAG generation metrics."""

from __future__ import annotations

import pytest

from src.libs.llm.base_llm import ChatResponse
from src.observability.evaluation.generation_metrics import (
    answer_correctness,
    faithfulness,
)


class FakeJudge:
    """Simple BaseLLM-like fake returning a JSON score."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def chat(self, messages, **kwargs):  # noqa: ANN001, ANN002, ANN003
        self.calls += 1
        return ChatResponse(content=self.content, model="fake")


def test_answer_correctness_parses_json_score() -> None:
    judge = FakeJudge('{"score": 0.75, "reason": "mostly correct"}')

    result = answer_correctness(
        query="What is RAG?",
        generated_answer="RAG retrieves context before generation.",
        reference_answer="RAG retrieves relevant context before answering.",
        retrieved_contexts=["retrieval context"],
        judge=judge,
    )

    assert result.score == pytest.approx(0.75)
    assert result.reason == "mostly correct"
    assert result.skipped is False
    assert judge.calls == 1


def test_answer_correctness_skips_when_judge_disabled() -> None:
    result = answer_correctness(
        query="q",
        generated_answer="answer",
        reference_answer="reference",
        judge=None,
    )

    assert result.score is None
    assert result.skipped is True
    assert result.reason == "judge_disabled"


def test_empty_answer_scores_zero_when_judge_enabled() -> None:
    result = answer_correctness(
        query="q",
        generated_answer="",
        reference_answer="reference",
        judge=FakeJudge('{"score": 1, "reason": "unused"}'),
    )

    assert result.score == 0.0
    assert result.reason == "empty_generated_answer"


def test_judge_json_parse_failure_is_skipped_not_raised() -> None:
    result = answer_correctness(
        query="q",
        generated_answer="answer",
        reference_answer="reference",
        judge=FakeJudge("not json"),
    )

    assert result.score is None
    assert result.skipped is True
    assert result.reason == "judge_failed"
    assert result.error


def test_faithfulness_scores_zero_without_contexts() -> None:
    result = faithfulness(
        query="q",
        generated_answer="answer",
        retrieved_contexts=[],
        judge=FakeJudge('{"score": 1, "reason": "unused"}'),
    )

    assert result.score == 0.0
    assert result.reason == "missing_retrieved_contexts"


def test_faithfulness_parses_structured_json() -> None:
    result = faithfulness(
        query="q",
        generated_answer="answer",
        retrieved_contexts=["supporting context"],
        judge=FakeJudge('{"score": 1.2, "reason": "supported"}'),
    )

    assert result.score == 1.0
    assert result.reason == "supported"
