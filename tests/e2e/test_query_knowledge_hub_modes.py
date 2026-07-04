"""E2E-style tests for query_knowledge_hub contexts/answer modes."""

import asyncio
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

if importlib.util.find_spec("mcp") is None:
    fake_mcp = ModuleType("mcp")

    class TextContent:
        def __init__(self, type: str, text: str) -> None:
            self.type = type
            self.text = text

    class ImageContent:
        pass

    class CallToolResult:
        def __init__(self, content, isError: bool = False) -> None:
            self.content = content
            self.isError = isError

    fake_mcp.types = SimpleNamespace(
        TextContent=TextContent,
        ImageContent=ImageContent,
        CallToolResult=CallToolResult,
        Tool=object,
    )
    sys.modules["mcp"] = fake_mcp

from src.core.response.answer_generator import GeneratedAnswer
from src.core.settings import (
    AnswerGenerationSettings,
    HallucinationGuardSettings,
    PerformanceSettings,
    QueryCacheSettings,
    ResponseSettings,
)
from src.core.types import RetrievalResult
from src.mcp_server.tools import query_knowledge_hub as tool_module
from src.mcp_server.tools.query_knowledge_hub import (
    TOOL_INPUT_SCHEMA,
    QueryKnowledgeHubConfig,
    QueryKnowledgeHubTool,
)


class DummyTraceCollector:
    def collect(self, trace) -> None:
        pass


class FakeAnswerGenerator:
    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, query, contexts, answer_style="concise", language="auto", trace=None):
        self.call_count += 1
        return GeneratedAnswer(answer="The endpoint is configured in settings.yaml [C1].", llm_latency_ms=12.0)


class FailingAnswerGenerator:
    def generate(self, query, contexts, answer_style="concise", language="auto", trace=None):
        raise RuntimeError("llm down")


def _settings(min_score: float = 0.2):
    return SimpleNamespace(
        embedding=SimpleNamespace(provider="fake", model="fake-embedding"),
        rerank=SimpleNamespace(enabled=False, provider="none", model="none", top_k=5),
        performance=PerformanceSettings(
            query_cache=QueryCacheSettings(enabled=False),
        ),
        response=ResponseSettings(
            answer_generation=AnswerGenerationSettings(
                enabled=True,
                default_mode="contexts",
                min_contexts=1,
                min_score=min_score,
                max_context_chars=2000,
                hallucination_guard=HallucinationGuardSettings(enabled=True),
            )
        ),
    )


def _result(score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-1",
        score=score,
        text="The endpoint is configured in settings.yaml.",
        metadata={"source_path": "settings.yaml", "page": 1},
    )


def _tool(monkeypatch, results, answer_generator=None, min_score: float = 0.2):
    tool = QueryKnowledgeHubTool(
        settings=_settings(min_score=min_score),
        config=QueryKnowledgeHubConfig(enable_rerank=False),
        answer_generator=answer_generator or FakeAnswerGenerator(),
    )

    monkeypatch.setattr(tool, "_ensure_initialized", lambda collection: None)
    monkeypatch.setattr(tool, "_perform_search", lambda query, top_k, trace=None: list(results))
    monkeypatch.setattr(tool_module, "TraceCollector", DummyTraceCollector)
    return tool


def test_schema_contains_mode_answer_style_and_language() -> None:
    props = TOOL_INPUT_SCHEMA["properties"]

    assert props["mode"]["enum"] == ["contexts", "answer"]
    assert "answer_style" in props
    assert "language" in props
    assert "include_sources" in props
    assert "include_citations" in props


def test_default_mode_is_contexts(monkeypatch) -> None:
    tool = _tool(monkeypatch, [_result()])

    response = asyncio.run(tool.execute("How configure endpoint?", top_k=1, collection="docs"))

    assert response.metadata["mode"] == "contexts"
    assert response.metadata["retrieval_status"] == "sufficient"
    assert response.metadata["results"][0]["chunk_id"] == "chunk-1"


def test_contexts_mode_keeps_context_response(monkeypatch) -> None:
    tool = _tool(monkeypatch, [_result()])

    response = asyncio.run(tool.execute("How configure endpoint?", mode="contexts", top_k=1))

    assert response.metadata["mode"] == "contexts"
    assert response.metadata["result_count"] == 1


def test_answer_mode_returns_grounded_answer(monkeypatch) -> None:
    generator = FakeAnswerGenerator()
    tool = _tool(monkeypatch, [_result()], answer_generator=generator)

    response = asyncio.run(tool.execute("How configure endpoint?", mode="answer", top_k=1))

    assert response.metadata["mode"] == "answer"
    assert response.metadata["answer"].startswith("The endpoint")
    assert response.metadata["citations"][0]["citation_id"] == "C1"
    assert response.metadata["used_chunk_ids"] == ["chunk-1"]
    assert generator.call_count == 1


def test_answer_mode_no_results_returns_refusal(monkeypatch) -> None:
    tool = _tool(monkeypatch, [])

    response = asyncio.run(tool.execute("Missing info?", mode="answer", language="en"))

    assert response.metadata["retrieval_status"] == "no_results"
    assert response.metadata["confidence"] == "low"
    assert response.metadata["citations"] == []
    assert "NO_RETRIEVAL_RESULTS" in response.metadata["warnings"]


def test_answer_mode_low_score_returns_insufficient(monkeypatch) -> None:
    generator = FakeAnswerGenerator()
    tool = _tool(monkeypatch, [_result(score=0.05)], answer_generator=generator, min_score=0.2)

    response = asyncio.run(tool.execute("Maybe relevant?", mode="answer", language="en"))

    assert response.metadata["retrieval_status"] == "insufficient"
    assert response.metadata["confidence"] == "low"
    assert "INSUFFICIENT_RETRIEVAL_RESULTS" in response.metadata["warnings"]
    assert generator.call_count == 0


def test_answer_mode_llm_failure_returns_controlled_fallback(monkeypatch) -> None:
    tool = _tool(monkeypatch, [_result()], answer_generator=FailingAnswerGenerator())

    response = asyncio.run(tool.execute("How configure endpoint?", mode="answer", language="en"))

    assert response.metadata["retrieval_status"] == "sufficient"
    assert response.metadata["fallback_reason"] == "answer_generation_failed"
    assert "ANSWER_GENERATION_FAILED" in response.metadata["warnings"]
