from src.core.response.answer_generator import GeneratedAnswer
from src.core.settings import load_settings
from src.core.types import RetrievalResult
from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool


class FakeAnswerGenerator:
    def __init__(self, answer: str):
        self.answer = answer

    def generate(self, **kwargs):
        return GeneratedAnswer(answer=self.answer)


def test_answer_mode_exposes_verification_and_confidence_metadata() -> None:
    tool = QueryKnowledgeHubTool(
        settings=load_settings(),
        answer_generator=FakeAnswerGenerator("功能已启用。[C1]"),
    )
    results = [
        RetrievalResult(
            chunk_id="chunk-1",
            score=0.9,
            text="功能已启用。",
            metadata={"source_path": "docs/a.md", "page": 1},
        )
    ]

    response = tool._build_answer_response(
        "功能状态是什么？",
        results,
        "default",
        "concise",
        "zh",
        True,
        True,
        None,
    )

    assert response.metadata["refused"] is False
    assert response.metadata["citation_verification"]
    assert response.metadata["confidence_score"] > 0
    assert response.metadata["confidence_factors"]
    assert response.metadata["citations"][0]["citation_id"] == "C1"


def test_answer_mode_refuses_empty_context_without_calling_llm() -> None:
    class ExplodingAnswerGenerator:
        def generate(self, **kwargs):
            raise AssertionError("generator must not be called without context")

    tool = QueryKnowledgeHubTool(settings=load_settings(), answer_generator=ExplodingAnswerGenerator())
    response = tool._build_answer_response(
        "What is the status?", [], "default", "concise", "en", True, True, None
    )

    assert response.metadata["refused"] is True
    assert response.metadata["refusal_reason"] == "NO_RETRIEVAL_RESULTS"
    assert "NO_RETRIEVAL_RESULTS" in response.metadata["warnings"]


def test_answer_generation_failure_is_structured() -> None:
    class FailingAnswerGenerator:
        def generate(self, **kwargs):
            raise RuntimeError("provider token should never be exposed")

    tool = QueryKnowledgeHubTool(settings=load_settings(), answer_generator=FailingAnswerGenerator())
    response = tool._build_answer_response(
        "功能状态是什么？",
        [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.9,
                text="功能已启用。",
                metadata={"source_path": "docs/a.md"},
            )
        ],
        "default",
        "concise",
        "zh",
        True,
        True,
        None,
    )

    assert "ANSWER_GENERATION_FAILED" in response.metadata["warnings"]
    assert response.metadata["fallback_reason"] == "answer_generation_failed"
    assert "provider token" not in response.content


def test_uncited_factual_answer_is_refused() -> None:
    tool = QueryKnowledgeHubTool(
        settings=load_settings(),
        answer_generator=FakeAnswerGenerator("功能已启用。"),
    )
    response = tool._build_answer_response(
        "功能状态是什么？",
        [RetrievalResult(chunk_id="chunk-1", score=0.9, text="功能已启用。", metadata={"source_path": "a.md"})],
        "default",
        "concise",
        "zh",
        True,
        True,
        None,
    )

    assert response.metadata["refused"] is True
    assert response.metadata["refusal_reason"] in {
        "NO_VALID_CITATIONS",
        "UNSUPPORTED_GENERATED_CLAIMS",
    }
    assert response.metadata["citation_coverage"] == 0.0


def test_conflicting_contexts_lower_confidence_and_warn() -> None:
    tool = QueryKnowledgeHubTool(
        settings=load_settings(),
        answer_generator=FakeAnswerGenerator("请人工确认来源差异。[C1][C2]"),
    )
    response = tool._build_answer_response(
        "功能状态是什么？",
        [
            RetrievalResult(chunk_id="a", score=0.9, text="功能已启用。", metadata={"source_path": "a.md"}),
            RetrievalResult(chunk_id="b", score=0.8, text="功能未启用。", metadata={"source_path": "b.md"}),
        ],
        "default",
        "concise",
        "zh",
        True,
        True,
        None,
    )

    assert "CONFLICTING_SOURCES" in response.metadata["warnings"]
    assert response.metadata["confidence_score"] < 0.75
