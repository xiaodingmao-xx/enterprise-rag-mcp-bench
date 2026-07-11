from src.core.response.refusal_policy import RefusalPolicy
from src.core.response.retrieval_status import RetrievalStatus


def test_refuses_without_context_in_english() -> None:
    decision = RefusalPolicy().decide(
        query="What is the release status?",
        results=[],
        retrieval_status=RetrievalStatus.NO_RESULTS,
    )

    assert decision.should_refuse is True
    assert decision.reason == "NO_RETRIEVAL_RESULTS"
    assert decision.message and decision.message.startswith("I could not")


def test_refuses_without_context_in_chinese() -> None:
    decision = RefusalPolicy().decide(query="发布状态是什么？", results=[])

    assert decision.should_refuse is True
    assert "知识库" in (decision.message or "")

