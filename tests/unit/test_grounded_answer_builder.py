"""Tests for grounded answer payload assembly."""

from src.core.response.grounded_answer_builder import GroundedAnswerBuilder
from src.core.response.retrieval_status import RetrievedContext, RetrievalStatus


def _context() -> RetrievedContext:
    return RetrievedContext(
        citation_id="C1",
        chunk_id="chunk-1",
        text="Evidence text",
        score=0.9,
        source="guide.pdf",
        page=2,
    )


def test_build_answer_payload_with_citations_and_sources() -> None:
    payload = GroundedAnswerBuilder().build(
        query="What is configured?",
        generated_answer="Use the endpoint [C1].",
        contexts=[_context()],
        retrieval_status=RetrievalStatus.SUFFICIENT,
        collection="docs",
        trace_id="trace-1",
        warnings=[],
    ).to_dict()

    assert payload["mode"] == "answer"
    assert payload["confidence"] == "medium"
    assert payload["citations"][0]["citation_id"] == "C1"
    assert payload["sources"][0]["chunk_id"] == "chunk-1"
    assert payload["used_chunk_ids"] == ["chunk-1"]
    assert payload["trace_id"] == "trace-1"


def test_build_answer_payload_can_omit_citations_and_sources() -> None:
    payload = GroundedAnswerBuilder().build(
        query="q",
        generated_answer="a",
        contexts=[_context()],
        retrieval_status="insufficient",
        include_sources=False,
        include_citations=False,
    ).to_dict()

    assert payload["confidence"] == "low"
    assert payload["citations"] == []
    assert payload["sources"] == []
