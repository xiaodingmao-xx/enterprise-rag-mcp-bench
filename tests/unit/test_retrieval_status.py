"""Tests for retrieval status and context normalization."""

from src.core.response.retrieval_status import (
    RetrievalStatus,
    assess_retrieval_status,
    contexts_from_results,
)
from src.core.types import RetrievalResult


def _result(chunk_id: str, score: float, text: str = "text") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        metadata={"source_path": "doc.pdf", "page": "3"},
    )


def test_no_results_status() -> None:
    assert assess_retrieval_status([]) == RetrievalStatus.NO_RESULTS


def test_low_score_is_insufficient() -> None:
    assert assess_retrieval_status([_result("c1", 0.1)], min_score=0.2) == RetrievalStatus.INSUFFICIENT


def test_enough_contexts_are_sufficient() -> None:
    assert assess_retrieval_status([_result("c1", 0.7)], min_contexts=1) == RetrievalStatus.SUFFICIENT


def test_contexts_from_results_maps_citation_source_and_page() -> None:
    contexts = contexts_from_results([_result("c1", 0.7, "hello world")])

    assert contexts[0].citation_id == "C1"
    assert contexts[0].chunk_id == "c1"
    assert contexts[0].source == "doc.pdf"
    assert contexts[0].page == 3
    assert contexts[0].to_citation_dict()["citation_id"] == "C1"
