"""Tests for rule-based hallucination guard."""

from src.core.response.hallucination_guard import HallucinationGuard
from src.core.response.retrieval_status import RetrievedContext, RetrievalStatus


def _context(cid: str = "C1") -> RetrievedContext:
    return RetrievedContext(
        citation_id=cid,
        chunk_id="chunk-1",
        text="Evidence",
        score=0.8,
        source="doc.pdf",
    )


def test_invalid_citation_marker_warning() -> None:
    result = HallucinationGuard().validate(
        "Supported [C2]",
        [_context("C1")],
        RetrievalStatus.SUFFICIENT,
    )

    assert "INVALID_CITATION_MARKER:C2" in result.warnings


def test_missing_citation_marker_warning() -> None:
    result = HallucinationGuard().validate("Supported answer", [_context()], "sufficient")

    assert "MISSING_CITATION_MARKER" in result.warnings


def test_external_inference_warning() -> None:
    result = HallucinationGuard().validate(
        "Generally speaking, this is true [C1].",
        [_context()],
        "sufficient",
    )

    assert "POSSIBLE_EXTERNAL_INFERENCE" in result.warnings


def test_no_results_warning() -> None:
    result = HallucinationGuard().validate("No answer", [], "no_results")

    assert "NO_RETRIEVAL_RESULTS" in result.warnings
    assert "NO_CONTEXT_FOR_ANSWER" in result.warnings
