"""Unit tests for MMDocRAG citation accuracy."""

from __future__ import annotations

import pytest

from src.core.types import RetrievalResult
from src.observability.evaluation.citation_metrics import (
    answer_citation_labels,
    citation_accuracy,
    expected_citation_labels,
)


def _result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-1",
        score=0.9,
        text="supporting text",
        metadata={
            "source_path": "docs/report.pdf",
            "page": 2,
            "image_id": "img-1",
        },
    )


def test_expected_citation_labels_merge_optional_schema_fields() -> None:
    labels = expected_citation_labels(
        expected_sources=["report.pdf"],
        expected_pages=[2],
        expected_chunk_ids=["chunk-1"],
        expected_evidence=[{"image_id": "img-1", "table_id": "tbl-1"}],
    )

    assert "source:report.pdf" in labels
    assert "page:2" in labels
    assert "chunk_id:chunk-1" in labels
    assert "image_id:img-1" in labels
    assert "table_id:tbl-1" in labels


def test_answer_bracket_citation_maps_to_retrieved_metadata() -> None:
    labels = answer_citation_labels("答案来自 [1]。", [_result()])

    assert "chunk_id:chunk-1" in labels
    assert "source:report.pdf" in labels
    assert "page:2" in labels
    assert "image_id:img-1" in labels


def test_citation_accuracy_scores_full_match() -> None:
    score = citation_accuracy(
        generated_answer="根据 [1]，答案来自报告第 2 页。",
        retrieved_results=[_result()],
        expected_sources=["report.pdf"],
        expected_pages=[2],
        expected_chunk_ids=["chunk-1"],
    )

    assert score == pytest.approx(1.0)


def test_citation_accuracy_scores_zero_when_answer_has_no_citation() -> None:
    score = citation_accuracy(
        generated_answer="没有引用的答案。",
        retrieved_results=[_result()],
        expected_sources=["report.pdf"],
    )

    assert score == 0.0


def test_citation_accuracy_skips_without_expected_annotations() -> None:
    assert citation_accuracy(generated_answer="answer", retrieved_results=[_result()]) is None


def test_citation_accuracy_handles_missing_fields_gracefully() -> None:
    score = citation_accuracy(
        generated_answer="答案来自 [1]。",
        retrieved_results=[{"metadata": {}}],
        expected_sources=["missing.pdf"],
    )

    assert score == 0.0
