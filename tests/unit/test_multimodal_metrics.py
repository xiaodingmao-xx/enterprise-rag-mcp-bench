"""Unit tests for MMDocRAG multimodal metrics."""

from __future__ import annotations

import pytest

from src.core.types import RetrievalResult
from src.observability.evaluation.multimodal_metrics import (
    evaluate_multimodal_at_k,
    image_hit_at_k,
    modality_recall_at_k,
    table_hit_at_k,
)


def _result(chunk_id: str, metadata: dict) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=1.0,
        text="content",
        metadata={"source_path": "doc.pdf", **metadata},
    )


def test_modality_recall_counts_expected_modalities() -> None:
    results = [
        _result("c1", {"modality": "text"}),
        _result("c2", {"has_image": True}),
    ]

    score = modality_recall_at_k(results, ["text", "image", "table"], 2)

    assert score == pytest.approx(2 / 3)


def test_image_and_table_hit_use_metadata_fallback_fields() -> None:
    results = [
        _result("c1", {"content_type": "paragraph"}),
        _result("c2", {"image_id": "img-1"}),
        _result("c3", {"table_id": "tbl-1"}),
    ]

    assert image_hit_at_k(results, ["image"], 2) == 1.0
    assert table_hit_at_k(results, ["table"], 2) == 0.0
    assert table_hit_at_k(results, ["table"], 3) == 1.0


def test_string_false_flags_do_not_count_as_modalities() -> None:
    results = [_result("c1", {"has_image": "false", "has_table": "0"})]

    assert image_hit_at_k(results, ["image"], 1) == 0.0
    assert table_hit_at_k(results, ["table"], 1) == 0.0


def test_multimodal_metrics_skip_when_expected_modalities_missing() -> None:
    results = [_result("c1", {"has_image": True})]

    assert modality_recall_at_k(results, [], 1) is None
    assert image_hit_at_k(results, [], 1) is None
    assert table_hit_at_k(results, None, 1) is None
    assert evaluate_multimodal_at_k(results, [], 1) == {}


def test_metric_bundle_uses_snake_case_names_with_k() -> None:
    results = [
        _result("c1", {"modality": "text"}),
        _result("c2", {"content_type": "table"}),
    ]

    metrics = evaluate_multimodal_at_k(results, ["text", "table"], 2)

    assert metrics == {
        "modality_recall@2": pytest.approx(1.0),
        "table_hit@2": pytest.approx(1.0),
    }
