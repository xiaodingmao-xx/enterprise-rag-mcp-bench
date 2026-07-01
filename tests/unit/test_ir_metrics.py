"""Unit tests for ranking IR metrics."""

from __future__ import annotations

import math

import pytest

from src.observability.evaluation.ir_metrics import (
    evaluate_ranking_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_precision_mrr_ndcg_for_perfect_ranking() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}

    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1.0)
    assert precision_at_k(retrieved, relevant, 2) == pytest.approx(1.0)
    assert mrr_at_k(retrieved, relevant, 2) == pytest.approx(1.0)
    assert ndcg_at_k(retrieved, relevant, 2) == pytest.approx(1.0)


def test_partial_match_metrics() -> None:
    retrieved = ["x", "b", "y"]
    relevant = {"a", "b"}

    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(0.5)
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)
    assert mrr_at_k(retrieved, relevant, 3) == pytest.approx(0.5)

    expected_ndcg = (1 / math.log2(3)) / (
        1 / math.log2(2) + 1 / math.log2(3)
    )
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(expected_ndcg)


def test_no_match_returns_zero() -> None:
    retrieved = ["x", "y"]
    relevant = {"a", "b"}

    assert recall_at_k(retrieved, relevant, 2) == 0.0
    assert precision_at_k(retrieved, relevant, 2) == 0.0
    assert mrr_at_k(retrieved, relevant, 2) == 0.0
    assert ndcg_at_k(retrieved, relevant, 2) == 0.0


def test_empty_relevant_or_non_positive_k_returns_zero() -> None:
    assert recall_at_k(["a"], [], 1) == 0.0
    assert precision_at_k(["a"], [], 1) == 0.0
    assert mrr_at_k(["a"], [], 1) == 0.0
    assert ndcg_at_k(["a"], [], 1) == 0.0

    assert recall_at_k(["a"], ["a"], 0) == 0.0
    assert precision_at_k(["a"], ["a"], 0) == 0.0
    assert mrr_at_k(["a"], ["a"], 0) == 0.0
    assert ndcg_at_k(["a"], ["a"], 0) == 0.0


def test_duplicate_retrieved_ids_do_not_double_count_hits() -> None:
    retrieved = ["a", "a", "b"]
    relevant = {"a", "b"}

    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(1.0)
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(0.5)
    assert precision_at_k(retrieved, relevant, 2) == pytest.approx(0.5)


def test_bundle_uses_metric_names_with_k() -> None:
    metrics = evaluate_ranking_at_k(["x", "b"], {"b"}, 2)

    assert metrics == {
        "recall@2": pytest.approx(1.0),
        "precision@2": pytest.approx(0.5),
        "mrr@2": pytest.approx(0.5),
        "ndcg@2": pytest.approx(1 / math.log2(3)),
    }


def test_non_integer_k_raises_type_error() -> None:
    with pytest.raises(TypeError, match="k must be an integer"):
        recall_at_k(["a"], ["a"], 1.5)  # type: ignore[arg-type]
