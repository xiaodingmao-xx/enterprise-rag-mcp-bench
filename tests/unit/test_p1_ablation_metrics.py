from scripts.run_ablation_eval import _aggregate_query_metrics, _percentile
import pytest


def test_ablation_metrics_include_latency_candidates_and_cost():
    metrics = _aggregate_query_metrics(
        [
            {"skipped": False, "metrics": {"recall@2": 1.0}, "latency_ms": 10, "candidate_count": 4, "rerank_latency_ms": 2, "cost_estimate": 0.01},
            {"skipped": False, "metrics": {"recall@2": 0.0}, "latency_ms": 30, "candidate_count": 8, "rerank_latency_ms": 4, "cost_estimate": 0.02},
            {"skipped": True, "metrics": {}, "latency_ms": 100, "candidate_count": 100, "cost_estimate": 1.0},
        ]
    )
    assert metrics["recall@2"] == 0.5
    assert metrics["latency_p50_ms"] == 20
    assert metrics["latency_p95_ms"] == pytest.approx(29)
    assert metrics["latency_p99_ms"] == pytest.approx(29.8)
    assert metrics["candidate_count"] == 6
    assert metrics["cost_estimate"] == 0.03


def test_percentile_empty_input_is_safe():
    assert _percentile([], 95) == 0.0
