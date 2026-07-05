"""Tests for already-processed ingestion trace handling."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.core.trace.trace_context import TraceContext
from src.ingestion.pipeline import IngestionPipeline
from src.observability.dashboard.pages.ingestion_traces import (
    _is_already_processed_trace,
)


def test_pipeline_records_already_processed_trace() -> None:
    class FakePipeline:
        collection = "default"
        force = False

    pipeline = FakePipeline()
    pipeline.integrity_checker = MagicMock()
    pipeline.integrity_checker.compute_sha256.return_value = "hash123"
    pipeline.integrity_checker.should_skip.return_value = True

    trace = TraceContext(trace_type="ingestion")
    result = IngestionPipeline.run(pipeline, "already.pdf", trace=trace)

    assert result.success is True
    assert result.stages["integrity"]["skipped"] is True
    assert result.stages["integrity"]["reason"] == "already_processed"
    assert trace.metadata["ingestion_status"] == "skipped"
    assert trace.metadata["skip_reason"] == "already_processed"
    assert trace.stages[0]["stage"] == "integrity"
    assert trace.stages[0]["data"]["skipped"] is True


def test_dashboard_detects_already_processed_trace() -> None:
    trace = {"metadata": {"skip_reason": "already_processed"}}
    stages_by_name = {
        "integrity": {
            "data": {
                "skipped": True,
                "reason": "already_processed",
            }
        }
    }

    assert _is_already_processed_trace(trace, stages_by_name) is True


def test_dashboard_detects_already_processed_from_integrity_stage() -> None:
    trace = {"metadata": {}}
    stages_by_name = {
        "integrity": {
            "data": {
                "skipped": True,
                "reason": "already_processed",
            }
        }
    }

    assert _is_already_processed_trace(trace, stages_by_name) is True
