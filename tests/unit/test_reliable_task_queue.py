"""Offline tests for retry, lease and recovery semantics."""

from __future__ import annotations

from src.ingestion.errors import NonRetryableIngestionError, RetryableIngestionError
from src.ingestion.task_queue_backend import SQLiteTaskQueueBackend


def _backend(tmp_path):
    return SQLiteTaskQueueBackend(
        tmp_path / "tasks.db",
        {
            "max_retries": 3,
            "retry_base_delay_seconds": 1,
            "retry_max_delay_seconds": 1,
            "retry_jitter": False,
        },
    )


def test_retryable_failure_enters_retrying_with_backoff(tmp_path):
    backend = _backend(tmp_path)
    task_id = backend.enqueue("ingest", {"path": "doc.txt"})
    backend.claim_next("worker-1", lease_seconds=30, now="2026-01-01T00:00:00+00:00")
    task = backend.mark_failed(task_id, RetryableIngestionError("temporary"), now="2026-01-01T00:00:00+00:00")
    assert task["status"] == "retrying"
    assert task["retry_count"] == 1
    assert task["next_retry_at"] == "2026-01-01T00:00:01+00:00"


def test_non_retryable_failure_is_failed_without_retry(tmp_path):
    backend = _backend(tmp_path)
    task_id = backend.enqueue("ingest")
    backend.claim_next("worker-1", now="2026-01-01T00:00:00+00:00")
    task = backend.mark_failed(task_id, NonRetryableIngestionError("bad file", code="CORRUPTED_FILE"), now="2026-01-01T00:00:00+00:00")
    assert task["status"] == "failed"
    assert task["retry_count"] == 0
    assert task["last_error_code"] == "CORRUPTED_FILE"


def test_max_retries_moves_to_dead_letter(tmp_path):
    backend = _backend(tmp_path)
    task_id = backend.enqueue("ingest")
    for second in range(3):
        task = backend.claim_next("worker-1", now=f"2026-01-01T00:00:0{second}+00:00")
        assert task is not None
        task = backend.mark_failed(task_id, RetryableIngestionError("temporary"), now=f"2026-01-01T00:00:0{second}+00:00")
        if second < 2:
            backend.recover_stale_tasks(f"2026-01-01T00:00:0{second + 1}+00:00")
    assert task["status"] == "dead_letter"
    assert task["retry_count"] == 3


def test_expired_lease_can_be_recovered_without_incrementing_retry(tmp_path):
    backend = _backend(tmp_path)
    task_id = backend.enqueue("ingest")
    backend.claim_next("worker-1", lease_seconds=10, now="2026-01-01T00:00:00+00:00")
    recovered = backend.recover_stale_tasks("2026-01-01T00:00:11+00:00")
    assert recovered[0]["task_id"] == task_id
    assert recovered[0]["status"] == "queued"
    assert recovered[0]["retry_count"] == 0


def test_dead_letter_can_be_manually_requeued(tmp_path):
    backend = _backend(tmp_path)
    task_id = backend.enqueue("ingest")
    backend.claim_next("worker-1")
    backend.move_to_dead_letter(task_id, "UNKNOWN_FATAL_ERROR", "manual diagnosis")
    assert backend.get_task(task_id)["status"] == "dead_letter"
    assert backend.requeue_dead_letter(task_id, clear_error=True)
    task = backend.get_task(task_id)
    assert task["status"] == "queued"
    assert task["last_error_code"] is None
