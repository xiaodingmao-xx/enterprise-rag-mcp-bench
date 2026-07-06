"""Tests for concurrent ingestion task queue."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from src.core.trace import TraceContext
from src.ingestion.pipeline import PipelineResult
from src.ingestion.storage_locks import collection_storage_lock
from src.ingestion.task_queue import IngestionJobStore, IngestionTaskQueue


class FakeTraceCollector:
    def __init__(self) -> None:
        self.traces = []

    def collect(self, trace: TraceContext) -> None:
        trace.finish()
        self.traces.append(trace)


def _settings(tmp_path, **overrides):
    concurrent_upload = {
        "max_workers": 2,
        "job_db_path": str(tmp_path / "jobs.db"),
        "upload_dir": str(tmp_path / "uploads"),
    }
    concurrent_upload.update(overrides)
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            concurrent_upload=concurrent_upload
        )
    )


def test_job_store_lifecycle(tmp_path) -> None:
    store = IngestionJobStore(tmp_path / "jobs.db")
    job_id = store.create_job(
        file_path="doc.txt",
        collection="docs",
        force=True,
        original_name="doc.txt",
    )

    store.mark_running(job_id, trace_id="trace-1")
    store.update_progress(job_id, "embed", 6, 7)
    store.mark_finished(
        job_id,
        status="succeeded",
        result=PipelineResult(success=True, file_path="doc.txt", chunk_count=2),
    )

    job = store.get_job(job_id)

    assert job is not None
    assert job["status"] == "succeeded"
    assert job["force"] is True
    assert job["progress_stage"] == "embed"
    assert job["progress_current"] == 6
    assert job["result"]["chunk_count"] == 2


def test_task_queue_runs_jobs_concurrently(tmp_path) -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def runner(file_path, collection, force, trace: TraceContext, on_progress):
        nonlocal active, max_active
        on_progress("load", 3, 7)
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.1)
        with lock:
            active -= 1
        return PipelineResult(success=True, file_path=file_path, chunk_count=1)

    queue = IngestionTaskQueue(
        settings=_settings(tmp_path),
        store=IngestionJobStore(tmp_path / "jobs.db"),
        max_workers=2,
        runner=runner,
        trace_collector=FakeTraceCollector(),
    )
    try:
        job_ids = [
            queue.submit_file(tmp_path / f"doc_{i}.txt", collection="docs")
            for i in range(4)
        ]
        for job_id in job_ids:
            queue.wait(job_id, timeout=5)

        jobs = [queue.get_job(job_id) for job_id in job_ids]
    finally:
        queue.shutdown()

    assert max_active >= 2
    assert all(job is not None and job["status"] == "succeeded" for job in jobs)
    assert all(job["progress_stage"] == "load" for job in jobs)


def test_task_queue_marks_failed_jobs(tmp_path) -> None:
    def runner(file_path, collection, force, trace, on_progress):
        raise RuntimeError("boom")

    queue = IngestionTaskQueue(
        settings=_settings(tmp_path),
        store=IngestionJobStore(tmp_path / "jobs.db"),
        max_workers=1,
        runner=runner,
        trace_collector=FakeTraceCollector(),
    )
    try:
        job_id = queue.submit_file(tmp_path / "bad.txt", collection="docs")
        queue.wait(job_id, timeout=5)
        job = queue.get_job(job_id)
    finally:
        queue.shutdown()

    assert job is not None
    assert job["status"] == "failed"
    assert "boom" in job["error"]


def test_task_queue_disabled_config_uses_single_worker(tmp_path) -> None:
    queue = IngestionTaskQueue(
        settings=_settings(tmp_path, enabled=False, max_workers=4),
        store=IngestionJobStore(tmp_path / "jobs.db"),
        runner=lambda *args: PipelineResult(success=True, file_path=args[0]),
        trace_collector=FakeTraceCollector(),
    )
    try:
        assert queue.max_workers == 1
    finally:
        queue.shutdown()


def test_collection_storage_lock_serializes_writes() -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def critical_section() -> None:
        nonlocal active, max_active
        with collection_storage_lock("same_collection"):
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1

    threads = [threading.Thread(target=critical_section) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max_active == 1
