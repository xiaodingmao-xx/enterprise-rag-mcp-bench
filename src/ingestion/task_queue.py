"""Background ingestion task queue for concurrent document uploads."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import closing
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.core.settings import Settings, load_settings, resolve_path
from src.core.trace import TraceCollector, TraceContext
from src.ingestion.pipeline import IngestionPipeline, PipelineResult

JobRunner = Callable[
    [str, str, bool, TraceContext, Callable[[str, int, int], None]],
    PipelineResult,
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestionJobStore:
    """SQLite-backed job state store for background ingestion."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = resolve_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def create_job(
        self,
        *,
        file_path: str,
        collection: str,
        force: bool = False,
        original_name: Optional[str] = None,
    ) -> str:
        job_id = uuid.uuid4().hex
        now = _utc_now()
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO ingestion_jobs (
                    job_id, file_path, original_name, collection, force,
                    status, progress_stage, progress_current, progress_total,
                    trace_id, error, result_json, created_at, started_at,
                    finished_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'queued', NULL, 0, 0, NULL, NULL, NULL,
                        ?, NULL, NULL, ?)
                """,
                (
                    job_id,
                    str(file_path),
                    original_name,
                    collection,
                    1 if force else 0,
                    now,
                    now,
                ),
            )
            conn.commit()
        return job_id

    def mark_running(self, job_id: str, trace_id: str) -> None:
        now = _utc_now()
        self.update_job(
            job_id,
            status="running",
            trace_id=trace_id,
            started_at=now,
            updated_at=now,
        )

    def update_progress(
        self,
        job_id: str,
        stage: str,
        current: int,
        total: int,
    ) -> None:
        self.update_job(
            job_id,
            progress_stage=stage,
            progress_current=int(current),
            progress_total=int(total),
            updated_at=_utc_now(),
        )

    def mark_finished(
        self,
        job_id: str,
        *,
        status: str,
        result: Optional[PipelineResult] = None,
        error: Optional[str] = None,
    ) -> None:
        now = _utc_now()
        result_json = json.dumps(result.to_dict(), ensure_ascii=False) if result else None
        self.update_job(
            job_id,
            status=status,
            error=error,
            result_json=result_json,
            finished_at=now,
            updated_at=now,
        )

    def mark_failed(self, job_id: str, error: str) -> None:
        self.mark_finished(job_id, status="failed", error=error)

    def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [job_id]
        with closing(self._connect()) as conn:
            conn.execute(
                f"UPDATE ingestion_jobs SET {assignments} WHERE job_id = ?",
                values,
            )
            conn.commit()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def list_jobs(
        self,
        *,
        collection: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM ingestion_jobs"
        params: List[Any] = []
        if collection:
            sql += " WHERE collection = ?"
            params.append(collection)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    original_name TEXT,
                    collection TEXT NOT NULL,
                    force INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    progress_stage TEXT,
                    progress_current INTEGER NOT NULL DEFAULT 0,
                    progress_total INTEGER NOT NULL DEFAULT 0,
                    trace_id TEXT,
                    error TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status
                ON ingestion_jobs(status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_collection
                ON ingestion_jobs(collection)
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["force"] = bool(data.get("force"))
        result_json = data.get("result_json")
        if result_json:
            try:
                data["result"] = json.loads(result_json)
            except json.JSONDecodeError:
                data["result"] = None
        else:
            data["result"] = None
        return data


class IngestionTaskQueue:
    """Thread-pool backed ingestion queue."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        store: Optional[IngestionJobStore] = None,
        max_workers: Optional[int] = None,
        runner: Optional[JobRunner] = None,
        trace_collector: Optional[Any] = None,
    ) -> None:
        self.settings = settings or load_settings()
        config = self._config
        configured_workers = int(config.get("max_workers", 2) or 2)
        enabled = bool(config.get("enabled", True))
        requested_workers = int(max_workers or configured_workers)
        self.max_workers = max(1, requested_workers if enabled else 1)
        self.store = store or IngestionJobStore(
            config.get("job_db_path", "./data/db/ingestion_jobs.db")
        )
        self._runner = runner
        self._trace_collector = trace_collector or TraceCollector()
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="ingestion-worker",
        )
        self._futures: Dict[str, Future[Any]] = {}
        self._futures_lock = threading.Lock()

    @property
    def _config(self) -> Dict[str, Any]:
        ingestion = getattr(self.settings, "ingestion", None)
        config = getattr(ingestion, "concurrent_upload", {}) if ingestion else {}
        return config if isinstance(config, dict) else {}

    @property
    def upload_dir(self) -> Path:
        return resolve_path(self._config.get("upload_dir", "./data/uploads/ingestion"))

    def submit_file(
        self,
        file_path: str | Path,
        *,
        collection: str = "default",
        force: bool = False,
        original_name: Optional[str] = None,
    ) -> str:
        job_id = self.store.create_job(
            file_path=str(file_path),
            collection=collection,
            force=force,
            original_name=original_name,
        )
        future = self._executor.submit(
            self._run_job,
            job_id,
            str(file_path),
            collection,
            force,
            original_name,
        )
        with self._futures_lock:
            self._futures[job_id] = future
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get_job(job_id)

    def list_jobs(
        self,
        *,
        collection: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return self.store.list_jobs(collection=collection, limit=limit)

    def wait(self, job_id: str, timeout: Optional[float] = None) -> Optional[Any]:
        with self._futures_lock:
            future = self._futures.get(job_id)
        if future is None:
            return None
        return future.result(timeout=timeout)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def _run_job(
        self,
        job_id: str,
        file_path: str,
        collection: str,
        force: bool,
        original_name: Optional[str],
    ) -> None:
        trace = TraceContext(trace_type="ingestion")
        trace.metadata["job_id"] = job_id
        trace.metadata["source_path"] = original_name or file_path
        trace.metadata["stored_path"] = file_path
        trace.metadata["collection"] = collection
        trace.metadata["source"] = "concurrent_upload_queue"
        self.store.mark_running(job_id, trace.trace_id)

        def on_progress(stage: str, current: int, total: int) -> None:
            self.store.update_progress(job_id, stage, current, total)

        try:
            runner = self._runner or self._run_pipeline
            result = runner(file_path, collection, force, trace, on_progress)
            self._trace_collector.collect(trace)
            if result.success:
                skipped = result.stages.get("integrity", {}).get("skipped", False)
                self.store.mark_finished(
                    job_id,
                    status="skipped" if skipped else "succeeded",
                    result=result,
                )
            else:
                self.store.mark_finished(
                    job_id,
                    status="failed",
                    result=result,
                    error=result.error,
                )
        except Exception as exc:
            self._trace_collector.collect(trace)
            self.store.mark_failed(job_id, str(exc))

    def _run_pipeline(
        self,
        file_path: str,
        collection: str,
        force: bool,
        trace: TraceContext,
        on_progress: Callable[[str, int, int], None],
    ) -> PipelineResult:
        pipeline = IngestionPipeline(
            self.settings,
            collection=collection,
            force=force,
        )
        try:
            return pipeline.run(file_path, trace=trace, on_progress=on_progress)
        finally:
            close = getattr(pipeline, "close", None)
            if callable(close):
                close()


_default_queue: Optional[IngestionTaskQueue] = None
_default_queue_lock = threading.Lock()


def get_default_ingestion_queue(
    settings: Optional[Settings] = None,
) -> IngestionTaskQueue:
    """Return a process-local default ingestion queue."""
    global _default_queue
    with _default_queue_lock:
        if _default_queue is None:
            _default_queue = IngestionTaskQueue(settings=settings)
        return _default_queue


# The durable backend is exported from the historical queue module so callers
# can migrate incrementally without changing their import path.
from src.ingestion.task_queue_backend import SQLiteTaskQueueBackend  # noqa: E402

TaskQueueBackend = SQLiteTaskQueueBackend


def get_task_queue_backend(settings: Optional[Settings] = None) -> SQLiteTaskQueueBackend:
    """Build the configured local durable queue backend."""
    settings = settings or load_settings()
    ingestion = getattr(settings, "ingestion", None)
    config = getattr(ingestion, "task_queue", {}) if ingestion is not None else {}
    if not isinstance(config, dict):
        config = {}
    backend = str(config.get("backend", "sqlite")).lower()
    if backend != "sqlite":
        raise ValueError(f"Unsupported task queue backend: {backend}; SQLite is the current default")
    return SQLiteTaskQueueBackend(config.get("db_path", "./data/db/ingestion_tasks.db"), config)
