"""SQLite task queue backend with leases, retries and dead-letter handling."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from src.core.settings import resolve_path
from src.ingestion.errors import classify_error
from src.ingestion.retry_policy import compute_backoff_delay


TASK_STATUSES = {"queued", "running", "retrying", "succeeded", "failed", "dead_letter", "cancelled"}


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteTaskQueueBackend:
    """Durable local task queue; Redis/Celery can implement the same methods later."""

    def __init__(self, db_path: str | Path = "./data/db/ingestion_tasks.db", config: Optional[dict[str, Any]] = None) -> None:
        self.db_path = resolve_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS ingestion_tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','retrying','succeeded','failed','dead_letter','cancelled')),
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 3,
                    next_retry_at TEXT,
                    worker_id TEXT,
                    lease_until TEXT,
                    heartbeat_at TEXT,
                    last_error_code TEXT,
                    last_error_message TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status_next_retry_at ON ingestion_tasks(status, next_retry_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_lease_until ON ingestion_tasks(lease_until)")
            conn.commit()

    @staticmethod
    def _row(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        if row is None:
            return None
        data = dict(row)
        try:
            data["payload"] = json.loads(data.pop("payload_json"))
        except (TypeError, ValueError):
            data["payload"] = {}
        return data

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        with closing(self._connect()) as conn:
            return self._row(conn.execute("SELECT * FROM ingestion_tasks WHERE task_id=?", (task_id,)).fetchone())

    def enqueue(self, task_type: str, payload: Optional[dict[str, Any]] = None, *, max_retries: Optional[int] = None, task_id: Optional[str] = None) -> str:
        task_id = task_id or uuid.uuid4().hex
        now = _now()
        retries = int(max_retries if max_retries is not None else self.config.get("max_retries", 3))
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO ingestion_tasks
                   (task_id, task_type, payload_json, status, max_retries, created_at, updated_at)
                   VALUES (?, ?, ?, 'queued', ?, ?, ?)""",
                (task_id, task_type, json.dumps(payload or {}, ensure_ascii=False), max(0, retries), now, now),
            )
            conn.commit()
        return task_id

    def claim_next(self, worker_id: str, lease_seconds: Optional[int] = None, *, now: Optional[str] = None) -> Optional[dict[str, Any]]:
        now = now or _now()
        lease = int(lease_seconds or self.config.get("lease_seconds", 300))
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT * FROM ingestion_tasks
                   WHERE (status='queued' OR (status='retrying' AND (next_retry_at IS NULL OR next_retry_at<=?)))
                   ORDER BY created_at ASC LIMIT 1""",
                (now,),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            lease_until = (_parse_time(now) + timedelta(seconds=lease)).isoformat()
            conn.execute(
                """UPDATE ingestion_tasks SET status='running', worker_id=?, lease_until=?, heartbeat_at=?,
                   started_at=COALESCE(started_at, ?), updated_at=? WHERE task_id=?""",
                (worker_id, lease_until, now, now, now, row["task_id"]),
            )
            claimed = conn.execute("SELECT * FROM ingestion_tasks WHERE task_id=?", (row["task_id"],)).fetchone()
            conn.execute("COMMIT")
        return self._row(claimed)

    def claim_next_task(self, worker_id: str, lease_seconds: Optional[int] = None, *, now: Optional[str] = None) -> Optional[dict[str, Any]]:
        return self.claim_next(worker_id, lease_seconds, now=now)

    def heartbeat(self, task_id: str, worker_id: str, lease_seconds: Optional[int] = None, *, now: Optional[str] = None) -> bool:
        now = now or _now()
        lease = int(lease_seconds or self.config.get("lease_seconds", 300))
        lease_until = (_parse_time(now) + timedelta(seconds=lease)).isoformat()
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """UPDATE ingestion_tasks SET heartbeat_at=?, lease_until=?, updated_at=?
                   WHERE task_id=? AND worker_id=? AND status='running'""",
                (now, lease_until, now, task_id, worker_id),
            )
            conn.commit()
        return cursor.rowcount > 0

    def mark_succeeded(self, task_id: str, *, result: Optional[dict[str, Any]] = None, now: Optional[str] = None) -> bool:
        now = now or _now()
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """UPDATE ingestion_tasks SET status='succeeded', finished_at=?, updated_at=?,
                   last_error_code=NULL, last_error_message=NULL, lease_until=NULL,
                   payload_json=? WHERE task_id=? AND status='running'""",
                (now, now, json.dumps(result or {}, ensure_ascii=False), task_id),
            )
            conn.commit()
        return cursor.rowcount > 0

    def mark_failed(self, task_id: str, error: BaseException | str, error_message: Optional[str] = None, *, error_code: Optional[str] = None, retryable: Optional[bool] = None, now: Optional[str] = None) -> dict[str, Any]:
        now = now or _now()
        # Also accept the specification-friendly form
        # mark_failed(task_id, error_code, error_message).
        if error_message is not None and isinstance(error, str) and error_code is None:
            error_code = error
            error = error_message
        if isinstance(error, BaseException):
            inferred_code, inferred_retryable = classify_error(error)
            message = str(error)
        else:
            inferred_code, inferred_retryable = "UNKNOWN_FATAL_ERROR", False
            message = str(error)
        code = error_code or inferred_code
        can_retry = inferred_retryable if retryable is None else bool(retryable)
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM ingestion_tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                raise KeyError(f"Unknown task_id: {task_id}")
            retry_count = int(row["retry_count"]) + (1 if can_retry else 0)
            max_retries = int(row["max_retries"])
            if can_retry and retry_count < max_retries:
                delay = compute_backoff_delay(
                    retry_count,
                    float(self.config.get("retry_base_delay_seconds", 10)),
                    float(self.config.get("retry_max_delay_seconds", 300)),
                    bool(self.config.get("retry_jitter", True)),
                )
                next_retry = (_parse_time(now) + timedelta(seconds=delay)).isoformat()
                status = "retrying"
                finished_at = None
            elif can_retry:
                status = "dead_letter"
                next_retry = None
                finished_at = now
            else:
                status = "failed"
                next_retry = None
                finished_at = now
            conn.execute(
                """UPDATE ingestion_tasks SET status=?, retry_count=?, next_retry_at=?,
                   last_error_code=?, last_error_message=?, finished_at=?, lease_until=NULL, updated_at=?
                   WHERE task_id=?""",
                (status, retry_count, next_retry, code, message, finished_at, now, task_id),
            )
            updated = conn.execute("SELECT * FROM ingestion_tasks WHERE task_id=?", (task_id,)).fetchone()
            conn.execute("COMMIT")
        return self._row(updated) or {}

    def move_to_dead_letter(self, task_id: str, error_code: str, error_message: str, *, now: Optional[str] = None) -> dict[str, Any]:
        return self._force_dead_letter(task_id, error_code, error_message, now=now)

    def _force_dead_letter(self, task_id: str, error_code: str, error_message: str, *, now: Optional[str] = None) -> dict[str, Any]:
        now = now or _now()
        with closing(self._connect()) as conn:
            conn.execute("UPDATE ingestion_tasks SET status='dead_letter', last_error_code=?, last_error_message=?, finished_at=?, next_retry_at=NULL, lease_until=NULL, updated_at=? WHERE task_id=?", (error_code, error_message, now, now, task_id))
            conn.commit()
        return self.get_task(task_id) or {}

    def recover_stale_tasks(self, now: Optional[str] = None) -> list[dict[str, Any]]:
        now = now or _now()
        recovered: list[dict[str, Any]] = []
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute("SELECT * FROM ingestion_tasks WHERE status='running' AND lease_until IS NOT NULL AND lease_until<=?", (now,)).fetchall()
            for row in rows:
                conn.execute("UPDATE ingestion_tasks SET status='queued', worker_id=NULL, lease_until=NULL, heartbeat_at=NULL, updated_at=? WHERE task_id=?", (now, row["task_id"]))
            due = conn.execute("SELECT * FROM ingestion_tasks WHERE status='retrying' AND (next_retry_at IS NULL OR next_retry_at<=?)", (now,)).fetchall()
            for row in due:
                conn.execute("UPDATE ingestion_tasks SET status='queued', next_retry_at=NULL, updated_at=? WHERE task_id=?", (now, row["task_id"]))
            for row in rows + due:
                updated = conn.execute("SELECT * FROM ingestion_tasks WHERE task_id=?", (row["task_id"],)).fetchone()
                if updated is not None:
                    recovered.append(self._row(updated) or {})
            conn.execute("COMMIT")
        return recovered

    def release_task(self, task_id: str, *, worker_id: Optional[str] = None) -> bool:
        with closing(self._connect()) as conn:
            if worker_id:
                cursor = conn.execute("UPDATE ingestion_tasks SET status='queued', worker_id=NULL, lease_until=NULL, heartbeat_at=NULL, updated_at=? WHERE task_id=? AND worker_id=? AND status='running'", (_now(), task_id, worker_id))
            else:
                cursor = conn.execute("UPDATE ingestion_tasks SET status='queued', worker_id=NULL, lease_until=NULL, heartbeat_at=NULL, updated_at=? WHERE task_id=? AND status='running'", (_now(), task_id))
            conn.commit()
        return cursor.rowcount > 0

    def cancel_task(self, task_id: str) -> bool:
        with closing(self._connect()) as conn:
            cursor = conn.execute("UPDATE ingestion_tasks SET status='cancelled', finished_at=?, lease_until=NULL, updated_at=? WHERE task_id=? AND status IN ('queued','running','retrying','failed')", (_now(), _now(), task_id))
            conn.commit()
        return cursor.rowcount > 0

    def requeue_dead_letter(self, task_id: str, *, clear_error: bool = False) -> bool:
        fields = "status='queued', next_retry_at=NULL, finished_at=NULL, updated_at=?"
        params: list[Any] = [_now()]
        if clear_error:
            fields += ", last_error_code=NULL, last_error_message=NULL"
        params.append(task_id)
        with closing(self._connect()) as conn:
            cursor = conn.execute(f"UPDATE ingestion_tasks SET {fields} WHERE task_id=? AND status='dead_letter'", params)
            conn.commit()
        return cursor.rowcount > 0
