"""Worker-start recovery helpers for the SQLite task backend."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class TaskRecovery:
    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def recover(self, now: str | datetime | None = None) -> list[dict[str, Any]]:
        if now is None:
            now = datetime.now(timezone.utc).isoformat()
        elif isinstance(now, datetime):
            now = now.astimezone(timezone.utc).isoformat()
        return self.backend.recover_stale_tasks(str(now))


def recover_stale_tasks(backend: Any, now: str | datetime | None = None) -> list[dict[str, Any]]:
    """Functional convenience wrapper for worker startup hooks."""
    return TaskRecovery(backend).recover(now)
