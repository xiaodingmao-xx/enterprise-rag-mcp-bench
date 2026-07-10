"""Retention, rotation, and trace deletion helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable, Optional


class LogRetentionManager:
    """Apply size and age policies to JSONL observability files."""

    def __init__(
        self,
        *,
        max_days: int = 30,
        max_file_size_mb: int = 100,
        rotation_count: int = 5,
    ) -> None:
        self.max_days = max(0, int(max_days))
        self.max_file_size_bytes = max(1, int(max_file_size_mb)) * 1024 * 1024
        self.rotation_count = max(1, int(rotation_count))

    def rotate_if_needed(self, path: str | Path) -> None:
        """Rotate one file if it exceeds the configured maximum size."""

        target = Path(path)
        if not target.exists() or target.stat().st_size < self.max_file_size_bytes:
            return

        # Remove only the explicitly named oldest backup before shifting files.
        oldest = target.with_name(f"{target.name}.{self.rotation_count}")
        if oldest.exists():
            oldest.unlink()
        for index in range(self.rotation_count - 1, 0, -1):
            source = target.with_name(f"{target.name}.{index}")
            destination = target.with_name(f"{target.name}.{index + 1}")
            if source.exists():
                os.replace(source, destination)
        os.replace(target, target.with_name(f"{target.name}.1"))

    def cleanup_expired(self, paths: Iterable[str | Path], *, now: Optional[float] = None) -> int:
        """Delete expired files one explicit path at a time and return a count."""

        if self.max_days <= 0:
            return 0
        cutoff = (time.time() if now is None else now) - self.max_days * 86400
        deleted = 0
        for raw_path in paths:
            path = Path(raw_path)
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted += 1
            except OSError:
                # Retention must not make the request path fail.
                continue
        return deleted

    def cleanup_directory(self, directory: str | Path, *, pattern: str = "*.jsonl") -> int:
        path = Path(directory)
        if not path.exists():
            return 0
        return self.cleanup_expired(path.glob(pattern))

    def delete_trace(self, path: str | Path, trace_id: str) -> bool:
        """Remove one trace record by id without exposing its contents."""

        target = Path(path)
        if not target.exists():
            return False
        temp = target.with_name(target.name + ".delete.tmp")
        removed = False
        try:
            with target.open("r", encoding="utf-8") as source, temp.open("w", encoding="utf-8") as destination:
                for line in source:
                    if line.strip():
                        try:
                            import json

                            record = json.loads(line)
                        except (ValueError, TypeError):
                            record = None
                        if isinstance(record, dict) and record.get("trace_id") == trace_id:
                            removed = True
                            continue
                    destination.write(line)
            if removed:
                os.replace(temp, target)
            else:
                temp.unlink()
        except OSError:
            if temp.exists():
                temp.unlink()
            raise
        return removed

