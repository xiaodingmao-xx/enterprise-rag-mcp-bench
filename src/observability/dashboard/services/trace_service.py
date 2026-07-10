"""TraceService – read and parse traces from logs/traces.jsonl.

Provides a typed, filterable interface over the raw JSONL trace log.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from src.core.settings import resolve_path
from src.observability.retention import LogRetentionManager

logger = logging.getLogger(__name__)

# Default path to the traces file (absolute, CWD-independent)
DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")
DEFAULT_AUDIT_PATH = resolve_path("logs/audit.jsonl")
CHINA_TIMEZONE = ZoneInfo("Asia/Shanghai")


class TraceService:
    """Read-only service for querying recorded traces.

    Args:
        traces_path: Path to the JSONL file.  Defaults to
            ``logs/traces.jsonl``.
    """

    def __init__(
        self,
        traces_path: Optional[str | Path] = None,
        audit_path: Optional[str | Path] = None,
    ) -> None:
        self.traces_path = Path(traces_path) if traces_path else DEFAULT_TRACES_PATH
        self.audit_path = Path(audit_path) if audit_path else DEFAULT_AUDIT_PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_traces(
        self,
        trace_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return traces in reverse-chronological order.

        Args:
            trace_type: Filter by ``trace_type`` field (e.g.
                ``"ingestion"`` or ``"query"``).  ``None`` = all.
            limit: Maximum number of traces to return.

        Returns:
            List of trace dicts (newest first).
        """
        traces = self._load_all()

        if trace_type:
            traces = [t for t in traces if t.get("trace_type") == trace_type]

        # Newest first
        traces.sort(key=lambda t: t.get("started_at", ""), reverse=True)

        return traces[:limit]

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single trace by its ``trace_id``.

        Returns:
            Trace dict, or ``None`` if not found.
        """
        for t in self._load_all():
            if t.get("trace_id") == trace_id:
                return t
        return None

    def delete_trace(self, trace_id: str) -> bool:
        """Delete one debug trace record by id; audit records remain untouched."""

        return LogRetentionManager().delete_trace(self.traces_path, trace_id)

    def list_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Read audit events from the separated audit stream."""

        records = self._load_jsonl(self.audit_path)
        records.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        return records[:limit]

    def trace_rows(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Flatten safe trace data into Dashboard rows without raw document text."""

        rows: List[Dict[str, Any]] = []
        for trace in self.list_traces(limit=limit):
            base = {
                "trace_id": trace.get("trace_id", ""),
                "status": trace.get("status", ""),
                "error_code": trace.get("error_code"),
                "duration": trace.get("latency_ms", trace.get("total_elapsed_ms", 0)),
            }
            metadata = trace.get("metadata", {}) if isinstance(trace.get("metadata"), dict) else {}
            default_document_id = metadata.get("document_id") or metadata.get("doc_id")
            emitted = False
            for stage in trace.get("stages", []) or []:
                stage_name = stage.get("stage", "")
                stage_data = stage.get("data", {}) if isinstance(stage.get("data"), dict) else {}
                chunks = stage_data.get("chunks") if isinstance(stage_data.get("chunks"), list) else []
                stage_document_id = stage_data.get("document_id") or stage_data.get("doc_id") or default_document_id
                if chunks:
                    for chunk in chunks:
                        if not isinstance(chunk, dict):
                            continue
                        rows.append({
                            **base,
                            "stage": stage_name,
                            "duration": stage.get("elapsed_ms", base["duration"]),
                            "document_id": chunk.get("document_id") or stage_document_id or "",
                            "chunk_id": chunk.get("chunk_id", ""),
                            "redacted_preview": (
                                chunk.get("redacted_preview")
                                or chunk.get("text_before_preview")
                                or chunk.get("text_after_preview")
                                or ""
                            ),
                        })
                        emitted = True
                if not chunks:
                    rows.append({
                        **base,
                        "stage": stage_name,
                        "duration": stage.get("elapsed_ms", base["duration"]),
                        "document_id": stage_document_id or "",
                        "chunk_id": stage_data.get("chunk_id", ""),
                        "redacted_preview": stage_data.get("redacted_preview", ""),
                    })
                    emitted = True
            if not emitted:
                rows.append({**base, "stage": trace.get("operation", ""), "document_id": default_document_id or "", "chunk_id": "", "redacted_preview": ""})
        return rows[:limit]

    def get_stage_timings(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract stage timings from a trace.

        Returns:
            List of dicts with keys: stage_name, elapsed_ms, data.
            Ordered by appearance.
        """
        stages = trace.get("stages", [])
        timings: List[Dict[str, Any]] = []
        for s in stages:
            # The raw stage dict has: stage, timestamp, data (dict), elapsed_ms
            # Extract the inner 'data' dict directly rather than flattening
            stage_data = s.get("data", {})
            if not isinstance(stage_data, dict):
                stage_data = {}
            if not stage_data:
                stage_data = {
                    key: value
                    for key, value in s.items()
                    if key not in {"stage", "timestamp", "elapsed_ms"}
                }
            timings.append(
                {
                    "stage_name": s.get("stage"),
                    "elapsed_ms": s.get("elapsed_ms", 0),
                    "data": stage_data,
                }
            )
        return timings

    @staticmethod
    def format_china_time(value: Any) -> str:
        """Format an ISO timestamp as China local time.

        Trace files store timestamps in UTC.  Dashboard pages should present
        them in China time so recent events match the user's local clock.
        """
        if not value:
            return "—"

        raw = str(value)
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return raw[:19]

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(CHINA_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> List[Dict[str, Any]]:
        """Parse every line in the JSONL file.

        Silently skips malformed lines.
        """
        if not self.traces_path.exists():
            return []

        return self._load_jsonl(self.traces_path)

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if not path.exists():
            return records
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed trace line: %s", line[:80])
        return records
