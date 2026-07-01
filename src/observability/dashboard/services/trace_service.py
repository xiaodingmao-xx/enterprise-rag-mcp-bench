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

logger = logging.getLogger(__name__)

# Default path to the traces file (absolute, CWD-independent)
DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")
CHINA_TIMEZONE = ZoneInfo("Asia/Shanghai")


class TraceService:
    """Read-only service for querying recorded traces.

    Args:
        traces_path: Path to the JSONL file.  Defaults to
            ``logs/traces.jsonl``.
    """

    def __init__(self, traces_path: Optional[str | Path] = None) -> None:
        self.traces_path = Path(traces_path) if traces_path else DEFAULT_TRACES_PATH

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

        traces: List[Dict[str, Any]] = []
        with self.traces_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed trace line: %s", line[:80])
        return traces
