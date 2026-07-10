"""Trace collector – receives finished TraceContext and persists them.

The collector is the bridge between in-memory TraceContext objects and
the on-disk JSON Lines log used by the Dashboard.  It is intentionally
decoupled from the logging module so that trace persistence remains
predictable and testable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.core.settings import resolve_path
from src.core.trace.trace_context import TraceContext
from src.observability.logger import write_operational
from src.observability.redaction import config_from_settings
from src.observability.retention import LogRetentionManager

logger = logging.getLogger(__name__)

# Default absolute path for traces file (CWD-independent)
_DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


class TraceCollector:
    """Collects finished traces and appends them to a JSON Lines file.

    Args:
        traces_path: File path for the ``traces.jsonl`` output.
            Parent directories are created automatically.
    """

    def __init__(self, traces_path: str | Path | None = None, *, settings: Any = None) -> None:
        observability = getattr(settings, "observability", None) if settings is not None else None
        configured_path = getattr(observability, "trace_file", None) if observability else None
        self._path = Path(traces_path or configured_path or _DEFAULT_TRACES_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._settings = settings
        self._redaction_config = config_from_settings(settings)
        retention_cfg = getattr(observability, "retention", None)
        self._retention = LogRetentionManager(
            max_days=getattr(retention_cfg, "max_days", 30),
            max_file_size_mb=getattr(retention_cfg, "max_file_size_mb", 100),
            rotation_count=getattr(retention_cfg, "rotation_count", 5),
        )
        self._operational_path = Path(
            getattr(observability, "operational_file", resolve_path("logs/operational.jsonl"))
            if observability
            else resolve_path("logs/operational.jsonl")
        )

    def collect(self, trace: TraceContext) -> None:
        """Persist a single trace as one JSON line.

        If the trace has not been finished yet, ``finish()`` is called
        automatically so the output always contains timing data.

        Args:
            trace: A populated :class:`TraceContext`.
        """
        if trace.finished_at is None:
            trace.finish()

        trace.configure_security(self._settings or self._redaction_config)
        self._retention.cleanup_directory(self._path.parent)
        self._retention.rotate_if_needed(self._path)
        trace_payload = trace.to_dict(self._redaction_config)
        for stage in trace_payload.get("stages", []):
            write_operational(
                {
                    "request_id": trace_payload.get("request_id"),
                    "trace_id": trace_payload.get("trace_id"),
                    "tenant_id": trace_payload.get("tenant_id"),
                    "user_id_hash": trace_payload.get("user_id_hash"),
                    "stage": stage.get("stage", trace_payload.get("operation")),
                    "status": trace_payload.get("status"),
                    "error_code": trace_payload.get("error_code"),
                    "latency_ms": stage.get("elapsed_ms", trace_payload.get("latency_ms", 0)),
                },
                self._operational_path,
                settings=self._settings,
            )

        # Sampling applies to debug traces only; operational records above are
        # always retained for service health and incident investigation.
        if not trace.should_sample(self._redaction_config):
            return

        line = json.dumps(trace_payload, ensure_ascii=False)
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            logger.error("Failed to write trace %s", trace.trace_id)

    @property
    def path(self) -> Path:
        """Return the resolved path of the traces file."""
        return self._path
