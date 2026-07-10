"""Observability logger utilities.

Provides:
- ``get_logger``: standard human-readable logger (unchanged from C-phase).
- ``JSONFormatter``: custom :class:`logging.Formatter` that emits JSON.
- ``get_trace_logger``: returns a logger backed by a JSON Lines file handler.
- ``write_trace``: convenience function to append a trace dict to
  ``logs/traces.jsonl``.
"""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.settings import resolve_path
from src.observability.redaction import config_from_settings, redact_exception, redact_text, redact_value
from src.observability.retention import LogRetentionManager

# Default path for traces file (absolute, CWD-independent)
_DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


# ── Human-readable logger (existing) ────────────────────────────────


def get_logger(name: str = "modular-rag", log_level: Optional[str] = None) -> logging.Logger:
    """Get a configured logger.

    Args:
        name: Logger name.
        log_level: Optional log level string (e.g., "INFO").

    Returns:
        Configured logger instance.
    """

    if log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = logging.INFO

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(SafeTextFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(handler)
    root.setLevel(level)

    # Suppress httpx logs (contains sensitive endpoint URLs)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logging.getLogger(name)


class SafeTextFormatter(logging.Formatter):
    """Human-readable formatter that never prints common secrets/PII."""

    def format(self, record: logging.LogRecord) -> str:
        message = redact_text(record.getMessage(), max_length=2048)
        if record.exc_info and record.exc_info[0] is not None:
            exception = redact_exception(self.formatException(record.exc_info), max_length=4096)
            message = f"{message} | exception={exception}"
        clone = logging.makeLogRecord(record.__dict__.copy())
        clone.msg = message
        clone.args = ()
        clone.exc_info = None
        return super().format(clone)


# ── JSON Lines formatter ────────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    """Logging formatter that outputs one JSON object per line.

    Each log record is serialised to a dict containing at least:
    ``timestamp``, ``level``, ``logger``, ``message``.  If the record
    carries an ``exc_info`` tuple the traceback is included as
    ``exception``.

    Extra attributes attached via *extra=* on the logger call are
    merged into the top-level dict (except internal Python fields).
    """

    _INTERNAL_ATTRS = frozenset({
        "args", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module",
        "msecs", "message", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread",
        "threadName", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        """Return the log record as a single-line JSON string."""
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_text(record.getMessage(), max_length=2048),
        }

        # merge extra fields the caller attached
        for key, val in record.__dict__.items():
            if key not in self._INTERNAL_ATTRS and key not in payload:
                try:
                    json.dumps(val)  # cheap serialisability test
                    payload[key] = redact_value(val, config_from_settings())
                except (TypeError, ValueError):
                    payload[key] = redact_text(val, max_length=2048)

        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = redact_exception(self.formatException(record.exc_info), max_length=4096)

        return json.dumps(payload, ensure_ascii=False)


# ── Trace logger ────────────────────────────────────────────────────


def get_trace_logger(
    traces_path: str | Path = _DEFAULT_TRACES_PATH,
    *,
    name: str = "modular-rag.trace",
    settings: Any = None,
) -> logging.Logger:
    """Return a logger that writes JSON Lines to *traces_path*.

    The logger uses :class:`JSONFormatter` and a :class:`FileHandler`
    configured to append.  Repeated calls with the same *name* return
    the same logger (standard :mod:`logging` semantics).

    Args:
        traces_path: File path for the JSONL output.  Parent directories
            are created automatically.
        name: Logger name.

    Returns:
        A :class:`logging.Logger` ready for JSON Lines output.
    """
    path = Path(traces_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers on repeated calls
    if not logger.handlers:
        observability = getattr(settings, "observability", None) if settings is not None else None
        retention_cfg = getattr(observability, "retention", None)
        retention = LogRetentionManager(
            max_days=getattr(retention_cfg, "max_days", 30),
            max_file_size_mb=getattr(retention_cfg, "max_file_size_mb", 100),
            rotation_count=getattr(retention_cfg, "rotation_count", 5),
        )
        handler = RotatingFileHandler(
            path,
            maxBytes=retention.max_file_size_bytes,
            backupCount=retention.rotation_count,
            encoding="utf-8",
        )
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False  # don't echo to console

    return logger


# ── Convenience writer for trace dicts ──────────────────────────────


def write_trace(
    trace_dict: Dict[str, Any],
    traces_path: str | Path = _DEFAULT_TRACES_PATH,
    *,
    settings: Any = None,
) -> None:
    """Append a single trace dictionary as one JSON line.

    This is a thin wrapper that writes directly — no logging
    framework involved — so the output is identical to what
    :class:`~src.core.trace.trace_collector.TraceCollector` produces.

    Args:
        trace_dict: A JSON-serialisable dictionary (typically from
            ``TraceContext.to_dict()``).
        traces_path: Output file path; parent directories are created
            automatically.
    """
    path = Path(traces_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = config_from_settings(settings)
    line = json.dumps(redact_value(trace_dict, config), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_operational(
    payload: Dict[str, Any],
    operational_path: str | Path = resolve_path("logs/operational.jsonl"),
    *,
    settings: Any = None,
) -> None:
    """Write the low-cardinality operational record separately from debug data."""

    path = Path(operational_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    observability = getattr(settings, "observability", None) if settings is not None else None
    retention_cfg = getattr(observability, "retention", None)
    retention = LogRetentionManager(
        max_days=getattr(retention_cfg, "max_days", 30),
        max_file_size_mb=getattr(retention_cfg, "max_file_size_mb", 100),
        rotation_count=getattr(retention_cfg, "rotation_count", 5),
    )
    retention.rotate_if_needed(path)
    safe_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": payload.get("request_id", ""),
        "trace_id": payload.get("trace_id", ""),
        "tenant_id": payload.get("tenant_id", ""),
        "user_id_hash": payload.get("user_id_hash", "anonymous"),
        "stage": payload.get("stage", payload.get("operation", "")),
        "status": payload.get("status", ""),
        "error_code": payload.get("error_code"),
        "latency_ms": payload.get("latency_ms", 0),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(safe_payload, ensure_ascii=False) + "\n")
