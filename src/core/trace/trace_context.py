"""Trace context for observability across pipeline stages.

Provides trace_id, trace_type (query/ingestion), per-stage timing,
finish() lifecycle, and to_dict() serialisation for JSON Lines output.
"""

import time
import uuid
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from src.observability.redaction import (
    RedactionConfig,
    config_from_settings,
    hash_user_id,
    redact_trace_payload,
)


@dataclass
class TraceContext:
    """Request-scoped trace context that records pipeline stages and timing.

    Attributes:
        trace_id: Unique identifier for this trace.
        trace_type: Either ``"query"`` or ``"ingestion"``.
        started_at: ISO-8601 timestamp when the trace was created.
        finished_at: ISO-8601 timestamp when ``finish()`` was called, or None.
        stages: Ordered list of recorded stage dicts.
        metadata: Arbitrary key/value pairs attached to the trace.
    """

    trace_type: Literal["query", "ingestion"] = "query"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = "default"
    user_id_hash: str = "anonymous"
    operation: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = field(default=None)
    status: str = "success"
    error_code: Optional[str] = None
    model: Optional[str] = None
    token_usage: Dict[str, Any] = field(default_factory=dict)
    estimated_cost: float = 0.0
    stages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Raw user id is accepted only as an in-memory convenience and is never
    # serialized.  Callers should prefer passing user_id_hash directly.
    user_id: Optional[str] = field(default=None, repr=False, compare=False)
    _redaction_config: Optional[RedactionConfig] = field(default=None, repr=False, compare=False)

    # internal monotonic clock for accurate elapsed calculation
    _start_mono: float = field(default_factory=time.monotonic, repr=False)
    _finish_mono: Optional[float] = field(default=None, repr=False)
    _stage_timings: Dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.operation:
            self.operation = self.trace_type
        if self.user_id is not None and (not self.user_id_hash or self.user_id_hash == "anonymous"):
            self.user_id_hash = hash_user_id(self.user_id)
        if not self.user_id_hash:
            self.user_id_hash = "anonymous"

    def configure_security(self, settings: Any = None) -> None:
        """Attach the application redaction policy to this request context."""

        self._redaction_config = config_from_settings(settings)

    def set_context(
        self,
        *,
        request_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        operation: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Set request-scoped identity without retaining raw user data."""

        if request_id:
            self.request_id = request_id
        if tenant_id:
            self.tenant_id = tenant_id
        if user_id is not None:
            config = self._redaction_config or RedactionConfig()
            self.user_id_hash = hash_user_id(user_id, enabled=config.hash_user_id)
        if operation:
            self.operation = operation
        if model:
            self.model = model

    # ---- recording ---------------------------------------------------

    def record_stage(
        self,
        stage_name: str,
        data: Dict[str, Any],
        elapsed_ms: Optional[float] = None,
    ) -> None:
        """Record data from a pipeline stage.

        Args:
            stage_name: Name of the stage (e.g. ``"dense_retrieval"``).
            data: Stage-specific payload (method, provider, details …).
            elapsed_ms: Pre-computed elapsed time in ms.  If *None* the
                caller should measure externally, or leave it to the
                ``stage_timer`` context-manager.
        """
        entry: Dict[str, Any] = {
            "stage": stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        if elapsed_ms is not None:
            entry["elapsed_ms"] = round(elapsed_ms, 2)
            self._stage_timings[stage_name] = elapsed_ms
        self.stages.append(entry)

    # ---- lifecycle ----------------------------------------------------

    def finish(self, *, status: Optional[str] = None, error_code: Optional[str] = None) -> None:
        """Mark the trace as finished and record wall-clock end time."""
        self._finish_mono = time.monotonic()
        self.finished_at = datetime.now(timezone.utc).isoformat()
        if status:
            self.status = status
        if error_code:
            self.error_code = error_code

    # ---- timing helpers -----------------------------------------------

    def elapsed_ms(self, stage_name: Optional[str] = None) -> float:
        """Return elapsed time in milliseconds.

        Args:
            stage_name: If given, return the elapsed time recorded for
                that stage.  If *None*, return the total trace elapsed
                time (start → finish, or start → now if not yet
                finished).

        Returns:
            Elapsed milliseconds.

        Raises:
            KeyError: If *stage_name* was provided but not found.
        """
        if stage_name is not None:
            if stage_name not in self._stage_timings:
                raise KeyError(f"Stage '{stage_name}' has no recorded timing")
            return self._stage_timings[stage_name]

        end = self._finish_mono if self._finish_mono is not None else time.monotonic()
        return (end - self._start_mono) * 1000.0

    # ---- serialisation ------------------------------------------------

    def should_sample(self, config: Optional[RedactionConfig] = None) -> bool:
        """Return whether this debug trace is selected by the sampling policy."""

        cfg = config or self._redaction_config or RedactionConfig()
        if cfg.sampling_rate >= 1.0:
            return True
        if cfg.sampling_rate <= 0.0:
            return False
        # Stable sampling keeps the same trace selected across retries/processes.
        digest = hashlib.sha256(self.trace_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:12], 16) / float(16**12)
        return bucket < cfg.sampling_rate

    def to_dict(self, redaction_config: Optional[RedactionConfig] = None) -> Dict[str, Any]:
        """Serialise the trace to a plain dict suitable for ``json.dumps``.

        Returns:
            Dictionary with all trace data.
        """
        total_elapsed_ms = self.elapsed_ms()
        rounded_elapsed_ms = round(total_elapsed_ms, 2)
        if self.finished_at is not None and rounded_elapsed_ms == 0:
            rounded_elapsed_ms = 0.01

        raw_payload = {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "user_id_hash": self.user_id_hash,
            "operation": self.operation,
            "start_time": self.started_at,
            "end_time": self.finished_at,
            "latency_ms": rounded_elapsed_ms,
            "status": self.status,
            "error_code": self.error_code,
            "model": self.model,
            "token_usage": dict(self.token_usage),
            "estimated_cost": float(self.estimated_cost or 0.0),
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": rounded_elapsed_ms,
            "stages": list(self.stages),
            "metadata": dict(self.metadata),
        }
        config = redaction_config or self._redaction_config or RedactionConfig()
        return redact_trace_payload(raw_payload, config)

    # ---- backwards-compat helper used in C5 / C6 -----------------------

    def get_stage_data(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve recorded data for a specific stage.

        Searches stages list (last-write-wins for duplicate names).

        Args:
            stage_name: Name of the stage to retrieve.

        Returns:
            The ``data`` dict of the matching stage, or *None*.
        """
        for entry in reversed(self.stages):
            if entry.get("stage") == stage_name:
                return entry.get("data")
        return None
