"""Separated audit log writer for security-relevant actions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.core.settings import resolve_path
from src.observability.redaction import config_from_settings, hash_user_id, redact_mapping, redact_text
from src.observability.retention import LogRetentionManager


class AuditLogger:
    """Write append-only audit events to a file separate from debug traces."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        settings: Any = None,
        retention: Optional[LogRetentionManager] = None,
    ) -> None:
        observability = getattr(settings, "observability", None) if settings is not None else None
        if observability is not None and observability.__class__.__module__ == "unittest.mock":
            observability = None
        configured = getattr(observability, "audit_file", "./logs/audit.jsonl") if observability else "./logs/audit.jsonl"
        if not isinstance(configured, (str, Path)):
            configured = "./logs/audit.jsonl"
        self.path = Path(path or resolve_path(configured))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config_from_settings(settings)
        self.retention = retention or LogRetentionManager(
            max_days=getattr(getattr(observability, "retention", None), "max_days", 30),
            max_file_size_mb=getattr(getattr(observability, "retention", None), "max_file_size_mb", 100),
            rotation_count=getattr(getattr(observability, "retention", None), "rotation_count", 5),
        )

    def write(
        self,
        *,
        trace_id: str,
        request_id: str,
        tenant_id: str,
        user_id: Any = None,
        user_id_hash: Optional[str] = None,
        operation: str,
        resource: str,
        tool: Optional[str] = None,
        query: Optional[str] = None,
        document_ids: Optional[Iterable[str]] = None,
        permission_denied: bool = False,
        success: bool = True,
        error_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append a minimally sufficient audit event."""

        actor_hash = user_id_hash or hash_user_id(user_id, enabled=self.config.hash_user_id)
        event: Dict[str, Any] = {
            "event_type": "audit",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "request_id": request_id,
            "tenant_id": redact_text(tenant_id, max_length=128),
            "user_id_hash": actor_hash,
            "operation": operation,
            "resource": redact_text(resource, max_length=256, is_path=False),
            "tool": tool or "",
            "permission_denied": bool(permission_denied),
            "document_ids": [str(item) for item in (document_ids or []) if item is not None],
            "success": bool(success),
            "error_code": error_code,
        }
        if query:
            event["query_hash"] = hash_user_id(query)
            event["query_preview"] = redact_text(query, max_length=self.config.max_text_length)
        event = redact_mapping(event, self.config)
        self.retention.rotate_if_needed(self.path)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event
