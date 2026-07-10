"""SQLite audit trail for document and ingestion lifecycle operations."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.core.settings import resolve_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteAuditLogStore:
    """Append-only audit log kept separate from operational/debug traces."""

    def __init__(self, db_path: str | Path = "./data/db/ingestion_audit.db") -> None:
        self.db_path = resolve_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS audit_logs (
                    audit_id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    document_id TEXT,
                    version_id TEXT,
                    tenant_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    detail_json TEXT NOT NULL DEFAULT '{}'
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_document_id ON audit_logs(document_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_tenant_created ON audit_logs(tenant_id, created_at)")
            conn.commit()

    def write(
        self,
        *,
        action: str,
        document_id: Optional[str] = None,
        version_id: Optional[str] = None,
        tenant_id: str = "",
        actor: str = "system",
        detail: Optional[dict[str, Any]] = None,
    ) -> str:
        audit_id = uuid.uuid4().hex
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO audit_logs
                   (audit_id, action, document_id, version_id, tenant_id, actor, created_at, detail_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (audit_id, action, document_id, version_id, tenant_id, actor, _utc_now(), json.dumps(detail or {}, ensure_ascii=False)),
            )
            conn.commit()
        return audit_id

    def list(self, *, document_id: Optional[str] = None, tenant_id: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if document_id is not None:
            clauses.append("document_id=?")
            params.append(document_id)
        if tenant_id is not None:
            clauses.append("tenant_id=?")
            params.append(tenant_id)
        sql = "SELECT * FROM audit_logs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["detail"] = json.loads(item.pop("detail_json"))
            except (TypeError, ValueError):
                item["detail"] = {}
            result.append(item)
        return result

