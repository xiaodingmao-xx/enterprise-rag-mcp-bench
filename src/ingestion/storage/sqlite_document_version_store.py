"""SQLite implementation of immutable document versions.

The schema is intentionally local-first.  All writes use short transactions
and WAL so the same interface can later be backed by PostgreSQL.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.core.settings import resolve_path
from src.ingestion.storage.document_version_store import (
    DocumentRecord,
    DocumentVersion,
    DocumentVersionStore,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteDocumentVersionStore(DocumentVersionStore):
    """Persist document identities and version fingerprints in SQLite."""

    def __init__(self, db_path: str | Path = "./data/db/document_versions.db", audit_log_store: Any = None) -> None:
        self.db_path = resolve_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_log_store = audit_log_store
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS document_records (
                    document_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    external_document_id TEXT NOT NULL,
                    current_version_id TEXT,
                    title TEXT NOT NULL DEFAULT '',
                    source_uri TEXT NOT NULL DEFAULT '',
                    source_type TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL CHECK(status IN ('active','deleted')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    UNIQUE(tenant_id, source_id, external_document_id)
                );
                CREATE TABLE IF NOT EXISTS document_versions (
                    version_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    metadata_hash TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    chunker_version TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('processing','active','failed','deleted')),
                    error_message TEXT,
                    FOREIGN KEY(document_id) REFERENCES document_records(document_id),
                    UNIQUE(document_id, content_hash, metadata_hash, parser_version,
                           chunker_version, embedding_model)
                );
                CREATE INDEX IF NOT EXISTS idx_document_records_tenant_source_external
                    ON document_records(tenant_id, source_id, external_document_id);
                CREATE INDEX IF NOT EXISTS idx_document_versions_document_id
                    ON document_versions(document_id);
                CREATE INDEX IF NOT EXISTS idx_document_versions_fingerprint
                    ON document_versions(document_id, content_hash, metadata_hash,
                                         parser_version, chunker_version, embedding_model);
                """
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(**dict(row))

    @staticmethod
    def _version(row: sqlite3.Row) -> DocumentVersion:
        return DocumentVersion(**dict(row))

    def get_or_create_record(
        self,
        tenant_id: str,
        source_id: str,
        external_document_id: str,
        *,
        document_id: Optional[str] = None,
        title: str = "",
        source_uri: str = "",
        source_type: str = "",
    ) -> DocumentRecord:
        if not tenant_id or not source_id or not external_document_id:
            raise ValueError("tenant_id, source_id and external_document_id are required")
        now = _utc_now()
        logical_id = document_id or uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{tenant_id}:{source_id}:{external_document_id}",
        ).hex
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT OR IGNORE INTO document_records
                   (document_id, tenant_id, source_id, external_document_id,
                    title, source_uri, source_type, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
                (logical_id, tenant_id, source_id, external_document_id, title, source_uri, source_type, now, now),
            )
            conn.execute(
                """UPDATE document_records SET title=COALESCE(NULLIF(?, ''), title),
                   source_uri=COALESCE(NULLIF(?, ''), source_uri),
                   source_type=COALESCE(NULLIF(?, ''), source_type),
                   updated_at=?, status='active', deleted_at=NULL
                   WHERE tenant_id=? AND source_id=? AND external_document_id=?""",
                (title, source_uri, source_type, now, tenant_id, source_id, external_document_id),
            )
            row = conn.execute(
                "SELECT * FROM document_records WHERE tenant_id=? AND source_id=? AND external_document_id=?",
                (tenant_id, source_id, external_document_id),
            ).fetchone()
            conn.execute("COMMIT")
        assert row is not None
        return self._record(row)

    # Verbose aliases keep the public API readable for callers outside the
    # ingestion package and match the original implementation specification.
    get_or_create_document_record = get_or_create_record

    def find_existing_version(
        self,
        document_id: str,
        content_hash: str,
        metadata_hash: str,
        parser_version: str,
        chunker_version: str,
        embedding_model: str,
    ) -> Optional[DocumentVersion]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """SELECT * FROM document_versions WHERE document_id=? AND content_hash=?
                   AND metadata_hash=? AND parser_version=? AND chunker_version=?
                   AND embedding_model=?""",
                (document_id, content_hash, metadata_hash, parser_version, chunker_version, embedding_model),
            ).fetchone()
        return self._version(row) if row is not None else None

    def create_version(
        self,
        document_id: str,
        content_hash: str,
        metadata_hash: str,
        parser_version: str = "unknown",
        chunker_version: str = "unknown",
        embedding_model: str = "unknown",
        *,
        version_id: Optional[str] = None,
    ) -> DocumentVersion:
        now = _utc_now()
        new_id = version_id or uuid.uuid4().hex
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT OR IGNORE INTO document_versions
                   (version_id, document_id, content_hash, metadata_hash, parser_version,
                    chunker_version, embedding_model, created_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'processing')""",
                (new_id, document_id, content_hash, metadata_hash, parser_version,
                 chunker_version, embedding_model, now),
            )
            row = conn.execute(
                """SELECT * FROM document_versions WHERE document_id=? AND content_hash=?
                   AND metadata_hash=? AND parser_version=? AND chunker_version=?
                   AND embedding_model=?""",
                (document_id, content_hash, metadata_hash, parser_version, chunker_version, embedding_model),
            ).fetchone()
            conn.execute("COMMIT")
        assert row is not None
        return self._version(row)

    create_document_version = create_version

    def ingest_document_idempotent(self, **kwargs: Any) -> DocumentVersion:
        """Create or return a version for a complete fingerprint.

        The caller supplies the document identity fields together with the
        version fingerprint.  No parsing or index work belongs in this method;
        returning an existing row is the signal for the pipeline to skip it.
        """
        record_keys = {
            "tenant_id", "source_id", "external_document_id", "document_id",
            "title", "source_uri", "source_type",
        }
        record = self.get_or_create_record(**{key: value for key, value in kwargs.items() if key in record_keys})
        version_keys = {
            "content_hash", "metadata_hash", "parser_version", "chunker_version",
            "embedding_model", "version_id",
        }
        existing = self.find_existing_version(document_id=record.document_id, **{key: value for key, value in kwargs.items() if key in version_keys and key != "version_id"})
        if existing is not None:
            if existing.status in {"failed", "deleted"}:
                return self.reset_version(existing.version_id)
            return existing
        return self.create_version(document_id=record.document_id, **{key: value for key, value in kwargs.items() if key in version_keys})

    def mark_version_active(self, version_id: str) -> DocumentVersion:
        with closing(self._connect()) as conn:
            conn.execute("UPDATE document_versions SET status='active', error_message=NULL WHERE version_id=?", (version_id,))
            row = conn.execute("SELECT * FROM document_versions WHERE version_id=?", (version_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown version_id: {version_id}")
        return self._version(row)

    def mark_version_failed(self, version_id: str, error_message: str) -> DocumentVersion:
        with closing(self._connect()) as conn:
            conn.execute("UPDATE document_versions SET status='failed', error_message=? WHERE version_id=?", (str(error_message), version_id))
            row = conn.execute("SELECT * FROM document_versions WHERE version_id=?", (version_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown version_id: {version_id}")
        return self._version(row)

    def reset_version(self, version_id: str) -> DocumentVersion:
        """Move a failed/deleted fingerprint back to processing for retry."""
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE document_versions SET status='processing', error_message=NULL WHERE version_id=?",
                (version_id,),
            )
            row = conn.execute("SELECT * FROM document_versions WHERE version_id=?", (version_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown version_id: {version_id}")
        return self._version(row)

    def activate_version(self, document_id: str, version_id: str, *, tenant_id: str = "", actor: str = "system") -> DocumentRecord:
        now = _utc_now()
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            version = conn.execute("SELECT * FROM document_versions WHERE version_id=? AND document_id=?", (version_id, document_id)).fetchone()
            if version is None or version["status"] == "deleted":
                conn.execute("ROLLBACK")
                raise ValueError("Version does not exist or is deleted")
            conn.execute("UPDATE document_versions SET status='active', error_message=NULL WHERE version_id=?", (version_id,))
            conn.execute("UPDATE document_records SET current_version_id=?, status='active', updated_at=?, deleted_at=NULL WHERE document_id=?", (version_id, now, document_id))
            row = conn.execute("SELECT * FROM document_records WHERE document_id=?", (document_id,)).fetchone()
            conn.execute("COMMIT")
        if row is None:
            raise KeyError(f"Unknown document_id: {document_id}")
        self._audit("activate_version", document_id, version_id, tenant_id, actor)
        return self._record(row)

    def rollback_to_version(self, document_id: str, version_id: str, **kwargs: Any) -> DocumentRecord:
        return self.activate_version(document_id, version_id, **kwargs)

    def delete_version(self, document_id: str, version_id: str, *, tenant_id: str = "", actor: str = "system") -> DocumentRecord:
        now = _utc_now()
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            target = conn.execute("SELECT status FROM document_versions WHERE document_id=? AND version_id=?", (document_id, version_id)).fetchone()
            if target is None:
                conn.execute("ROLLBACK")
                raise KeyError(f"Unknown version_id: {version_id}")
            conn.execute("UPDATE document_versions SET status='deleted' WHERE document_id=? AND version_id=?", (document_id, version_id))
            record = conn.execute("SELECT * FROM document_records WHERE document_id=?", (document_id,)).fetchone()
            if record is None:
                conn.execute("ROLLBACK")
                raise KeyError(f"Unknown document_id: {document_id}")
            if record["current_version_id"] == version_id:
                replacement = conn.execute("""SELECT version_id FROM document_versions WHERE document_id=? AND status='active' AND version_id<>? ORDER BY created_at DESC LIMIT 1""", (document_id, version_id)).fetchone()
                if replacement:
                    conn.execute("UPDATE document_records SET current_version_id=?, status='active', updated_at=?, deleted_at=NULL WHERE document_id=?", (replacement[0], now, document_id))
                else:
                    conn.execute("UPDATE document_records SET current_version_id=NULL, status='deleted', updated_at=?, deleted_at=? WHERE document_id=?", (now, now, document_id))
            else:
                conn.execute("UPDATE document_records SET updated_at=? WHERE document_id=?", (now, document_id))
            row = conn.execute("SELECT * FROM document_records WHERE document_id=?", (document_id,)).fetchone()
            conn.execute("COMMIT")
        self._audit("delete_version", document_id, version_id, tenant_id, actor)
        return self._record(row)

    def delete_all_versions(self, document_id: str, *, tenant_id: str = "", actor: str = "system") -> DocumentRecord:
        now = _utc_now()
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE document_versions SET status='deleted' WHERE document_id=?", (document_id,))
            conn.execute("UPDATE document_records SET current_version_id=NULL, status='deleted', updated_at=?, deleted_at=? WHERE document_id=?", (now, now, document_id))
            row = conn.execute("SELECT * FROM document_records WHERE document_id=?", (document_id,)).fetchone()
            conn.execute("COMMIT")
        if row is None:
            raise KeyError(f"Unknown document_id: {document_id}")
        self._audit("delete_all_versions", document_id, None, tenant_id, actor)
        return self._record(row)

    def get_record(self, document_id: str) -> Optional[DocumentRecord]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM document_records WHERE document_id=?", (document_id,)).fetchone()
        return self._record(row) if row is not None else None

    def list_versions(self, document_id: str) -> list[DocumentVersion]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT * FROM document_versions WHERE document_id=? ORDER BY created_at ASC", (document_id,)).fetchall()
        return [self._version(row) for row in rows]

    def _audit(self, action: str, document_id: str, version_id: Optional[str], tenant_id: str, actor: str) -> None:
        if self.audit_log_store is not None:
            self.audit_log_store.write(action=action, document_id=document_id, version_id=version_id, tenant_id=tenant_id, actor=actor)
