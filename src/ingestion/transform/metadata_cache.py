"""SQLite cache for metadata enrichment results."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.settings import resolve_path
from src.observability.logger import get_logger

logger = get_logger(__name__)


def sha256_text(text: str) -> str:
    """Return SHA256 hash for text."""

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def stable_config_hash(config: dict[str, Any]) -> str:
    """Return stable hash for enrichment cache-relevant config."""

    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str)
    return sha256_text(payload)


def build_cache_key(chunk_id: str, text_hash: str, config_hash: str) -> str:
    """Build deterministic cache key from chunk id, text hash, and config hash."""

    return sha256_text(f"{chunk_id}:{text_hash}:{config_hash}")


class MetadataEnrichmentCache:
    """Small SQLite-backed cache for metadata enrichment output."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = resolve_path(db_path)
        self._available = True
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
        except Exception as exc:
            self._available = False
            logger.warning(f"Metadata enrichment cache disabled: {exc}")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=30)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata_enrichment_cache (
                    cache_key TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    enrichment_result TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metadata_cache_chunk_id "
                "ON metadata_enrichment_cache(chunk_id)"
            )

    def get(self, cache_key: str) -> dict[str, Any] | None:
        """Return cached enrichment result, or None on miss/failure."""

        if not self._available:
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT enrichment_result FROM metadata_enrichment_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
            if not row:
                return None
            parsed = json.loads(row[0])
            return parsed if isinstance(parsed, dict) else None
        except Exception as exc:
            logger.warning(f"Metadata enrichment cache read failed: {exc}")
            return None

    def set(
        self,
        *,
        cache_key: str,
        chunk_id: str,
        text_hash: str,
        config_hash: str,
        enrichment_result: dict[str, Any],
    ) -> None:
        """Write enrichment result to cache. Failures are non-fatal."""

        if not self._available:
            return
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(enrichment_result, ensure_ascii=False, sort_keys=True)
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO metadata_enrichment_cache (
                        cache_key, chunk_id, text_hash, config_hash,
                        enrichment_result, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        enrichment_result = excluded.enrichment_result,
                        updated_at = excluded.updated_at
                    """,
                    (
                        cache_key,
                        chunk_id,
                        text_hash,
                        config_hash,
                        payload,
                        now,
                        now,
                    ),
                )
        except Exception as exc:
            logger.warning(f"Metadata enrichment cache write failed: {exc}")
