"""Persistent chunk-level embedding cache.

The cache stores dense vectors by content hash plus embedding namespace. This
lets ingestion reuse vectors for unchanged chunks even when the surrounding
file hash changes.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import resolve_path
from src.core.types import Chunk


class SQLiteEmbeddingCache:
    """SQLite-backed cache for dense embeddings."""

    def __init__(self, db_path: str | Path, enabled: bool = True) -> None:
        self.db_path = resolve_path(db_path)
        self.enabled = enabled
        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def annotate_chunks(self, chunks: List[Chunk]) -> None:
        """Attach full content hashes to chunk metadata in-place."""
        for chunk in chunks:
            chunk.metadata["content_hash"] = self.content_hash(chunk.text)

    def get_many(
        self,
        chunks: List[Chunk],
        *,
        collection: str,
        provider: str,
        model: str,
        dimensions: int,
    ) -> Dict[int, List[float]]:
        """Return cached vectors keyed by chunk index."""
        if not self.enabled or not chunks:
            return {}

        hits: Dict[int, List[float]] = {}
        with self._connect() as conn:
            for idx, chunk in enumerate(chunks):
                content_hash = chunk.metadata.get("content_hash")
                if not content_hash:
                    content_hash = self.content_hash(chunk.text)
                    chunk.metadata["content_hash"] = content_hash

                row = conn.execute(
                    """
                    SELECT vector_json
                    FROM embedding_cache
                    WHERE collection = ?
                      AND provider = ?
                      AND model = ?
                      AND dimensions = ?
                      AND content_hash = ?
                    """,
                    (collection, provider, model, dimensions, content_hash),
                ).fetchone()

                if row is None:
                    continue

                vector = self._decode_vector(row[0])
                if vector is not None:
                    hits[idx] = vector

        return hits

    def set_many(
        self,
        chunks: List[Chunk],
        vectors: List[List[float]],
        *,
        collection: str,
        provider: str,
        model: str,
        dimensions: int,
    ) -> None:
        """Store vectors for chunks in the same order."""
        if not self.enabled or not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have same length"
            )

        rows = []
        for chunk, vector in zip(chunks, vectors):
            content_hash = chunk.metadata.get("content_hash")
            if not content_hash:
                content_hash = self.content_hash(chunk.text)
                chunk.metadata["content_hash"] = content_hash

            rows.append(
                (
                    collection,
                    provider,
                    model,
                    dimensions,
                    content_hash,
                    len(chunk.text),
                    json.dumps(vector),
                )
            )

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO embedding_cache (
                    collection,
                    provider,
                    model,
                    dimensions,
                    content_hash,
                    text_length,
                    vector_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(collection, provider, model, dimensions, content_hash)
                DO UPDATE SET
                    text_length = excluded.text_length,
                    vector_json = excluded.vector_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                rows,
            )

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    collection TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    text_length INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (
                        collection,
                        provider,
                        model,
                        dimensions,
                        content_hash
                    )
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embedding_cache_content_hash
                ON embedding_cache(content_hash)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _decode_vector(self, raw: str) -> Optional[List[float]]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, list):
            return None

        try:
            return [float(value) for value in data]
        except (TypeError, ValueError):
            return None
