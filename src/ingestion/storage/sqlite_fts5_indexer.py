"""SQLite FTS5 sparse indexer.

This module provides an incremental, transaction-safe sparse retrieval backend
using SQLite FTS5. It is intended to replace the JSON BM25 index for ingestion
workloads where documents are added one at a time.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class SQLiteFTS5Indexer:
    """Incremental sparse index backed by SQLite FTS5."""

    supports_collection_query = True
    requires_chunk_text = True

    def __init__(
        self,
        db_path: str,
        tokenizer: str = "unicode61",
        match_mode: str = "or",
        busy_timeout_ms: int = 10000,
        max_retries: int = 3,
    ) -> None:
        self.db_path = Path(db_path)
        self.tokenizer = self._normalise_tokenizer(tokenizer)
        self.match_mode = self._normalise_match_mode(match_mode)
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.max_retries = int(max_retries)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise_schema()

    def add_documents(
        self,
        term_stats: Optional[List[Dict[str, Any]]] = None,
        *,
        chunks: Optional[Sequence[Any]] = None,
        chunk_ids: Optional[Sequence[str]] = None,
        collection: str = "default",
        doc_id: Optional[str] = None,
        source_path: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> None:
        """Add or replace all chunks for a document.

        Args:
            term_stats: Optional sparse stats from ``SparseEncoder``. Only
                ``doc_length`` is used by FTS5 storage.
            chunks: Processed chunks containing text and metadata.
            chunk_ids: Optional vector-store IDs aligned with ``chunks``.
            collection: Target collection.
            doc_id: Stable document identifier.
            source_path: Original source path, used to remove stale versions.
            trace: Optional trace context.
        """
        if not chunks:
            return

        resolved_doc_id = str(doc_id or self._infer_doc_id(chunks, source_path))
        resolved_source_path = str(source_path or self._infer_source_path(chunks) or "")
        stats = term_stats or []
        rows = self._prepare_rows(
            chunks=chunks,
            chunk_ids=chunk_ids,
            term_stats=stats,
            collection=collection,
            doc_id=resolved_doc_id,
            source_path=resolved_source_path,
        )

        def write(conn: sqlite3.Connection) -> None:
            if resolved_source_path:
                self._delete_by_source_path_tx(conn, collection, resolved_source_path)
            self._delete_document_tx(conn, collection, resolved_doc_id)
            conn.executemany(
                """
                INSERT INTO sparse_chunks (
                    collection, chunk_id, doc_id, source_path, text,
                    doc_length, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["collection"],
                        row["chunk_id"],
                        row["doc_id"],
                        row["source_path"],
                        row["text"],
                        row["doc_length"],
                        row["metadata_json"],
                        row["updated_at"],
                    )
                    for row in rows
                ],
            )
            conn.executemany(
                """
                INSERT INTO sparse_chunks_fts (
                    text, chunk_id, collection, doc_id, source_path
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["text"],
                        row["chunk_id"],
                        row["collection"],
                        row["doc_id"],
                        row["source_path"],
                    )
                    for row in rows
                ],
            )

        self._write_with_retry(write)
        logger.debug(
            "SQLite FTS5 indexed %s chunks for doc_id=%s collection=%s",
            len(rows),
            resolved_doc_id,
            collection,
        )

    def remove_document(
        self,
        doc_id: str,
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> bool:
        """Remove all chunks for a document."""
        removed = {"count": 0}

        def write(conn: sqlite3.Connection) -> None:
            removed["count"] = self._delete_document_tx(conn, collection, str(doc_id))

        self._write_with_retry(write)
        return removed["count"] > 0

    def remove_by_source_path(
        self,
        collection: str,
        source_path: str,
        trace: Optional[Any] = None,
    ) -> int:
        """Remove all chunks matching a source path."""
        removed = {"count": 0}

        def write(conn: sqlite3.Connection) -> None:
            removed["count"] = self._delete_by_source_path_tx(
                conn,
                collection,
                str(source_path),
            )

        self._write_with_retry(write)
        return removed["count"]

    def load(self, collection: str = "default", trace: Optional[Any] = None) -> bool:
        """Return whether the collection has indexed chunks."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM sparse_chunks WHERE collection = ? LIMIT 1",
                (collection,),
            ).fetchone()
        finally:
            conn.close()
        return row is not None

    def query(
        self,
        query_terms: List[str],
        top_k: int = 10,
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Query FTS5 and return results ordered by relevance."""
        if not query_terms:
            return []

        match_query = self._build_match_query(query_terms)
        if not match_query:
            return []

        sql = """
            SELECT
                f.chunk_id AS chunk_id,
                m.text AS text,
                m.metadata_json AS metadata_json,
                bm25(sparse_chunks_fts) AS raw_score
            FROM sparse_chunks_fts f
            JOIN sparse_chunks m
              ON m.collection = f.collection
             AND m.chunk_id = f.chunk_id
            WHERE sparse_chunks_fts MATCH ?
              AND f.collection = ?
            ORDER BY raw_score ASC
            LIMIT ?
        """
        try:
            conn = self._connect()
            try:
                rows = conn.execute(sql, (match_query, collection, int(top_k))).fetchall()
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"SQLite FTS5 query failed: {exc}") from exc

        results: List[Dict[str, Any]] = []
        for row in rows:
            raw_score = float(row["raw_score"])
            score = self._normalise_score(raw_score)
            metadata = self._load_metadata(row["metadata_json"])
            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "score": score,
                    "raw_score": raw_score,
                    "text": row["text"],
                    "metadata": metadata,
                }
            )
        return results

    def count_chunks(self, collection: str = "default") -> int:
        """Return number of indexed chunks for a collection."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM sparse_chunks WHERE collection = ?",
                (collection,),
            ).fetchone()
        finally:
            conn.close()
        return int(row["count"])

    def _initialise_schema(self) -> None:
        tokenizer_sql = self._tokenizer_sql()
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sparse_chunks (
                    collection TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    source_path TEXT,
                    text TEXT NOT NULL,
                    doc_length INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (collection, chunk_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sparse_chunks_doc
                ON sparse_chunks(collection, doc_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sparse_chunks_source
                ON sparse_chunks(collection, source_path)
                """
            )
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS sparse_chunks_fts USING fts5(
                    text,
                    chunk_id UNINDEXED,
                    collection UNINDEXED,
                    doc_id UNINDEXED,
                    source_path UNINDEXED,
                    tokenize = {tokenizer_sql}
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=max(self.busy_timeout_ms / 1000.0, 0.001),
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        return conn

    def _write_with_retry(self, operation: Any) -> None:
        delay_seconds = 0.05
        attempts = max(self.max_retries, 0) + 1
        for attempt in range(attempts):
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                operation(conn)
                conn.commit()
                return
            except sqlite3.OperationalError as exc:
                conn.rollback()
                if "locked" not in str(exc).lower() or attempt == attempts - 1:
                    raise
                time.sleep(delay_seconds)
                delay_seconds *= 2
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _delete_document_tx(
        self,
        conn: sqlite3.Connection,
        collection: str,
        doc_id: str,
    ) -> int:
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM sparse_chunks
            WHERE collection = ? AND doc_id = ?
            """,
            (collection, doc_id),
        ).fetchone()
        count = int(row["count"])
        conn.execute(
            "DELETE FROM sparse_chunks_fts WHERE collection = ? AND doc_id = ?",
            (collection, doc_id),
        )
        conn.execute(
            "DELETE FROM sparse_chunks WHERE collection = ? AND doc_id = ?",
            (collection, doc_id),
        )
        return count

    def _delete_by_source_path_tx(
        self,
        conn: sqlite3.Connection,
        collection: str,
        source_path: str,
    ) -> int:
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM sparse_chunks
            WHERE collection = ? AND source_path = ?
            """,
            (collection, source_path),
        ).fetchone()
        count = int(row["count"])
        conn.execute(
            """
            DELETE FROM sparse_chunks_fts
            WHERE collection = ? AND source_path = ?
            """,
            (collection, source_path),
        )
        conn.execute(
            """
            DELETE FROM sparse_chunks
            WHERE collection = ? AND source_path = ?
            """,
            (collection, source_path),
        )
        return count

    def _prepare_rows(
        self,
        *,
        chunks: Sequence[Any],
        chunk_ids: Optional[Sequence[str]],
        term_stats: Sequence[Dict[str, Any]],
        collection: str,
        doc_id: str,
        source_path: str,
    ) -> List[Dict[str, Any]]:
        if chunk_ids is not None and len(chunk_ids) != len(chunks):
            raise ValueError("chunk_ids must match chunks length")

        timestamp = datetime.now(timezone.utc).isoformat()
        rows: List[Dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            text = str(getattr(chunk, "text", "") or "")
            if not text.strip():
                continue

            chunk_id = (
                str(chunk_ids[index])
                if chunk_ids is not None
                else str(getattr(chunk, "id", ""))
            )
            if not chunk_id:
                raise ValueError(f"Chunk at index {index} has no ID")

            metadata = dict(getattr(chunk, "metadata", {}) or {})
            metadata["chunk_id"] = chunk_id
            metadata["doc_id"] = doc_id
            if source_path:
                metadata["source_path"] = source_path

            doc_length = 0
            if index < len(term_stats):
                doc_length = int(term_stats[index].get("doc_length", 0) or 0)

            rows.append(
                {
                    "collection": collection,
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "source_path": source_path,
                    "text": text,
                    "doc_length": doc_length,
                    "metadata_json": json.dumps(
                        metadata,
                        ensure_ascii=False,
                        default=str,
                    ),
                    "updated_at": timestamp,
                }
            )
        return rows

    def _build_match_query(self, query_terms: List[str]) -> str:
        terms = [self._quote_term(term) for term in query_terms if str(term).strip()]
        terms = [term for term in terms if term]
        if not terms:
            return ""
        operator = " AND " if self.match_mode == "and" else " OR "
        return operator.join(terms)

    @staticmethod
    def _quote_term(term: Any) -> str:
        text = str(term).strip()
        if not text:
            return ""
        return '"' + text.replace('"', '""') + '"'

    @staticmethod
    def _load_metadata(raw: Any) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            loaded = json.loads(str(raw))
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    @staticmethod
    def _normalise_score(raw_score: float) -> float:
        if raw_score < 0:
            return max(0.0, -raw_score)
        return 1.0 / (1.0 + raw_score)

    @staticmethod
    def _infer_doc_id(chunks: Sequence[Any], source_path: Optional[str]) -> str:
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
            for key in ("doc_id", "doc_hash", "source_hash", "file_hash"):
                if metadata.get(key):
                    return str(metadata[key])
        if source_path:
            return str(source_path)
        first = chunks[0]
        return str(getattr(first, "source_ref", None) or getattr(first, "id", "unknown"))

    @staticmethod
    def _infer_source_path(chunks: Sequence[Any]) -> Optional[str]:
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
            if metadata.get("source_path"):
                return str(metadata["source_path"])
        return None

    @staticmethod
    def _normalise_tokenizer(tokenizer: str) -> str:
        normalized = str(tokenizer or "unicode61").strip().lower()
        if normalized in {"unicode61", "trigram"}:
            return normalized
        raise ValueError(
            "Unsupported FTS5 tokenizer. Expected 'unicode61' or 'trigram', "
            f"got {tokenizer!r}"
        )

    @staticmethod
    def _normalise_match_mode(match_mode: str) -> str:
        normalized = str(match_mode or "or").strip().lower()
        if normalized not in {"or", "and"}:
            return "or"
        return normalized

    def _tokenizer_sql(self) -> str:
        return f"'{self.tokenizer}'"
