"""Factory for sparse retrieval index backends."""

from __future__ import annotations

from typing import Any, Optional

from src.core.settings import resolve_path


def get_sparse_backend(settings: Any) -> str:
    retrieval = getattr(settings, "retrieval", None)
    backend = getattr(retrieval, "sparse_backend", "json_bm25")
    return str(backend or "json_bm25").strip().lower()


def create_sparse_indexer(
    settings: Any,
    *,
    collection: str = "default",
    index_dir: Optional[str] = None,
) -> Any:
    """Create the configured sparse index backend."""
    backend = get_sparse_backend(settings)
    if backend in {"sqlite_fts5", "fts5"}:
        from src.ingestion.storage.sqlite_fts5_indexer import SQLiteFTS5Indexer

        fts5 = getattr(getattr(settings, "retrieval", None), "fts5", None)
        db_path = getattr(fts5, "db_path", "./data/db/sparse_fts5.db")
        tokenizer = getattr(fts5, "tokenizer", "unicode61")
        match_mode = getattr(fts5, "match_mode", "or")
        busy_timeout_ms = getattr(fts5, "busy_timeout_ms", 10000)
        max_retries = getattr(fts5, "max_retries", 3)
        return SQLiteFTS5Indexer(
            db_path=str(resolve_path(db_path)),
            tokenizer=tokenizer,
            match_mode=match_mode,
            busy_timeout_ms=busy_timeout_ms,
            max_retries=max_retries,
        )

    if backend not in {"json_bm25", "bm25", "json"}:
        raise ValueError(
            "Unsupported sparse backend: "
            f"{backend!r}. Expected 'json_bm25' or 'sqlite_fts5'."
        )

    from src.ingestion.storage.bm25_indexer import BM25Indexer

    resolved_index_dir = index_dir or str(resolve_path(f"data/db/bm25/{collection}"))
    return BM25Indexer(index_dir=resolved_index_dir)
