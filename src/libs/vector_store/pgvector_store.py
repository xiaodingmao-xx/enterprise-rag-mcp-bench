"""PgVector provider contract without importing psycopg/pgvector."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.libs.vector_store.base_vector_store import BaseVectorStore


class PgVectorStore(BaseVectorStore):
    def __init__(self, settings: Any, **kwargs: Any) -> None:
        config = getattr(settings, "vector_store", settings)
        block = config.get("pgvector", {}) if isinstance(config, dict) else getattr(config, "pgvector", {})
        self.dsn = kwargs.get("dsn") or (block.get("dsn") if isinstance(block, dict) else None)
        if not self.dsn:
            raise ValueError("PgVector provider requires vector_store.pgvector.dsn")

    def upsert(self, records: List[Dict[str, Any]], trace: Optional[Any] = None, **kwargs: Any) -> None:
        raise NotImplementedError("PgVector backend is reserved but not fully implemented; use Chroma local mode.")

    def query(self, vector: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None, trace: Optional[Any] = None, **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError("PgVector backend is reserved but not fully implemented; use Chroma local mode.")
