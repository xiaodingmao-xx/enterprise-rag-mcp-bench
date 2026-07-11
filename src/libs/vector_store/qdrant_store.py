"""Qdrant provider contract; real client integration is intentionally deferred."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.libs.vector_store.base_vector_store import BaseVectorStore


class QdrantStore(BaseVectorStore):
    def __init__(self, settings: Any, **kwargs: Any) -> None:
        config = getattr(settings, "vector_store", settings)
        block = config.get("qdrant", {}) if isinstance(config, dict) else getattr(config, "qdrant", {})
        self.url = kwargs.get("url") or (block.get("url") if isinstance(block, dict) else None)
        if not self.url:
            raise ValueError("Qdrant provider requires vector_store.qdrant.url")

    def upsert(self, records: List[Dict[str, Any]], trace: Optional[Any] = None, **kwargs: Any) -> None:
        raise NotImplementedError("Qdrant backend is reserved but not fully implemented; use Chroma local mode.")

    def query(self, vector: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None, trace: Optional[Any] = None, **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError("Qdrant backend is reserved but not fully implemented; use Chroma local mode.")
