"""OpenSearch provider contract without importing opensearch-py."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.libs.vector_store.base_vector_store import BaseVectorStore


class OpenSearchStore(BaseVectorStore):
    def __init__(self, settings: Any, **kwargs: Any) -> None:
        config = getattr(settings, "vector_store", settings)
        block = config.get("opensearch", {}) if isinstance(config, dict) else getattr(config, "opensearch", {})
        self.hosts = kwargs.get("hosts") or (block.get("hosts") if isinstance(block, dict) else None)
        if not self.hosts:
            raise ValueError("OpenSearch provider requires vector_store.opensearch.hosts")

    def upsert(self, records: List[Dict[str, Any]], trace: Optional[Any] = None, **kwargs: Any) -> None:
        raise NotImplementedError("OpenSearch backend is reserved but not fully implemented; use Chroma local mode.")

    def query(self, vector: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None, trace: Optional[Any] = None, **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError("OpenSearch backend is reserved but not fully implemented; use Chroma local mode.")
