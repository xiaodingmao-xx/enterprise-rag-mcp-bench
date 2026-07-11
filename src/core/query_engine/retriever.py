"""Compatibility protocol for dense and sparse retrievers."""

from __future__ import annotations

from typing import Any, Protocol

from src.core.query_engine.retrieval_filter import RetrievalFilter


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 10, filters: RetrievalFilter | dict[str, Any] | None = None, **kwargs: Any) -> list[Any]:
        ...
