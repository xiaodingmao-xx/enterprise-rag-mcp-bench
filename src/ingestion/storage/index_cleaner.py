"""Pluggable cleanup boundary for versioned index and cache data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IndexCleaner(ABC):
    @abstractmethod
    def delete_vectors(self, document_id: str, version_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_sparse_index(self, document_id: str, version_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_images(self, document_id: str, version_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_cache(self, document_id: str, version_id: str) -> None:
        raise NotImplementedError


class NoopIndexCleaner(IndexCleaner):
    """Safe default when a backend has no version-aware delete API yet."""

    def delete_vectors(self, document_id: str, version_id: str) -> None:
        return None

    def delete_sparse_index(self, document_id: str, version_id: str) -> None:
        return None

    def delete_images(self, document_id: str, version_id: str) -> None:
        return None

    def delete_cache(self, document_id: str, version_id: str) -> None:
        return None


class LocalIndexCleaner(IndexCleaner):
    """Best-effort adapter over the project's current local stores."""

    def __init__(self, vector_store: Any = None, sparse_index: Any = None, image_storage: Any = None, cache: Any = None) -> None:
        self.vector_store = vector_store
        self.sparse_index = sparse_index
        self.image_storage = image_storage
        self.cache = cache

    def delete_vectors(self, document_id: str, version_id: str) -> None:
        method = getattr(self.vector_store, "delete_by_metadata", None)
        if callable(method):
            method({"document_id": document_id, "version_id": version_id})

    def delete_sparse_index(self, document_id: str, version_id: str) -> None:
        method = getattr(self.sparse_index, "remove_document", None)
        if callable(method):
            method(version_id)

    def delete_images(self, document_id: str, version_id: str) -> None:
        method = getattr(self.image_storage, "delete_by_document", None)
        if callable(method):
            method(document_id, version_id)

    def delete_cache(self, document_id: str, version_id: str) -> None:
        method = getattr(self.cache, "delete_version", None)
        if callable(method):
            method(document_id, version_id)

