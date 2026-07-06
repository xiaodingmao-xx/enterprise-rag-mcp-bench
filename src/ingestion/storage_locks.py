"""In-process locks for ingestion storage critical sections."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

_registry_lock = threading.Lock()
_collection_locks: dict[str, threading.RLock] = {}
_global_lock = threading.RLock()


def get_collection_storage_lock(collection: str) -> threading.RLock:
    """Return the storage lock for a collection."""
    key = str(collection or "default")
    with _registry_lock:
        lock = _collection_locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _collection_locks[key] = lock
        return lock


def get_global_storage_lock() -> threading.RLock:
    """Return the process-wide storage lock."""
    return _global_lock


def get_storage_lock(collection: str, scope: str = "collection") -> threading.RLock:
    """Return a storage lock using either collection or global scope."""
    if str(scope or "collection").lower() == "global":
        return get_global_storage_lock()
    return get_collection_storage_lock(collection)


@contextmanager
def collection_storage_lock(collection: str) -> Iterator[None]:
    """Serialize writes to shared storage for one collection."""
    lock = get_collection_storage_lock(collection)
    with lock:
        yield


@contextmanager
def global_storage_lock() -> Iterator[None]:
    """Serialize writes across all collections."""
    with _global_lock:
        yield
