"""Small in-process LRU + TTL cache for query results."""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Generic, Hashable, Optional, TypeVar


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    evictions: int
    size: int
    max_size: int


class LruTtlCache(Generic[K, V]):
    """Thread-safe LRU cache with per-entry TTL.

    Values are deep-copied on get/set so callers can safely mutate returned
    response metadata without changing the cached copy.
    """

    def __init__(self, max_size: int = 128, ttl_seconds: int = 300) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")

        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[K, tuple[float, V]] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: K) -> Optional[V]:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None

            expires_at, value = entry
            if expires_at <= now:
                del self._entries[key]
                self._misses += 1
                return None

            self._entries.move_to_end(key)
            self._hits += 1
            return copy.deepcopy(value)

    def set(self, key: K, value: V) -> None:
        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            self._entries[key] = (expires_at, copy.deepcopy(value))
            self._entries.move_to_end(key)

            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                size=len(self._entries),
                max_size=self.max_size,
            )
