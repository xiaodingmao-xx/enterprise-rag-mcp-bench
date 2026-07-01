"""Unit tests for LRU + TTL query cache."""

import time

from src.core.query_engine.query_cache import LruTtlCache


def test_cache_returns_deep_copy() -> None:
    cache = LruTtlCache[str, dict](max_size=2, ttl_seconds=60)
    cache.set("q", {"metadata": {"cache_hit": False}})

    cached = cache.get("q")
    assert cached is not None
    cached["metadata"]["cache_hit"] = True

    cached_again = cache.get("q")
    assert cached_again is not None
    assert cached_again["metadata"]["cache_hit"] is False


def test_cache_expires_entries() -> None:
    cache = LruTtlCache[str, str](max_size=2, ttl_seconds=0.05)
    cache.set("q", "answer")

    assert cache.get("q") == "answer"
    time.sleep(0.06)
    assert cache.get("q") is None


def test_cache_evicts_least_recently_used() -> None:
    cache = LruTtlCache[str, str](max_size=2, ttl_seconds=60)
    cache.set("a", "A")
    cache.set("b", "B")

    assert cache.get("a") == "A"
    cache.set("c", "C")

    assert cache.get("a") == "A"
    assert cache.get("b") is None
    assert cache.get("c") == "C"
    assert cache.stats().evictions == 1
