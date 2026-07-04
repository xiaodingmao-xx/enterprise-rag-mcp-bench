"""Chunking configuration compatibility helpers."""

from __future__ import annotations

from typing import Any, Dict


def get_chunking_settings(settings: Any) -> Any:
    ingestion = getattr(settings, "ingestion", None)
    if isinstance(ingestion, dict):
        return ingestion.get("chunking") or {
            "strategy": ingestion.get("splitter", "recursive"),
            "chunk_size": ingestion.get("chunk_size", 1000),
            "chunk_overlap": ingestion.get("chunk_overlap", 200),
        }
    if ingestion is not None:
        chunking = getattr(ingestion, "chunking", None)
        if chunking is not None:
            return chunking
        return {
            "strategy": getattr(ingestion, "splitter", "recursive"),
            "chunk_size": getattr(ingestion, "chunk_size", 1000),
            "chunk_overlap": getattr(ingestion, "chunk_overlap", 200),
        }
    return {"strategy": "recursive", "chunk_size": 1000, "chunk_overlap": 200}


def get_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def get_section(config: Any, name: str) -> Dict[str, Any]:
    value = get_value(config, name, {})
    return value if isinstance(value, dict) else {}


def get_strategy(settings: Any) -> str:
    chunking = get_chunking_settings(settings)
    return str(get_value(chunking, "strategy", "recursive") or "recursive").lower()


def get_chunk_size(settings: Any, default: int = 1000) -> int:
    chunking = get_chunking_settings(settings)
    return _as_int(get_value(chunking, "chunk_size", default), default, minimum=1)


def get_chunk_overlap(settings: Any, default: int = 200) -> int:
    chunking = get_chunking_settings(settings)
    return _as_int(get_value(chunking, "chunk_overlap", default), default, minimum=0)


def _as_int(value: Any, default: int, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)
