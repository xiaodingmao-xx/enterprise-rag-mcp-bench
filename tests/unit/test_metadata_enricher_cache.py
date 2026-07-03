"""SQLite cache tests for metadata enrichment."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from src.core.settings import Settings
from src.core.types import Chunk
from src.ingestion.transform.metadata_enricher import MetadataEnricher


def _settings(cache_path: Path, **config) -> Settings:
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.metadata_enrichment = {
        "use_llm": False,
        "cache_enabled": True,
        "cache_path": str(cache_path),
        **config,
    }
    settings.ingestion.metadata_enricher = {}
    return settings


def test_metadata_enrichment_cache_returns_hit_on_same_chunk(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "metadata_cache.sqlite"
    settings = _settings(cache_path)
    chunk = Chunk(
        id="chunk_cache",
        text="# Cached Section\n\nThis chunk should be cached.",
        metadata={"source_path": "cached.md"},
    )

    first = MetadataEnricher(settings).transform([chunk])[0]
    second = MetadataEnricher(settings).transform([chunk])[0]

    assert first.metadata["enrichment_cached"] is False
    assert second.metadata["enrichment_cached"] is True
    assert second.metadata["title"] == first.metadata["title"]
    assert cache_path.exists()


def test_metadata_enrichment_cache_misses_when_text_changes(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "metadata_cache.sqlite"
    settings = _settings(cache_path)
    chunk_v1 = Chunk(
        id="chunk_cache",
        text="# Version One\n\nInitial content.",
        metadata={"source_path": "cached.md"},
    )
    chunk_v2 = Chunk(
        id="chunk_cache",
        text="# Version Two\n\nChanged content.",
        metadata={"source_path": "cached.md"},
    )

    MetadataEnricher(settings).transform([chunk_v1])
    result = MetadataEnricher(settings).transform([chunk_v2])[0]

    assert result.metadata["enrichment_cached"] is False
    assert result.metadata["title"] == "Version Two"


def test_metadata_enrichment_cache_can_be_disabled(tmp_path: Path) -> None:
    cache_path = tmp_path / "disabled_cache.sqlite"
    settings = _settings(cache_path, cache_enabled=False)
    chunk = Chunk(
        id="chunk_no_cache",
        text="# No Cache\n\nCaching is disabled.",
        metadata={"source_path": "no-cache.md"},
    )

    first = MetadataEnricher(settings).transform([chunk])[0]
    second = MetadataEnricher(settings).transform([chunk])[0]

    assert first.metadata["enrichment_cached"] is False
    assert second.metadata["enrichment_cached"] is False
    assert not cache_path.exists()
