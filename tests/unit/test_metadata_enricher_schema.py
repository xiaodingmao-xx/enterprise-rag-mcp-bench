"""Schema tests for structured metadata enrichment."""

from __future__ import annotations

from unittest.mock import Mock

from src.core.settings import Settings
from src.core.types import Chunk
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.ingestion.transform.metadata_schema import (
    extract_json_object,
    validate_enrichment_metadata,
)


def _settings(config: dict | None = None, legacy: dict | None = None) -> Settings:
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.metadata_enrichment = {
        "use_llm": False,
        "cache_enabled": False,
        **(config or {}),
    }
    settings.ingestion.metadata_enricher = legacy or {}
    return settings


def test_enriched_metadata_contains_full_structured_schema() -> None:
    chunk = Chunk(
        id="chunk_schema",
        text=(
            "# Installation\n\n"
            "Configure APIClient v1.2.3 through /api/v1/search. "
            "[TABLE: tbl_config] [IMAGE: img_flow]"
        ),
        metadata={
            "source_path": "docs/setup.md",
            "page_start": 2,
            "page_end": 3,
        },
        source_ref="setup.md",
    )
    enricher = MetadataEnricher(_settings())

    result = enricher.transform([chunk])[0]
    metadata = result.metadata

    for key in (
        "title",
        "summary",
        "tags",
        "section_path",
        "heading_path",
        "page_range",
        "table_ids",
        "image_ids",
        "entities",
        "questions",
        "enrichment_method",
        "enrichment_cached",
    ):
        assert key in metadata

    assert metadata["title"] == "Installation"
    assert metadata["page_range"] == {"start": 2, "end": 3}
    assert metadata["table_ids"] == ["tbl_config"]
    assert metadata["image_ids"] == ["img_flow"]
    assert metadata["enrichment_method"] == "rule_based"
    assert metadata["tags_text"]
    assert metadata["heading_path_text"] == "Installation"
    assert metadata["page_range_text"] == "2-3"


def test_metadata_enrichment_config_takes_precedence_over_legacy_config() -> None:
    settings = _settings(
        {"use_llm": False, "cache_enabled": False},
        legacy={"use_llm": True, "cache_enabled": True},
    )

    enricher = MetadataEnricher(settings)

    assert enricher.use_llm is False
    assert enricher.config.cache_enabled is False


def test_json_schema_normalises_missing_and_invalid_fields() -> None:
    metadata = validate_enrichment_metadata(
        {
            "title": "A" * 100,
            "summary": "",
            "tags": "rag, rag, metadata",
            "page_range": {"start": 5, "end": 4},
            "questions": ["What is metadata enrichment?"],
            "unexpected": {"kept": True},
        },
        fallback={"summary": "Fallback summary"},
    )

    assert metadata["title"].endswith("...")
    assert metadata["summary"] == "Fallback summary"
    assert metadata["tags"] == ["rag", "metadata"]
    assert metadata["page_range"] == {"start": 4, "end": 5}
    assert metadata["raw_llm_metadata"]


def test_extract_json_object_supports_fenced_json() -> None:
    parsed = extract_json_object(
        '```json\n{"title": "JSON Title", "tags": ["rag"]}\n```'
    )

    assert parsed == {"title": "JSON Title", "tags": ["rag"]}
