"""Rule-based structured metadata enrichment tests."""

from __future__ import annotations

from unittest.mock import Mock

from src.core.settings import Settings
from src.core.types import Chunk
from src.ingestion.transform.metadata_enricher import MetadataEnricher


def _settings(**config) -> Settings:
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.metadata_enrichment = {
        "use_llm": False,
        "cache_enabled": False,
        "generate_questions": True,
        "extract_entities": True,
        **config,
    }
    settings.ingestion.metadata_enricher = {}
    return settings


def test_rule_based_extracts_heading_page_media_and_entities() -> None:
    chunk = Chunk(
        id="chunk_rule",
        text=(
            "# Admin Guide\n\n"
            "## Retry Policy\n\n"
            "The PaymentServiceClient() handles HTTP 503 for API v2.4.1. "
            "Set retry_count=3 before calling /api/v1/payments. "
            "[TABLE: table_retry] [IMAGE: image_sequence]"
        ),
        metadata={
            "source_path": "docs/admin.md",
            "page": 7,
            "doc_type": "markdown",
        },
        source_ref="admin.md#retry",
    )
    enricher = MetadataEnricher(_settings())

    result = enricher.transform([chunk])[0]
    metadata = result.metadata

    assert metadata["title"] == "Retry Policy"
    assert metadata["heading_path"] == ["Admin Guide", "Retry Policy"]
    assert metadata["section_path"] == ["Admin Guide", "Retry Policy"]
    assert metadata["page_range"] == {"start": 7, "end": 7}
    assert metadata["table_ids"] == ["table_retry"]
    assert metadata["image_ids"] == ["image_sequence"]
    assert "HTTP 503" in metadata["entities"]
    assert "PaymentServiceClient" in metadata["entities"]
    assert metadata["questions"]
    assert metadata["enriched_by"] == "rule"


def test_rule_based_respects_question_and_entity_switches() -> None:
    chunk = Chunk(
        id="chunk_switch",
        text="API Gateway v1.0 returns HTTP 404 when route mapping is missing.",
        metadata={"source_path": "docs/gateway.txt"},
    )
    enricher = MetadataEnricher(
        _settings(generate_questions=False, extract_entities=False)
    )

    result = enricher.transform([chunk])[0]

    assert result.metadata["entities"] == []
    assert result.metadata["questions"] == []


def test_rule_based_fallback_handles_bad_chunk_atomically() -> None:
    chunks = [
        Chunk(id="ok", text="Valid text.", metadata={"source_path": "ok.txt"}),
        Chunk(id="bad", text=None, metadata={"source_path": "bad.txt"}),
    ]
    enricher = MetadataEnricher(_settings())

    result = enricher.transform(chunks)

    assert result[0].metadata["enriched_by"] == "rule"
    assert result[1].metadata["enriched_by"] == "error"
    assert result[1].metadata["title"] == "Untitled"
    assert "enrich_error" in result[1].metadata
