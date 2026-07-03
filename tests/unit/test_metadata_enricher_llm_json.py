"""LLM JSON metadata enrichment tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

from src.core.settings import Settings
from src.core.types import Chunk
from src.ingestion.transform.metadata_enricher import MetadataEnricher


def _prompt(tmp_path: Path) -> str:
    prompt = tmp_path / "metadata_prompt.txt"
    prompt.write_text("Return JSON for:\n{chunk_text}", encoding="utf-8")
    return str(prompt)


def _settings(prompt_path: str, **config) -> Settings:
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.metadata_enrichment = {
        "use_llm": True,
        "cache_enabled": False,
        "prompt_path": prompt_path,
        "max_concurrency": 2,
        "budget_usd_per_run": 2.0,
        "output_schema": "json",
        **config,
    }
    settings.ingestion.metadata_enricher = {}
    settings.llm = Mock()
    settings.llm.provider = "openai"
    return settings


def test_llm_json_response_enriches_structured_fields(tmp_path: Path) -> None:
    mock_llm = Mock()
    mock_llm.chat.return_value = "```json\n" + json.dumps(
        {
            "title": "LLM Metadata",
            "summary": "LLM generated metadata for retrieval.",
            "tags": ["metadata", "rag"],
            "section_path": ["Docs", "Metadata"],
            "heading_path": ["Metadata"],
            "page_range": {"start": 4, "end": 4},
            "table_ids": ["table_llm"],
            "image_ids": ["image_llm"],
            "entities": ["MetadataEnricher"],
            "questions": ["How does metadata enrichment work?"],
        }
    ) + "\n```"
    chunk = Chunk(
        id="chunk_llm",
        text="# Rule Heading\n\n[TABLE: table_rule] [IMAGE: image_rule]",
        metadata={"source_path": "docs/meta.md"},
    )
    enricher = MetadataEnricher(
        _settings(_prompt(tmp_path)),
        llm=mock_llm,
    )

    result = enricher.transform([chunk])[0]
    metadata = result.metadata

    assert metadata["title"] == "LLM Metadata"
    assert metadata["enriched_by"] == "llm"
    assert metadata["enrichment_method"] == "hybrid"
    assert metadata["page_range"] == {"start": 4, "end": 4}
    assert "table_llm" in metadata["table_ids"]
    assert "table_rule" in metadata["table_ids"]
    assert "image_llm" in metadata["image_ids"]
    assert "image_rule" in metadata["image_ids"]
    assert "MetadataEnricher" in metadata["entities"]
    mock_llm.chat.assert_called_once()


def test_llm_invalid_json_falls_back_to_rule_based(tmp_path: Path) -> None:
    mock_llm = Mock()
    mock_llm.chat.return_value = "not a JSON object"
    chunk = Chunk(
        id="chunk_bad_json",
        text="# Rule Title\n\nRule content.",
        metadata={"source_path": "docs/rule.md"},
    )
    enricher = MetadataEnricher(
        _settings(_prompt(tmp_path)),
        llm=mock_llm,
    )

    result = enricher.transform([chunk])[0]

    assert result.metadata["title"] == "Rule Title"
    assert result.metadata["enriched_by"] == "rule"
    assert result.metadata["enrichment_method"] == "fallback"
    assert result.metadata["enrich_fallback_reason"] == "llm_failed"


def test_llm_budget_limit_skips_call_and_uses_rule_based(tmp_path: Path) -> None:
    mock_llm = Mock()
    chunk = Chunk(
        id="chunk_budget",
        text="# Budget Title\n\nThis chunk should not call the LLM.",
        metadata={"source_path": "docs/budget.md"},
    )
    enricher = MetadataEnricher(
        _settings(_prompt(tmp_path), budget_usd_per_run=0.0),
        llm=mock_llm,
    )

    result = enricher.transform([chunk])[0]

    assert result.metadata["title"] == "Budget Title"
    assert result.metadata["enriched_by"] == "rule"
    assert result.metadata["enrich_fallback_reason"] == "budget_exceeded"
    assert result.metadata["enrichment_budget_exceeded"] is True
    mock_llm.chat.assert_not_called()
