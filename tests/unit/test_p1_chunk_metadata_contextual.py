import time
from types import SimpleNamespace

import pytest

from src.core.types import Chunk, Document
from src.ingestion.chunking.chunk_metadata import normalize_chunk_metadata, to_vector_metadata
from src.ingestion.chunking.chunk_quality import evaluate_chunk_quality
from src.ingestion.chunking.chunker_factory import ChunkerFactory
from src.ingestion.chunking.contextualizer import LLMContextualizer, RuleBasedContextualizer
from src.ingestion.chunking.metadata_validator import MetadataValidationError, validate_chunk_metadata
from src.ingestion.chunking.parent_child_retrieval import get_parent_context
from src.ingestion.chunking.table_aware_chunker import TableAwareChunker
from src.ingestion.chunking.code_aware_chunker import CodeAwareChunker
from src.ingestion.chunking.section_aware_chunker import SectionAwareChunker


def _settings(chunk_size=40):
    chunking = SimpleNamespace(
        strategy="recursive",
        chunk_size=chunk_size,
        chunk_overlap=5,
        section_aware={"max_section_chars": 30},
        table_aware={"max_table_rows_per_chunk": 4},
        code_aware={"max_code_lines_per_chunk": 20},
    )
    return SimpleNamespace(ingestion=SimpleNamespace(chunking=chunking))


def test_metadata_normalization_and_vector_flattening():
    metadata = normalize_chunk_metadata(
        {
            "chunk_id": "c1",
            "source_path": "docs/guide.pdf",
            "doc_id": "doc1",
            "page_range": {"start": 2, "end": 3},
            "heading_path": ["Guide", "Install"],
            "allowed_roles": ["reader"],
            "bbox": [1, 2, 3, 4],
        }
    )
    assert metadata["source_uri"] == "docs/guide.pdf"
    assert metadata["document_id"] == "doc1"
    assert metadata["page_start"] == 2 and metadata["page_end"] == 3
    assert metadata["acl_roles"] == ["reader"]
    vector_metadata = to_vector_metadata(metadata)
    assert vector_metadata["heading_path_text"] == "Guide > Install"
    assert vector_metadata["bbox_text"] == "1.0,2.0,3.0,4.0"


def test_metadata_validation_rejects_malformed_coordinates_and_pages():
    with pytest.raises(MetadataValidationError):
        validate_chunk_metadata({"chunk_id": "c1", "bbox": [1, 2]}, stage="upsert")
    with pytest.raises(MetadataValidationError):
        validate_chunk_metadata({"chunk_id": "c1", "page": "unknown"}, stage="chunk")


def test_chunk_quality_emits_metrics_and_flags():
    quality = evaluate_chunk_quality(
        "# Heading\n\nshort",
        metadata={"heading": "Heading"},
        config={"min_chars": 50},
    )
    assert quality["character_count"] > 0
    assert quality["heading_count"] == 1
    assert "too_short" in quality["flags"]
    assert quality["quality_status"] == "warning"


def test_specialized_chunkers_preserve_tables_and_code():
    table_doc = Document(
        id="table-doc",
        text="# Data\n\n| key | value |\n| --- | --- |\n| a | 1 |\n| b | 2 |",
        metadata={"source_path": "table.md"},
    )
    table_drafts = TableAwareChunker(_settings()).split(table_doc)
    assert table_drafts and any(draft.metadata.get("table_ids") for draft in table_drafts)

    code_doc = Document(
        id="code-doc",
        text="Before\n\n```python\nprint('ok')\n```",
        metadata={"source_path": "code.md"},
    )
    code_drafts = CodeAwareChunker(_settings()).split(code_doc)
    assert any(draft.metadata.get("source_type") == "code" for draft in code_drafts)

    section_doc = Document(
        id="section-doc",
        text="# A\n\n" + ("content " * 20) + "\n\n## B\n\nchild",
        metadata={"source_path": "section.md"},
    )
    section_drafts = SectionAwareChunker(_settings()).split(section_doc)
    assert section_drafts
    assert "A" in section_drafts[0].heading_path
    assert {"section_aware", "table_aware", "code_aware"}.issubset(set(ChunkerFactory.list_strategies()))


def test_contextualizer_fallback_budget_cache_and_parent_non_mutation():
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return "contextual answer"

    contextualizer = LLMContextualizer(llm=llm, token_budget=2, cache_enabled=True)
    assert contextualizer.add_context("doc", "section", "chunk") == "contextual answer"
    assert contextualizer.add_context("doc", "section", "chunk") == "contextual answer"
    assert len(calls) == 1
    assert len(calls[0]) <= 8

    def slow(_prompt):
        time.sleep(0.05)
        return "late"

    fallback = RuleBasedContextualizer(max_context_chars=20)
    timed_out = LLMContextualizer(llm=slow, fallback=fallback, timeout_seconds=0.001)
    assert "chunk" in timed_out.add_context("doc", "section", "chunk")
    assert timed_out.last_fallback is True

    parent = Chunk(id="parent", text="parent text that is longer", metadata={"source_path": "x"})
    child = Chunk(
        id="child",
        text="child",
        metadata={"source_path": "x", "parent_chunk_id": "parent"},
        parent_chunk_id="parent",
    )
    context = get_parent_context(child, [parent], max_chars=6)
    assert context.text == "parent"
    assert parent.text == "parent text that is longer"
