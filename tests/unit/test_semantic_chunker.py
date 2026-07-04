from types import SimpleNamespace

from src.core.types import Document
from src.ingestion.chunking.semantic_chunker import SemanticChunker


def _settings(enabled=False):
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            chunk_size=50,
            chunk_overlap=5,
            splitter="recursive",
            chunking=SimpleNamespace(
                strategy="semantic",
                chunk_size=50,
                chunk_overlap=5,
                recursive={"separators": [" ", ""]},
                markdown_header={},
                semantic={
                    "enabled": enabled,
                    "min_chunk_size": 20,
                    "max_chunk_size": 60,
                },
                parent_child={},
                sliding_window={},
            ),
        )
    )


def test_semantic_disabled_falls_back_to_recursive():
    document = Document(
        id="doc1",
        text="Sentence one. Sentence two. Sentence three. " * 5,
        metadata={"source_path": "doc.txt"},
    )

    drafts = SemanticChunker(_settings(enabled=False)).split(document)

    assert drafts
    assert all(d.metadata["fallback_reason"] == "semantic_disabled" for d in drafts)


def test_semantic_enabled_uses_sentence_packing():
    document = Document(
        id="doc1",
        text="Sentence one. Sentence two. Sentence three. Sentence four.",
        metadata={"source_path": "doc.txt"},
    )

    drafts = SemanticChunker(_settings(enabled=True)).split(document)

    assert drafts
    assert drafts[0].metadata["semantic_strategy"] == "sentence_pack"
