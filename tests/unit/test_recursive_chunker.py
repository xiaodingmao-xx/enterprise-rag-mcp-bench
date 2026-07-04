from types import SimpleNamespace

from src.core.types import Document
from src.ingestion.chunking.recursive_chunker import RecursiveChunker


def _settings(chunk_size=40, chunk_overlap=10):
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            splitter="recursive",
            chunking=SimpleNamespace(
                strategy="recursive",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                recursive={"separators": [" ", ""]},
                markdown_header={},
                semantic={},
                parent_child={},
                sliding_window={},
            ),
        )
    )


def test_recursive_chunker_splits_long_text_and_offsets():
    text = " ".join(f"word{i}" for i in range(40))
    document = Document(
        id="doc1",
        text=text,
        metadata={"source_path": "test.txt", "heading_path": ["Root"]},
    )

    drafts = RecursiveChunker(_settings()).split(document)

    assert len(drafts) > 1
    assert all(d.char_start is not None and d.char_end is not None for d in drafts)
    assert drafts[0].char_start == 0
    assert drafts[0].heading_path == ["Root"]
    assert drafts[1].char_start < drafts[0].char_end


def test_recursive_chunker_preserves_metadata_page_range():
    document = Document(
        id="doc1",
        text="Short text",
        metadata={"source_path": "test.txt", "page_range": {"start": 3, "end": 4}},
    )

    drafts = RecursiveChunker(_settings()).split(document)

    assert drafts[0].page_range == (3, 4)
