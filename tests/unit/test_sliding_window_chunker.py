from types import SimpleNamespace

from src.core.types import Document
from src.ingestion.chunking.sliding_window_chunker import SlidingWindowChunker


def _settings(window_size=20, step_size=10):
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            chunk_size=window_size,
            chunk_overlap=0,
            splitter="recursive",
            chunking=SimpleNamespace(
                strategy="sliding_window",
                chunk_size=window_size,
                chunk_overlap=0,
                recursive={},
                markdown_header={},
                semantic={},
                parent_child={},
                sliding_window={
                    "window_size": window_size,
                    "step_size": step_size,
                    "preserve_sentence_boundary": False,
                },
            ),
        )
    )


def test_short_text_returns_single_chunk():
    document = Document(id="doc1", text="short text", metadata={"source_path": "log.txt"})

    drafts = SlidingWindowChunker(_settings(window_size=50)).split(document)

    assert len(drafts) == 1
    assert drafts[0].char_start == 0
    assert drafts[0].char_end == len("short text")


def test_window_and_step_size_apply():
    text = "abcdefghijklmnopqrstuvwxyz"
    document = Document(id="doc1", text=text, metadata={"source_path": "log.txt"})

    drafts = SlidingWindowChunker(_settings(window_size=10, step_size=5)).split(document)

    assert [d.text for d in drafts[:3]] == ["abcdefghij", "fghijklmno", "klmnopqrst"]
    assert [d.char_start for d in drafts[:3]] == [0, 5, 10]
    assert [d.char_end for d in drafts[:3]] == [10, 15, 20]
    assert len(drafts) < 10
