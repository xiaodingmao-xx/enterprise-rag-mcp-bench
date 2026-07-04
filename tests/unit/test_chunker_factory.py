from types import SimpleNamespace

import pytest

from src.ingestion.chunking.chunker_factory import ChunkerFactory
from src.ingestion.chunking.markdown_header_chunker import MarkdownHeaderChunker
from src.ingestion.chunking.parent_child_chunker import ParentChildChunker
from src.ingestion.chunking.recursive_chunker import RecursiveChunker
from src.ingestion.chunking.semantic_chunker import SemanticChunker
from src.ingestion.chunking.sliding_window_chunker import SlidingWindowChunker


def _settings(strategy: str):
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            chunk_size=80,
            chunk_overlap=10,
            splitter="recursive",
            chunking=SimpleNamespace(
                strategy=strategy,
                chunk_size=80,
                chunk_overlap=10,
                recursive={},
                markdown_header={},
                semantic={},
                parent_child={},
                sliding_window={},
            ),
        )
    )


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("recursive", RecursiveChunker),
        ("markdown_header", MarkdownHeaderChunker),
        ("semantic", SemanticChunker),
        ("parent_child", ParentChildChunker),
        ("sliding_window", SlidingWindowChunker),
        ("MARKDOWN_HEADER", MarkdownHeaderChunker),
    ],
)
def test_chunker_factory_returns_strategy(strategy, expected):
    assert isinstance(ChunkerFactory.create(_settings(strategy)), expected)


def test_chunker_factory_unknown_strategy():
    with pytest.raises(ValueError, match="Unsupported chunking strategy"):
        ChunkerFactory.create(_settings("unknown"))
