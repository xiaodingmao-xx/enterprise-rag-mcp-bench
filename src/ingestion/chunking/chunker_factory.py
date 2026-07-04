"""Factory for concrete chunking strategies."""

from __future__ import annotations

from typing import Any, Dict, Type

from src.ingestion.chunking.config import get_strategy
from src.ingestion.chunking.markdown_header_chunker import MarkdownHeaderChunker
from src.ingestion.chunking.parent_child_chunker import ParentChildChunker
from src.ingestion.chunking.recursive_chunker import RecursiveChunker
from src.ingestion.chunking.semantic_chunker import SemanticChunker
from src.ingestion.chunking.sliding_window_chunker import SlidingWindowChunker


class ChunkerFactory:
    """Create chunker instances from settings."""

    _CHUNKERS: Dict[str, Type[Any]] = {
        "recursive": RecursiveChunker,
        "markdown_header": MarkdownHeaderChunker,
        "semantic": SemanticChunker,
        "parent_child": ParentChildChunker,
        "sliding_window": SlidingWindowChunker,
    }

    @classmethod
    def create(cls, settings: Any, strategy: str | None = None) -> Any:
        name = (strategy or get_strategy(settings) or "recursive").lower()
        chunker_class = cls._CHUNKERS.get(name)
        if chunker_class is None:
            available = ", ".join(sorted(cls._CHUNKERS))
            raise ValueError(f"Unsupported chunking strategy: '{name}'. Available: {available}")
        return chunker_class(settings)

    @classmethod
    def list_strategies(cls) -> list[str]:
        return sorted(cls._CHUNKERS)
