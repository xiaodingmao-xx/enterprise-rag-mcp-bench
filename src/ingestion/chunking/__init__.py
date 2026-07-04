"""Chunking module - document splitting adapter layer.

This module provides the business adapter for text splitting, transforming
Document objects into Chunk objects with proper metadata and traceability.
"""

from src.ingestion.chunking.chunker_factory import ChunkerFactory
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.chunking.markdown_header_chunker import MarkdownHeaderChunker
from src.ingestion.chunking.parent_child_chunker import ParentChildChunker
from src.ingestion.chunking.recursive_chunker import RecursiveChunker
from src.ingestion.chunking.semantic_chunker import SemanticChunker
from src.ingestion.chunking.sliding_window_chunker import SlidingWindowChunker

__all__ = [
    "ChunkerFactory",
    "DocumentChunker",
    "MarkdownHeaderChunker",
    "ParentChildChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SlidingWindowChunker",
]
