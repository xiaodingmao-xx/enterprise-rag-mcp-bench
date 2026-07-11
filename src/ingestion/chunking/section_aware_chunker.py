"""Section-aware strategy built on the existing Markdown header chunker."""

from __future__ import annotations

from typing import Any, List

from src.core.types import Document
from src.ingestion.chunking.chunk_schema import ChunkDraft
from src.ingestion.chunking.config import get_chunk_size, get_chunking_settings, get_section
from src.ingestion.chunking.markdown_header_chunker import MarkdownHeaderChunker
from src.ingestion.chunking.recursive_chunker import RecursiveChunker


class SectionAwareChunker:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        config = get_section(get_chunking_settings(settings), "section_aware")
        self.max_section_chars = int(config.get("max_section_chars", get_chunk_size(settings) * 3))
        self.base = MarkdownHeaderChunker(settings)
        self.recursive = RecursiveChunker(settings)

    def split(self, document: Document) -> List[ChunkDraft]:
        drafts = self.base.split(document)
        output: List[ChunkDraft] = []
        for draft in drafts:
            if len(draft.text) <= self.max_section_chars:
                output.append(draft)
                continue
            output.extend(
                self.recursive.split_text(
                    draft.text,
                    base_metadata=draft.metadata,
                    base_offset=draft.char_start or 0,
                    metadata_overrides={
                        "heading_path": draft.heading_path,
                        "section_path": draft.section_path,
                        "heading": draft.heading,
                    },
                )
            )
        return output
