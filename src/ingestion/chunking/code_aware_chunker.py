"""Fenced-code preserving chunker with recursive fallback."""

from __future__ import annotations

import re
from typing import Any, List

from src.core.types import Document
from src.ingestion.chunking.chunk_schema import ChunkDraft
from src.ingestion.chunking.recursive_chunker import RecursiveChunker
from src.ingestion.chunking.config import get_chunking_settings, get_section


class CodeAwareChunker:
    def __init__(self, settings: Any) -> None:
        config = get_section(get_chunking_settings(settings), "code_aware")
        self.max_lines = max(1, int(config.get("max_code_lines_per_chunk", 120)))
        self.recursive = RecursiveChunker(settings)

    def split(self, document: Document) -> List[ChunkDraft]:
        parts = re.split(r"(```[\s\S]*?```)", document.text)
        drafts: List[ChunkDraft] = []
        cursor = 0
        for part in parts:
            if not part.strip():
                cursor += len(part)
                continue
            if part.startswith("```"):
                lines = part.splitlines()
                for start in range(0, len(lines), self.max_lines):
                    text = "\n".join(lines[start : start + self.max_lines])
                    drafts.append(
                        ChunkDraft(
                            text=text,
                            metadata={**document.metadata, "source_type": "code"},
                            char_start=cursor,
                            char_end=cursor + len(part),
                        )
                    )
            else:
                drafts.extend(self.recursive.split_text(part, base_metadata=document.metadata, base_offset=cursor))
            cursor += len(part)
        return drafts or self.recursive.split(document)
