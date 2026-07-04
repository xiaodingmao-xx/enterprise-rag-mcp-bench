"""Sliding window chunker for continuous text."""

from __future__ import annotations

from typing import Any, List
import re

from src.core.types import Document
from src.ingestion.chunking.chunk_schema import ChunkDraft, normalise_page_range, normalise_string_list
from src.ingestion.chunking.config import get_chunk_size, get_chunking_settings, get_section


class SlidingWindowChunker:
    """Split text with a fixed-size sliding window."""

    def __init__(self, settings: Any) -> None:
        chunking = get_chunking_settings(settings)
        config = get_section(chunking, "sliding_window")
        self.window_size = int(config.get("window_size") or get_chunk_size(settings))
        self.step_size = int(config.get("step_size") or max(1, self.window_size // 2))
        self.preserve_sentence_boundary = bool(config.get("preserve_sentence_boundary", True))
        self.window_size = max(1, self.window_size)
        self.step_size = max(1, self.step_size)

    def split(self, document: Document) -> List[ChunkDraft]:
        text = document.text
        metadata = document.metadata or {}
        if len(text) <= self.window_size:
            return [
                ChunkDraft(
                    text=text,
                    metadata=dict(metadata),
                    page_range=normalise_page_range(metadata.get("page_range")),
                    heading_path=normalise_string_list(metadata.get("heading_path")),
                    section_path=normalise_string_list(metadata.get("section_path")),
                    char_start=0,
                    char_end=len(text),
                )
            ]

        drafts: List[ChunkDraft] = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + self.window_size, text_len)
            if self.preserve_sentence_boundary and end < text_len:
                end = self._align_end(text, start, end)
            chunk_text = text[start:end].strip()
            if chunk_text:
                leading_ws = len(text[start:end]) - len(text[start:end].lstrip())
                trailing_ws = len(text[start:end]) - len(text[start:end].rstrip())
                drafts.append(
                    ChunkDraft(
                        text=chunk_text,
                        metadata=dict(metadata),
                        page_range=normalise_page_range(metadata.get("page_range")),
                        heading_path=normalise_string_list(metadata.get("heading_path")),
                        section_path=normalise_string_list(metadata.get("section_path")),
                        char_start=start + leading_ws,
                        char_end=end - trailing_ws,
                    )
                )
            if end >= text_len:
                break
            next_start = start + self.step_size
            if next_start <= start:
                break
            start = next_start
        return drafts

    @staticmethod
    def _align_end(text: str, start: int, end: int) -> int:
        window = text[start:end]
        matches = list(re.finditer(r"[.!?。！？]\s+", window))
        if not matches:
            return end
        candidate = start + matches[-1].end()
        minimum = start + max(1, int((end - start) * 0.6))
        return candidate if candidate >= minimum else end
