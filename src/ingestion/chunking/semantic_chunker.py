"""Semantic-style chunker with safe recursive fallback."""

from __future__ import annotations

from typing import Any, List, Tuple
import re

from src.core.types import Document
from src.ingestion.chunking.chunk_schema import ChunkDraft, normalise_page_range, normalise_string_list
from src.ingestion.chunking.config import get_chunk_size, get_chunking_settings, get_section
from src.ingestion.chunking.recursive_chunker import RecursiveChunker


class SemanticChunker:
    """Sentence-boundary semantic chunker.

    This implementation intentionally avoids LLM calls. When semantic chunking
    is disabled or cannot run, it falls back to ``RecursiveChunker`` and records
    ``fallback_reason`` in chunk metadata.
    """

    def __init__(self, settings: Any) -> None:
        chunking = get_chunking_settings(settings)
        config = get_section(chunking, "semantic")
        self.enabled = bool(config.get("enabled", False))
        self.min_chunk_size = int(config.get("min_chunk_size") or 300)
        self.max_chunk_size = int(config.get("max_chunk_size") or get_chunk_size(settings, 1200))
        self.min_chunk_size = max(1, self.min_chunk_size)
        self.max_chunk_size = max(self.min_chunk_size, self.max_chunk_size)
        self.recursive = RecursiveChunker(settings)

    def split(self, document: Document) -> List[ChunkDraft]:
        if not self.enabled:
            return self._fallback(document, "semantic_disabled")
        try:
            sentences = self._sentences_with_offsets(document.text)
            if not sentences:
                return self._fallback(document, "semantic_no_sentences")
            drafts = self._pack_sentences(document, sentences)
            return drafts or self._fallback(document, "semantic_empty_result")
        except Exception as exc:
            return self._fallback(document, f"semantic_failed: {exc}")

    def _fallback(self, document: Document, reason: str) -> List[ChunkDraft]:
        drafts = self.recursive.split(document)
        for draft in drafts:
            draft.metadata["fallback_reason"] = reason
            draft.metadata["semantic_fallback"] = True
        return drafts

    def _pack_sentences(
        self,
        document: Document,
        sentences: List[Tuple[str, int, int]],
    ) -> List[ChunkDraft]:
        metadata = document.metadata or {}
        drafts: List[ChunkDraft] = []
        buffer: List[Tuple[str, int, int]] = []

        def flush() -> None:
            if not buffer:
                return
            chunk_text = " ".join(item[0].strip() for item in buffer if item[0].strip()).strip()
            if not chunk_text:
                buffer.clear()
                return
            drafts.append(
                ChunkDraft(
                    text=chunk_text,
                    metadata={**metadata, "semantic_strategy": "sentence_pack"},
                    page_range=normalise_page_range(metadata.get("page_range")),
                    heading_path=normalise_string_list(metadata.get("heading_path")),
                    section_path=normalise_string_list(metadata.get("section_path")),
                    char_start=buffer[0][1],
                    char_end=buffer[-1][2],
                )
            )
            buffer.clear()

        current_len = 0
        for sentence in sentences:
            sentence_len = len(sentence[0])
            if buffer and current_len + sentence_len > self.max_chunk_size:
                flush()
                current_len = 0
            buffer.append(sentence)
            current_len += sentence_len + 1
            if current_len >= self.min_chunk_size:
                next_is_large = current_len >= self.max_chunk_size
                if next_is_large:
                    flush()
                    current_len = 0
        flush()
        return drafts

    @staticmethod
    def _sentences_with_offsets(text: str) -> List[Tuple[str, int, int]]:
        pattern = re.compile(r".+?(?:[.!?。！？](?:\s+|$)|$)", re.DOTALL)
        output: List[Tuple[str, int, int]] = []
        for match in pattern.finditer(text or ""):
            value = match.group(0).strip()
            if not value:
                continue
            leading_ws = len(match.group(0)) - len(match.group(0).lstrip())
            trailing_ws = len(match.group(0)) - len(match.group(0).rstrip())
            output.append((value, match.start() + leading_ws, match.end() - trailing_ws))
        return output
