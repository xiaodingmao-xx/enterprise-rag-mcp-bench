"""Recursive chunker strategy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.types import Document
from src.ingestion.chunking.chunk_schema import (
    ChunkDraft,
    normalise_page_range,
    normalise_string_list,
)
from src.ingestion.chunking.config import (
    get_chunk_overlap,
    get_chunk_size,
    get_chunking_settings,
    get_section,
)
from src.libs.splitter.recursive_splitter import RecursiveSplitter


class RecursiveChunker:
    """LangChain RecursiveCharacterTextSplitter backed chunker."""

    def __init__(
        self,
        settings: Any,
        *,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ) -> None:
        self.settings = settings
        chunking = get_chunking_settings(settings)
        recursive_config = get_section(chunking, "recursive")
        self.chunk_size = chunk_size if chunk_size is not None else get_chunk_size(settings)
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else get_chunk_overlap(settings)
        )
        self.separators = separators or recursive_config.get("separators")
        self._splitter = RecursiveSplitter(
            settings=settings,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def split(self, document: Document) -> List[ChunkDraft]:
        return self.split_text(
            document.text,
            base_metadata=document.metadata,
            base_offset=0,
        )

    def split_text(
        self,
        text: str,
        *,
        base_metadata: Optional[Dict[str, Any]] = None,
        base_offset: int = 0,
        metadata_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkDraft]:
        fragments = self._splitter.split_text(text)
        metadata = dict(base_metadata or {})
        if metadata_overrides:
            metadata.update(metadata_overrides)

        drafts: List[ChunkDraft] = []
        cursor = 0
        for fragment in fragments:
            if not fragment or not fragment.strip():
                continue
            local_start = self._find_fragment_start(text, fragment, cursor)
            local_end = local_start + len(fragment)
            cursor = max(local_start + 1, local_end - self.chunk_overlap)

            page_range = normalise_page_range(metadata.get("page_range"))
            heading_path = normalise_string_list(metadata.get("heading_path"))
            section_path = normalise_string_list(metadata.get("section_path")) or heading_path
            drafts.append(
                ChunkDraft(
                    text=fragment,
                    metadata=dict(metadata),
                    page_range=page_range,
                    heading_path=heading_path,
                    section_path=section_path,
                    heading=metadata.get("heading"),
                    parent_chunk_id=metadata.get("parent_chunk_id"),
                    char_start=base_offset + local_start,
                    char_end=base_offset + local_end,
                )
            )

        return drafts

    @staticmethod
    def _find_fragment_start(text: str, fragment: str, cursor: int) -> int:
        search_start = max(0, cursor)
        found = text.find(fragment, search_start)
        if found >= 0:
            return found
        found = text.find(fragment, max(0, cursor - len(fragment)))
        if found >= 0:
            return found
        found = text.find(fragment)
        if found >= 0:
            return found
        return min(cursor, max(0, len(text) - len(fragment)))
