"""Markdown heading aware chunker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional
import re

from src.core.types import Document
from src.ingestion.chunking.chunk_schema import ChunkDraft
from src.ingestion.chunking.config import get_chunk_size, get_chunking_settings, get_section
from src.ingestion.chunking.recursive_chunker import RecursiveChunker


@dataclass
class _Section:
    text: str
    start: int
    end: int
    heading_path: List[str]
    heading: Optional[str]


class MarkdownHeaderChunker:
    """Split Markdown documents by heading hierarchy."""

    DEFAULT_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        chunking = get_chunking_settings(settings)
        config = get_section(chunking, "markdown_header")
        self.headers = config.get("headers") or self.DEFAULT_HEADERS
        self.preserve_headers = bool(config.get("preserve_headers", True))
        self.split_under_headers = bool(config.get("split_under_headers", True))
        self.chunk_size = get_chunk_size(settings)
        self.recursive = RecursiveChunker(settings)

    def split(self, document: Document) -> List[ChunkDraft]:
        sections = self._extract_sections(document.text)
        if not sections:
            drafts = self.recursive.split(document)
            for draft in drafts:
                draft.metadata["fallback_reason"] = "markdown_no_headings"
            return drafts

        drafts: List[ChunkDraft] = []
        for section in sections:
            metadata_overrides = {
                "heading_path": section.heading_path,
                "section_path": section.heading_path,
                "heading": section.heading,
            }
            if self.split_under_headers and len(section.text) > self.chunk_size:
                drafts.extend(
                    self.recursive.split_text(
                        section.text,
                        base_metadata=document.metadata,
                        base_offset=section.start,
                        metadata_overrides=metadata_overrides,
                    )
                )
            else:
                drafts.append(
                    ChunkDraft(
                        text=section.text,
                        metadata={**document.metadata, **metadata_overrides},
                        heading_path=section.heading_path,
                        section_path=section.heading_path,
                        heading=section.heading,
                        char_start=section.start,
                        char_end=section.end,
                    )
                )
        return drafts

    def _extract_sections(self, text: str) -> List[_Section]:
        lines = text.splitlines(keepends=True)
        offsets: List[int] = []
        cursor = 0
        for line in lines:
            offsets.append(cursor)
            cursor += len(line)

        header_levels = {str(header[0]): index + 1 for index, header in enumerate(self.headers)}
        stack: List[str] = []
        sections: List[_Section] = []
        active: Optional[dict[str, Any]] = None
        in_code = False

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue

            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line.rstrip("\r\n"))
            if not match:
                continue

            marks, heading = match.groups()
            if marks not in header_levels:
                continue

            if active is not None:
                sections.extend(self._close_section(text, active, offsets[idx]))

            level = header_levels[marks]
            heading = heading.strip()
            stack = stack[: level - 1]
            stack.append(heading)
            header_start = offsets[idx]
            content_start = header_start if self.preserve_headers else header_start + len(line)
            active = {
                "start": content_start,
                "heading_path": list(stack),
                "heading": heading,
            }

        if active is not None:
            sections.extend(self._close_section(text, active, len(text)))
        return sections

    @staticmethod
    def _close_section(text: str, active: dict[str, Any], end: int) -> List[_Section]:
        start = int(active["start"])
        section_text = text[start:end].strip()
        if not section_text:
            return []
        leading_ws = len(text[start:end]) - len(text[start:end].lstrip())
        trailing_ws = len(text[start:end]) - len(text[start:end].rstrip())
        return [
            _Section(
                text=section_text,
                start=start + leading_ws,
                end=end - trailing_ws,
                heading_path=list(active["heading_path"]),
                heading=active.get("heading"),
            )
        ]
