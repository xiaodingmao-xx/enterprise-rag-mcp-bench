"""Markdown table preserving chunking strategy."""

from __future__ import annotations

import hashlib
from typing import Any, List

from src.core.types import Document
from src.ingestion.chunking.chunk_schema import ChunkDraft
from src.ingestion.chunking.config import get_chunk_size, get_chunking_settings, get_section
from src.ingestion.chunking.recursive_chunker import RecursiveChunker


class TableAwareChunker:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        config = get_section(get_chunking_settings(settings), "table_aware")
        self.max_rows = max(2, int(config.get("max_table_rows_per_chunk", 30)))
        self.recursive = RecursiveChunker(settings)
        self.chunk_size = get_chunk_size(settings)

    def split(self, document: Document) -> List[ChunkDraft]:
        lines = document.text.splitlines(keepends=True)
        output: List[ChunkDraft] = []
        normal: List[str] = []
        normal_start = 0
        index = 0
        while index < len(lines):
            if "|" not in lines[index]:
                normal.append(lines[index])
                index += 1
                continue
            table_start = index
            table_lines = []
            while index < len(lines) and ("|" in lines[index] or not lines[index].strip()):
                table_lines.append(lines[index])
                index += 1
            table_text = "".join(table_lines).strip()
            if table_text.count("|") < 4:
                normal.extend(table_lines)
                continue
            if normal:
                output.extend(self.recursive.split_text("".join(normal), base_metadata=document.metadata, base_offset=normal_start))
                normal = []
            rows = [line for line in table_text.splitlines() if line.strip()]
            table_id = f"table_{hashlib.sha256(table_text.encode('utf-8')).hexdigest()[:12]}"
            header = rows[:2]
            for offset in range(2, len(rows), self.max_rows - 1):
                body = rows[offset : offset + self.max_rows - 1]
                text = "\n".join(header + body)
                output.append(
                    ChunkDraft(
                        text=text,
                        metadata={**document.metadata, "table_ids": [table_id]},
                        char_start=sum(len(item) for item in lines[:table_start]),
                        char_end=sum(len(item) for item in lines[:index]),
                    )
                )
            normal_start = sum(len(item) for item in lines[:index])
        if normal:
            output.extend(self.recursive.split_text("".join(normal), base_metadata=document.metadata, base_offset=normal_start))
        return output or self.recursive.split(document)
