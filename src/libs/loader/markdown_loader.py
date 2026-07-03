"""Markdown document loader."""

from __future__ import annotations

from pathlib import Path

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader


class MarkdownLoader(BaseLoader):
    """Load Markdown files while preserving the original structure."""

    file_type = "markdown"

    def load(self, file_path: str | Path) -> Document:
        path = self._validate_file(file_path)
        text = path.read_text(encoding="utf-8")
        title = self._extract_title(text) or path.name

        return Document(
            id=self._document_id(path),
            text=text,
            metadata={
                "source_path": str(path),
                "file_name": path.name,
                "file_type": self.file_type,
                "doc_type": self.file_type,
                "title": title,
            },
        )

    @staticmethod
    def _extract_title(text: str) -> str | None:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                return stripped[2:].strip() or None
        return None
