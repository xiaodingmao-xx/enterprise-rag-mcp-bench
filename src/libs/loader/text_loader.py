"""Plain text document loader."""

from __future__ import annotations

from pathlib import Path

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader


class TextLoader(BaseLoader):
    """Load plain text files without aggressive cleanup."""

    file_type = "text"
    encodings = ("utf-8", "gbk", "latin-1")

    def load(self, file_path: str | Path) -> Document:
        path = self._validate_file(file_path)
        text = self._read_text(path)

        return Document(
            id=self._document_id(path),
            text=text,
            metadata={
                "source_path": str(path),
                "file_name": path.name,
                "file_type": self.file_type,
                "doc_type": self.file_type,
                "title": path.name,
            },
        )

    def _read_text(self, path: Path) -> str:
        last_error: UnicodeDecodeError | None = None
        for encoding in self.encodings:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        raise UnicodeDecodeError(
            last_error.encoding if last_error else "unknown",
            last_error.object if last_error else b"",
            last_error.start if last_error else 0,
            last_error.end if last_error else 0,
            f"Failed to decode text file with {', '.join(self.encodings)}",
        )
