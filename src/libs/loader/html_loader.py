"""HTML document loader."""

from __future__ import annotations

from pathlib import Path

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

try:
    from bs4 import BeautifulSoup

    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False


class HtmlLoader(BaseLoader):
    """Load HTML files and extract readable text."""

    file_type = "html"

    def __init__(self) -> None:
        if not BEAUTIFULSOUP_AVAILABLE:
            raise ImportError(
                "beautifulsoup4 is required for HtmlLoader. "
                "Install with: pip install beautifulsoup4"
            )

    def load(self, file_path: str | Path) -> Document:
        path = self._validate_file(file_path)
        html = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        title = self._extract_title(soup) or path.name
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        readable_text = "\n".join(line for line in lines if line)

        return Document(
            id=self._document_id(path),
            text=readable_text,
            metadata={
                "source_path": str(path),
                "file_name": path.name,
                "file_type": self.file_type,
                "doc_type": self.file_type,
                "title": title,
            },
        )

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str | None:
        if soup.title and soup.title.string:
            return soup.title.string.strip() or None

        heading = soup.find("h1")
        if heading:
            return heading.get_text(strip=True) or None
        return None
