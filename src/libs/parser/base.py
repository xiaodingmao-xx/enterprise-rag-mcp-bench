"""Parser abstractions and a dependency-light text parser."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
from pathlib import Path
from typing import Any

from src.libs.loader.parsed_document import Block, Page, ParsedDocument


class BaseParser(ABC):
    """Common parser interface; optional engines are loaded by implementations."""

    @abstractmethod
    def parse(self, file_path: str | Path, **kwargs: Any) -> ParsedDocument:
        raise NotImplementedError


class BasicTextParser(BaseParser):
    """Parse text-bearing files without OCR/layout/table dependencies."""

    def parse(self, file_path: str | Path, **kwargs: Any) -> ParsedDocument:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        document_id = str(
            kwargs.get("document_id")
            or f"doc_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
        )
        source_type = str(kwargs.get("source_type") or path.suffix.lower().lstrip("."))
        if path.suffix.lower() == ".pdf":
            return self._parse_pdf_text(path, document_id, source_type)
        text = self._load_existing_format(path)
        block = Block(
            block_id=f"{document_id}_p1_b1",
            block_type="paragraph",
            text=text,
            page_number=1,
            bbox=None,
            reading_order_index=0,
        )
        page = Page(
            page_number=1,
            text=text,
            text_density=float(len(text)),
            blocks=[block],
            bbox=None,
        )
        return ParsedDocument(
            document_id=document_id,
            source_path=str(path),
            source_type=source_type,
            pages=[page],
            blocks=[block],
            paragraphs=[block],
            reading_order=[block.block_id],
            text=text,
        )

    @staticmethod
    def _read_text(path: Path) -> str:
        if path.suffix.lower() in {".html", ".htm"}:
            import re

            raw = path.read_text(encoding="utf-8", errors="ignore")
            return re.sub(r"<[^>]+>", " ", raw)
        return path.read_text(encoding="utf-8", errors="ignore")

    @classmethod
    def _load_existing_format(cls, path: Path) -> str:
        """Reuse existing format loaders without creating a second parser."""

        try:
            from src.libs.loader.loader_factory import LoaderFactory

            return str(LoaderFactory.get_loader(path).load(path).text or "")
        except Exception:
            return cls._read_text(path)

    def _parse_pdf_text(self, path: Path, document_id: str, source_type: str) -> ParsedDocument:
        try:
            import fitz
        except ImportError:
            text = path.read_bytes().decode("utf-8", errors="ignore")
            return self._single_page(path, document_id, source_type, text)

        pages = []
        blocks = []
        with fitz.open(path) as pdf:
            for page_number, pdf_page in enumerate(pdf, start=1):
                text = pdf_page.get_text("text") or ""
                rect = pdf_page.rect
                block = Block(
                    block_id=f"{document_id}_p{page_number}_b1",
                    block_type="paragraph",
                    text=text.strip(),
                    page_number=page_number,
                    bbox=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
                    reading_order_index=len(blocks),
                )
                page = Page(
                    page_number=page_number,
                    width=float(rect.width),
                    height=float(rect.height),
                    text=text,
                    text_density=float(len(text)),
                    blocks=[block] if block.text else [],
                    bbox=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
                )
                pages.append(page)
                if block.text:
                    blocks.append(block)
        all_text = "\n\n".join(block.text for block in blocks if block.text)
        return ParsedDocument(
            document_id=document_id,
            source_path=str(path),
            source_type=source_type,
            pages=pages,
            blocks=blocks,
            paragraphs=list(blocks),
            reading_order=[block.block_id for block in blocks],
            text=all_text,
        )

    @staticmethod
    def _single_page(path: Path, document_id: str, source_type: str, text: str) -> ParsedDocument:
        block = Block(
            block_id=f"{document_id}_p1_b1",
            block_type="paragraph",
            text=text,
            page_number=1,
            reading_order_index=0,
        )
        page = Page(page_number=1, text=text, text_density=float(len(text)), blocks=[block])
        return ParsedDocument(
            document_id=document_id,
            source_path=str(path),
            source_type=source_type,
            pages=[page],
            blocks=[block] if text else [],
            paragraphs=[block] if text else [],
            reading_order=[block.block_id] if text else [],
            text=text,
        )
