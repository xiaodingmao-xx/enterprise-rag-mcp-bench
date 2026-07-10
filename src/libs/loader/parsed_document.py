"""Serializable structured parsing contract used as an enhancement layer.

``Document`` remains the ingestion contract.  ``ParsedDocument`` is a
document-level parse artifact that can be adapted to ``Document.metadata``
without copying large tables, images, or complete page structures into every
chunk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


BBox = Optional[Tuple[float, float, float, float]]


@dataclass
class TableCell:
    row_index: int
    col_index: int
    text: str = ""
    bbox: BBox = None
    confidence: float = 0.0
    is_header: bool = False


@dataclass
class Table:
    table_id: str
    page_number: int
    bbox: BBox = None
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    cells: List[TableCell] = field(default_factory=list)
    markdown: str = ""
    plain_text: str = ""
    confidence: float = 0.0
    extraction_method: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Image:
    image_id: str
    page_number: int
    bbox: BBox = None
    width: int = 0
    height: int = 0
    caption: str = ""
    alt_text: str = ""
    extracted_path: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_legacy_metadata(self) -> Dict[str, Any]:
        """Return the existing ``Document.metadata.images`` shape."""

        path = self.extracted_path or ""
        return {
            "id": self.image_id,
            "path": path,
            "page": self.page_number,
            "text_offset": int(self.metadata.get("text_offset", 0) or 0),
            "text_length": int(self.metadata.get("text_length", 0) or 0),
            "position": {
                "bbox": list(self.bbox) if self.bbox else None,
                "width": self.width,
                "height": self.height,
                "page": self.page_number,
            },
            "caption": self.caption,
            "confidence": self.confidence,
        }


@dataclass
class Block:
    block_id: str
    block_type: str
    text: str = ""
    page_number: int = 0
    bbox: BBox = None
    parent_block_id: Optional[str] = None
    confidence: float = 1.0
    reading_order_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Page:
    page_number: int
    width: float = 0.0
    height: float = 0.0
    text: str = ""
    text_density: float = 0.0
    blocks: List[Block] = field(default_factory=list)
    images: List[Image] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    bbox: BBox = None


@dataclass
class ExtractionQuality:
    text_density: float = 0.0
    garbled_ratio: float = 0.0
    ocr_confidence: float = 0.0
    empty_page_ratio: float = 0.0
    table_extraction_success: bool = True
    duplicate_block_ratio: float = 0.0
    quality_status: str = "accepted"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedDocument:
    document_id: str
    source_path: str
    source_type: str = "unknown"
    pages: List[Page] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    paragraphs: List[Block] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    images: List[Image] = field(default_factory=list)
    headers: List[Block] = field(default_factory=list)
    footers: List[Block] = field(default_factory=list)
    headings: List[Block] = field(default_factory=list)
    reading_order: List[str] = field(default_factory=list)
    source_coordinates: Dict[str, Any] = field(default_factory=dict)
    extraction_quality: ExtractionQuality = field(default_factory=ExtractionQuality)
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, *, ensure_ascii: bool = False) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, default=str)

    def summary_metadata(self) -> Dict[str, Any]:
        """Return only lightweight references suitable for Document metadata."""

        return {
            "document_id": self.document_id,
            "source_type": self.source_type,
            "page_count": len(self.pages),
            "block_count": len(self.blocks),
            "table_count": len(self.tables),
            "image_count": len(self.images),
            "heading_count": len(self.headings),
            "pages": [
                {
                    "page_number": page.page_number,
                    "width": page.width,
                    "height": page.height,
                    "text_density": page.text_density,
                    "bbox": list(page.bbox) if page.bbox else None,
                }
                for page in self.pages
            ],
            "blocks": [
                {
                    "block_id": block.block_id,
                    "block_type": block.block_type,
                    "page_number": block.page_number,
                    "bbox": list(block.bbox) if block.bbox else None,
                }
                for block in self.blocks
            ],
            "table_refs": [
                {
                    "table_id": table.table_id,
                    "page_number": table.page_number,
                    "bbox": list(table.bbox) if table.bbox else None,
                }
                for table in self.tables
            ],
            "image_refs": [
                {
                    "image_id": image.image_id,
                    "page_number": image.page_number,
                    "bbox": list(image.bbox) if image.bbox else None,
                    "caption": image.caption,
                    "extracted_path": image.extracted_path,
                }
                for image in self.images
            ],
            "headers_footers": {
                "headers": [block.text for block in self.headers],
                "footers": [block.text for block in self.footers],
            },
            "extraction_quality": self.extraction_quality.to_dict(),
        }

    def to_document(self, *, extra_metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Adapt to the existing ``src.core.types.Document`` contract."""

        from src.core.types import Document

        metadata: Dict[str, Any] = dict(self.metadata)
        summary = self.summary_metadata()
        metadata.update(
            {
                "source_path": self.source_path,
                "doc_type": self.source_type,
                "document_id": self.document_id,
                "page_count": len(self.pages),
                "parser_version": "enhanced-pdf-v1",
                "parsed_document_summary": summary,
                "pages": summary["pages"],
                "blocks": summary["blocks"],
                "table_refs": summary["table_refs"],
                "image_refs": summary["image_refs"],
                "extraction_quality": self.extraction_quality.to_dict(),
                "headers_footers": {
                    "headers": [block.text for block in self.headers],
                    "footers": [block.text for block in self.footers],
                },
                "images": [image.to_legacy_metadata() for image in self.images],
            }
        )
        if extra_metadata:
            metadata.update(extra_metadata)
        return Document(id=self.document_id, text=self.text, metadata=metadata)
