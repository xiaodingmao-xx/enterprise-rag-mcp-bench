"""Lightweight PyMuPDF-based page/block and reading-order parser."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.libs.loader.parsed_document import Block, Image, Page, ParsedDocument
from src.libs.parser.base import BaseParser


class LayoutParser(BaseParser):
    """Extract blocks with coordinates without requiring a layout model."""

    def __init__(self, *, extract_images: bool = True, image_storage_dir: str | Path = "data/images") -> None:
        self.extract_images = bool(extract_images)
        self.image_storage_dir = Path(image_storage_dir)

    def parse(self, file_path: str | Path, **kwargs: Any) -> ParsedDocument:
        path = Path(file_path).resolve()
        document_id = str(kwargs.get("document_id") or self._document_id(path))
        try:
            import fitz
        except ImportError:
            return ParsedDocument(
                document_id=document_id,
                source_path=str(path),
                source_type=path.suffix.lower().lstrip("."),
                metadata={"warnings": ["LAYOUT_DEPENDENCY_MISSING"]},
            )

        pages: List[Page] = []
        blocks: List[Block] = []
        images: List[Image] = []
        with fitz.open(path) as pdf:
            for page_number, pdf_page in enumerate(pdf, start=1):
                page, page_blocks, page_images = self.parse_page(
                    pdf_page,
                    document_id=document_id,
                    page_number=page_number,
                    image_storage_dir=self.image_storage_dir,
                    extract_images=self.extract_images,
                )
                pages.append(page)
                blocks.extend(page_blocks)
                images.extend(page_images)
        blocks = self.order_blocks(blocks)
        for index, block in enumerate(blocks):
            block.reading_order_index = index
        text = "\n\n".join(block.text for block in blocks if block.text)
        return ParsedDocument(
            document_id=document_id,
            source_path=str(path),
            source_type="pdf",
            pages=pages,
            blocks=blocks,
            paragraphs=[block for block in blocks if block.block_type == "paragraph"],
            images=images,
            headings=[block for block in blocks if block.block_type == "heading"],
            reading_order=[block.block_id for block in blocks],
            source_coordinates={block.block_id: block.bbox for block in blocks},
            text=text,
        )

    @classmethod
    def parse_page(
        cls,
        pdf_page: Any,
        *,
        document_id: str,
        page_number: int,
        image_storage_dir: str | Path = "data/images",
        extract_images: bool = True,
    ) -> Tuple[Page, List[Block], List[Image]]:
        rect = pdf_page.rect
        page_bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
        raw = pdf_page.get_text("dict") or {}
        raw_blocks = raw.get("blocks", [])
        blocks: List[Block] = []
        images: List[Image] = []

        for block_index, raw_block in enumerate(raw_blocks, start=1):
            bbox = cls._bbox(raw_block.get("bbox"))
            if int(raw_block.get("type", 0)) == 1:
                image = cls._image_from_block(
                    pdf_page,
                    raw_block,
                    document_id=document_id,
                    page_number=page_number,
                    image_index=len(images) + 1,
                    bbox=bbox,
                    image_storage_dir=image_storage_dir,
                    extract_images=extract_images,
                )
                if image is not None:
                    images.append(image)
                    blocks.append(
                        Block(
                            block_id=f"{document_id}_p{page_number}_img{len(images)}",
                            block_type="image",
                            page_number=page_number,
                            bbox=bbox,
                            confidence=image.confidence,
                            metadata={"image_id": image.image_id},
                        )
                    )
                continue

            text = cls._block_text(raw_block).strip()
            if not text:
                continue
            block_type, features = cls.classify_text_block(raw_block, text)
            blocks.append(
                Block(
                    block_id=f"{document_id}_p{page_number}_b{block_index}",
                    block_type=block_type,
                    text=text,
                    page_number=page_number,
                    bbox=bbox,
                    confidence=1.0,
                    metadata=features,
                )
            )

        # Some PDFs expose images through get_images but omit image blocks in
        # the text dictionary.  Add those images with page association.
        if extract_images:
            for image_info in pdf_page.get_images(full=True):
                xref = int(image_info[0])
                if any(item.metadata.get("xref") == xref for item in images):
                    continue
                rects = pdf_page.get_image_rects(xref)
                image_bbox = cls._bbox(rects[0]) if rects else None
                image = cls._extract_image(
                    pdf_page,
                    xref,
                    document_id=document_id,
                    page_number=page_number,
                    image_index=len(images) + 1,
                    bbox=image_bbox,
                    image_storage_dir=image_storage_dir,
                    extract_images=extract_images,
                )
                if image is not None:
                    images.append(image)
                    blocks.append(
                        Block(
                            block_id=f"{document_id}_p{page_number}_img{len(images)}",
                            block_type="image",
                            page_number=page_number,
                            bbox=image_bbox,
                            confidence=image.confidence,
                            metadata={"image_id": image.image_id},
                        )
                    )

        text = "\n".join(block.text for block in blocks if block.text)
        page = Page(
            page_number=page_number,
            width=float(rect.width),
            height=float(rect.height),
            text=text,
            text_density=float(len(text)),
            blocks=blocks,
            images=images,
            bbox=page_bbox,
        )
        return page, blocks, images

    @staticmethod
    def order_blocks(blocks: List[Block]) -> List[Block]:
        return sorted(
            blocks,
            key=lambda block: (
                int(block.page_number),
                float(block.bbox[1]) if block.bbox else 0.0,
                float(block.bbox[0]) if block.bbox else 0.0,
            ),
        )

    @classmethod
    def classify_text_block(cls, raw_block: Dict[str, Any], text: str) -> Tuple[str, Dict[str, Any]]:
        spans = [
            span
            for line in raw_block.get("lines", [])
            for span in line.get("spans", [])
        ]
        max_size = max((float(span.get("size", 0.0)) for span in spans), default=0.0)
        fonts = " ".join(str(span.get("font", "")) for span in spans).lower()
        bold = "bold" in fonts or "black" in fonts or "semibold" in fonts
        compact = " ".join(text.split())
        numbered = bool(re.match(r"^(?:\d+(?:\.\d+)*[.)]?|[一二三四五六七八九十]+[、.)])\s+", compact))
        bullet = bool(re.match(r"^(?:[-*•▪◦]|\(?\d+[.)])\s+", compact))
        caption = bool(re.match(r"^(?:图|表|figure|fig\.?|table)\s*\d*\s*[:：.、]", compact, re.I))
        code = "mono" in fonts or bool(re.search(r"[{};]|=>|def\s+\w+\(|class\s+\w+", compact))
        formula = sum(char in "=+-*/∑∫√≤≥≈" for char in compact) >= max(2, len(compact) // 8)
        if caption:
            block_type = "caption"
        elif code:
            block_type = "code"
        elif formula:
            block_type = "formula"
        elif numbered or bullet:
            block_type = "list"
        elif bold or max_size >= 14 or (len(compact) <= 100 and re.match(r"^\d+(?:\.\d+)*\s", compact)):
            block_type = "heading"
        else:
            block_type = "paragraph"
        return block_type, {"font_size": max_size, "bold": bold, "font_names": fonts}

    @staticmethod
    def _block_text(raw_block: Dict[str, Any]) -> str:
        lines = []
        for line in raw_block.get("lines", []):
            value = "".join(str(span.get("text", "")) for span in line.get("spans", []))
            if value:
                lines.append(value)
        return "\n".join(lines) or str(raw_block.get("text", ""))

    @staticmethod
    def _bbox(value: Any) -> Optional[Tuple[float, float, float, float]]:
        try:
            if hasattr(value, "x0"):
                return (float(value.x0), float(value.y0), float(value.x1), float(value.y1))
            if value is None or len(value) != 4:
                return None
            return tuple(float(item) for item in value)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _document_id(path: Path) -> str:
        return f"doc_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"

    @classmethod
    def _image_from_block(
        cls,
        pdf_page: Any,
        raw_block: Dict[str, Any],
        *,
        document_id: str,
        page_number: int,
        image_index: int,
        bbox: Any,
        image_storage_dir: str | Path,
        extract_images: bool,
    ) -> Optional[Image]:
        xref = raw_block.get("xref")
        if xref:
            image = cls._extract_image(
                pdf_page,
                int(xref),
                document_id=document_id,
                page_number=page_number,
                image_index=image_index,
                bbox=bbox,
                image_storage_dir=image_storage_dir,
                extract_images=extract_images,
            )
            if image is not None:
                return image
        image_id = f"{document_id[4:12]}_{page_number}_{image_index}"
        return Image(
            image_id=image_id,
            page_number=page_number,
            bbox=bbox,
            width=int(raw_block.get("width", 0) or 0),
            height=int(raw_block.get("height", 0) or 0),
            confidence=0.8,
            metadata={"xref": xref} if xref else {},
        )

    @classmethod
    def _extract_image(
        cls,
        pdf_page: Any,
        xref: int,
        *,
        document_id: str,
        page_number: int,
        image_index: int,
        bbox: Any,
        image_storage_dir: str | Path,
        extract_images: bool,
    ) -> Optional[Image]:
        image_id = f"{document_id[4:12]}_{page_number}_{image_index}"
        parent = getattr(pdf_page, "parent", None)
        if parent is None or not hasattr(parent, "extract_image"):
            return Image(image_id=image_id, page_number=page_number, bbox=bbox, confidence=0.8, metadata={"xref": xref})
        try:
            data = parent.extract_image(xref)
            width, height = int(data.get("width", 0)), int(data.get("height", 0))
            extracted_path = None
            if extract_images:
                target_dir = Path(image_storage_dir) / document_id
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / f"{image_id}.{data.get('ext', 'bin')}"
                target.write_bytes(data["image"])
                try:
                    extracted_path = str(target.relative_to(Path.cwd()))
                except ValueError:
                    extracted_path = str(target.resolve())
            return Image(
                image_id=image_id,
                page_number=page_number,
                bbox=bbox,
                width=width,
                height=height,
                extracted_path=extracted_path,
                confidence=1.0,
                metadata={"xref": xref},
            )
        except Exception:
            return Image(image_id=image_id, page_number=page_number, bbox=bbox, confidence=0.5, metadata={"xref": xref})
