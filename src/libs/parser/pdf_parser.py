"""Enhanced PDF parser orchestrating layout, OCR, tables, images and quality."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from src.libs.loader.document_quality import assess_parsed_document_quality
from src.libs.loader.parsed_document import Block, ExtractionQuality, Page, ParsedDocument, Table
from src.libs.parser.base import BaseParser, BasicTextParser
from src.libs.parser.header_footer import HeaderFooterDetector
from src.libs.parser.layout_parser import LayoutParser
from src.libs.parser.ocr_parser import OCRParser
from src.libs.parser.table_parser import TableParser


class PdfParser(BaseParser):
    """Parse text-layer PDFs first and use OCR only for low-density pages."""

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        ocr_parser: Any = None,
        layout_parser: Any = None,
        table_parser: Any = None,
        header_footer_detector: Any = None,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images",
    ) -> None:
        self.config = dict(config or {})
        self.ocr_config = self._mapping(self.config.get("ocr"))
        self.layout_config = self._mapping(self.config.get("layout"))
        self.table_config = self._mapping(self.config.get("tables"))
        self.header_footer_config = self._mapping(self.config.get("headers_footers"))
        self.quality_config = self._mapping(self.config.get("quality"))
        self.extract_images = bool(extract_images)
        self.image_storage_dir = image_storage_dir
        self.ocr_parser = ocr_parser or OCRParser(
            languages=[str(item) for item in self.ocr_config.get("languages", ["eng"])],
            min_confidence=float(self.ocr_config.get("min_ocr_confidence", 0.60)),
        )
        self.layout_parser = layout_parser or LayoutParser(
            extract_images=self.extract_images and bool(self.layout_config.get("extract_images", True)),
            image_storage_dir=image_storage_dir,
        )
        self.table_parser = table_parser or TableParser(
            engine=str(self.table_config.get("engine", "auto")),
            extract_markdown=bool(self.table_config.get("extract_markdown", True)),
        )
        self.header_footer_detector = header_footer_detector or HeaderFooterDetector(
            similarity_threshold=float(self.header_footer_config.get("similarity_threshold", 0.85)),
            min_repeat_pages=int(self.header_footer_config.get("min_repeat_pages", 2)),
            top_margin_ratio=float(self.header_footer_config.get("top_margin_ratio", 0.12)),
            bottom_margin_ratio=float(self.header_footer_config.get("bottom_margin_ratio", 0.12)),
        )
        self.last_warnings: List[str] = []

    def parse(self, file_path: str | Path, **kwargs: Any) -> ParsedDocument:
        path = Path(file_path).resolve()
        document_id = str(kwargs.get("document_id") or self._document_id(path))
        if not bool(self.config.get("enabled", True)):
            parsed = BasicTextParser().parse(path, document_id=document_id)
            parsed.extraction_quality = ExtractionQuality(
                text_density=parsed.pages[0].text_density if parsed.pages else 0.0,
                quality_status="warning",
                warnings=["ENHANCED_PARSING_DISABLED"],
            )
            parsed.metadata["warnings"] = ["ENHANCED_PARSING_DISABLED"]
            return parsed

        try:
            import fitz
        except ImportError:
            parsed = BasicTextParser().parse(path, document_id=document_id)
            parsed.extraction_quality = ExtractionQuality(
                quality_status="warning",
                warnings=["LAYOUT_DEPENDENCY_MISSING"],
            )
            return parsed

        pages: List[Page] = []
        blocks: List[Block] = []
        tables: List[Table] = []
        images = []
        warnings: List[str] = []
        table_warnings: List[str] = []
        ocr_confidence = 0.0

        with fitz.open(path) as pdf:
            for page_number, pdf_page in enumerate(pdf, start=1):
                try:
                    page, page_blocks, page_images = self.layout_parser.parse_page(
                        pdf_page,
                        document_id=document_id,
                        page_number=page_number,
                        image_storage_dir=self.image_storage_dir,
                        extract_images=self.extract_images,
                    )
                except TypeError:
                    page, page_blocks, page_images = self.layout_parser.parse_page(
                        pdf_page, page_number, document_id
                    )
                if bool(self.table_config.get("enabled", True)):
                    try:
                        page_tables = self.table_parser.parse_page(
                            pdf_page,
                            page_number=page_number,
                            document_id=document_id,
                        )
                    except TypeError:
                        page_tables = self.table_parser.parse_page(pdf_page, page_number, document_id)
                else:
                    page_tables = []
                page_table_warnings = list(getattr(self.table_parser, "last_warnings", []) or [])
                table_warnings.extend(page_table_warnings)
                if not page_tables and "TABLE_EXTRACTION_FAILED" in page_table_warnings:
                    fallback_text = str(pdf_page.get_text("text") or "").strip()
                    if fallback_text:
                        page_blocks.append(
                            Block(
                                block_id=f"{document_id}_p{page_number}_table_fallback",
                                block_type="table",
                                text=fallback_text,
                                page_number=page_number,
                                confidence=0.2,
                                metadata={"extraction_method": "plain_text_fallback"},
                            )
                        )
                for table in page_tables:
                    table_block = Block(
                        block_id=table.table_id,
                        block_type="table",
                        text=f"[TABLE: {table.table_id}]\n{table.markdown or table.plain_text}",
                        page_number=page_number,
                        bbox=table.bbox,
                        confidence=table.confidence,
                        metadata={"table_id": table.table_id, **table.metadata},
                    )
                    page_blocks.append(table_block)
                page.tables = page_tables
                page.blocks = page_blocks
                page.text = "\n".join(block.text for block in page_blocks if block.text)
                page.text_density = float(len(page.text))
                pages.append(page)
                blocks.extend(page_blocks)
                tables.extend(page_tables)
                images.extend(page_images)

            average_density = sum(len(page.text) for page in pages) / max(1, len(pages))
            min_density = float(self.ocr_config.get("min_text_density", 20) or 20)
            if average_density < min_density:
                warnings.append("LOW_TEXT_DENSITY")
                if bool(self.ocr_config.get("enabled", False)):
                    ocr_confidence, ocr_warnings = self._apply_ocr(
                        pdf,
                        pages,
                        blocks,
                        document_id=document_id,
                        file_path=path,
                        min_text_density=min_density,
                    )
                    warnings.extend(ocr_warnings)

        if bool(self.header_footer_config.get("enabled", True)):
            headers, footers, body_blocks, header_warnings = self.header_footer_detector.detect(pages)
            warnings.extend(header_warnings)
        else:
            headers, footers = [], []
            body_blocks = [block for page in pages for block in page.blocks]
        self._associate_captions(images, body_blocks)
        self._mark_cross_page_tables(tables, warnings)
        body_blocks = LayoutParser.order_blocks(body_blocks)
        for index, block in enumerate(body_blocks):
            block.reading_order_index = index
        headings = [block for block in blocks if block.block_type == "heading"]
        paragraphs = [
            block
            for block in body_blocks
            if block.block_type in {"paragraph", "heading", "caption", "list", "code", "formula", "table"}
        ]
        text = "\n\n".join(block.text for block in body_blocks if block.text)
        quality = assess_parsed_document_quality(
            pages,
            blocks,
            tables,
            ocr_confidence=ocr_confidence,
            warnings=warnings + table_warnings,
            config=self.quality_config,
        )
        parsed = ParsedDocument(
            document_id=document_id,
            source_path=str(path),
            source_type="pdf",
            pages=pages,
            blocks=blocks,
            paragraphs=paragraphs,
            tables=tables,
            images=images,
            headers=headers,
            footers=footers,
            headings=headings,
            reading_order=[block.block_id for block in body_blocks],
            source_coordinates={block.block_id: block.bbox for block in blocks},
            extraction_quality=quality,
            metadata={
                "enhanced_pdf": True,
                "excluded_headers": [block.text for block in headers],
                "excluded_footers": [block.text for block in footers],
            },
            text=text,
        )
        self.last_warnings = quality.warnings
        return parsed

    def _apply_ocr(
        self,
        pdf: Any,
        pages: List[Page],
        blocks: List[Block],
        *,
        document_id: str,
        file_path: Path,
        min_text_density: float,
    ):
        warnings: List[str] = []
        confidences: List[float] = []
        for page_number, page in enumerate(pages, start=1):
            if page.text_density >= min_text_density and page.text.strip():
                continue
            pdf_page = pdf[page_number - 1]
            parse_page = getattr(self.ocr_parser, "parse_page", None)
            if callable(parse_page):
                try:
                    ocr_blocks, confidence, page_warnings = parse_page(
                        pdf_page,
                        page_number=page_number,
                        document_id=document_id,
                    )
                except TypeError:
                    ocr_blocks, confidence, page_warnings = parse_page(pdf_page, page_number, document_id)
                except Exception:
                    ocr_blocks, confidence, page_warnings = [], 0.0, ["OCR_FAILED"]
            else:
                result = self.ocr_parser.parse(
                    str(file_path),
                    document_id=document_id,
                    pages=[page_number],
                )
                ocr_page = result.pages[0] if getattr(result, "pages", None) else None
                ocr_blocks = list(getattr(ocr_page, "blocks", []) if ocr_page else [])
                confidence = float(getattr(getattr(result, "extraction_quality", None), "ocr_confidence", 0.0) or 0.0)
                page_warnings = list(getattr(getattr(result, "extraction_quality", None), "warnings", []) or [])
            confidences.append(float(confidence or 0.0))
            warnings.extend(page_warnings or [])
            if ocr_blocks:
                page.blocks.extend(ocr_blocks)
                page.text = "\n".join(block.text for block in page.blocks if block.text)
                page.text_density = float(len(page.text))
                blocks.extend(ocr_blocks)
            elif not page_warnings:
                warnings.append("OCR_FAILED")
        return (sum(confidences) / len(confidences) if confidences else 0.0), list(dict.fromkeys(warnings))

    @staticmethod
    def _associate_captions(images: List[Any], blocks: List[Block]) -> None:
        for image in images:
            candidates = [
                block
                for block in blocks
                if block.page_number == image.page_number
                and block.block_type == "caption"
                and block.bbox
                and image.bbox
                and abs(block.bbox[1] - image.bbox[3]) <= 140
            ]
            if candidates:
                candidate = sorted(candidates, key=lambda block: abs(block.bbox[1] - image.bbox[3]))[0]
                image.caption = candidate.text
                image.alt_text = candidate.text
                image.metadata["caption_block_id"] = candidate.block_id

    @staticmethod
    def _mark_cross_page_tables(tables: List[Table], warnings: List[str]) -> None:
        by_page: Dict[int, List[Table]] = {}
        for table in tables:
            by_page.setdefault(table.page_number, []).append(table)
        for page_number in sorted(by_page):
            next_tables = by_page.get(page_number + 1, [])
            for current in by_page[page_number]:
                for following in next_tables:
                    current_columns = len(current.headers) or max((len(row) for row in current.rows), default=0)
                    following_columns = len(following.headers) or max((len(row) for row in following.rows), default=0)
                    if current_columns and current_columns == following_columns:
                        current.metadata["possible_continuation"] = True
                        following.metadata["possible_continuation"] = True
                        warnings.append("CROSS_PAGE_TABLE_CANDIDATE")

    @staticmethod
    def _mapping(value: Any) -> Dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _document_id(path: Path) -> str:
        return f"doc_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"


EnhancedPdfParser = PdfParser
