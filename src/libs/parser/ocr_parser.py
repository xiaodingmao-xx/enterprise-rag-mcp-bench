"""Optional OCR parser with lazy imports and a fake-friendly page interface."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.libs.loader.parsed_document import Block, ExtractionQuality, Page, ParsedDocument
from src.libs.parser.base import BaseParser


class OCRParser(BaseParser):
    """Tesseract adapter; no OCR package is imported at module import time."""

    WARNING_DEPENDENCY_MISSING = "OCR_DEPENDENCY_MISSING"
    WARNING_FAILED = "OCR_FAILED"

    def __init__(self, *, languages: List[str] | None = None, min_confidence: float = 0.60) -> None:
        self.languages = list(languages or ["eng"])
        self.min_confidence = float(min_confidence)
        self.last_confidence = 0.0
        self.last_warnings: List[str] = []

    @property
    def available(self) -> bool:
        return bool(importlib.util.find_spec("pytesseract") and importlib.util.find_spec("PIL"))

    def parse(self, file_path: str | Path, **kwargs: Any) -> ParsedDocument:
        path = Path(file_path).resolve()
        document_id = str(kwargs.get("document_id") or "ocr_document")
        try:
            import fitz
        except ImportError:
            return self._empty_document(path, document_id, [self.WARNING_DEPENDENCY_MISSING])
        if not self.available:
            return self._empty_document(path, document_id, [self.WARNING_DEPENDENCY_MISSING])

        pages: List[Page] = []
        blocks: List[Block] = []
        warnings: List[str] = []
        confidences: List[float] = []
        try:
            with fitz.open(path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    page_blocks, confidence, page_warnings = self.parse_page(
                        page,
                        page_number=page_number,
                        document_id=document_id,
                        languages=self.languages,
                    )
                    confidences.append(confidence)
                    warnings.extend(page_warnings)
                    text = "\n".join(block.text for block in page_blocks)
                    rect = page.rect
                    pages.append(
                        Page(
                            page_number=page_number,
                            width=float(rect.width),
                            height=float(rect.height),
                            text=text,
                            text_density=float(len(text)),
                            blocks=page_blocks,
                            bbox=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
                        )
                    )
                    blocks.extend(page_blocks)
        except Exception:
            warnings.append(self.WARNING_FAILED)

        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        self.last_confidence = confidence
        self.last_warnings = list(dict.fromkeys(warnings))
        text = "\n\n".join(block.text for block in blocks if block.text)
        status = "accepted" if text and confidence >= self.min_confidence else "needs_manual_review"
        return ParsedDocument(
            document_id=document_id,
            source_path=str(path),
            source_type="pdf",
            pages=pages,
            blocks=blocks,
            paragraphs=blocks,
            reading_order=[block.block_id for block in blocks],
            extraction_quality=ExtractionQuality(
                ocr_confidence=confidence,
                quality_status=status,
                warnings=self.last_warnings,
            ),
            text=text,
        )

    def parse_page(
        self,
        pdf_page: Any,
        *,
        page_number: int,
        document_id: str,
        languages: List[str] | None = None,
    ) -> Tuple[List[Block], float, List[str]]:
        if not self.available:
            return [], 0.0, [self.WARNING_DEPENDENCY_MISSING]
        try:
            import fitz
            from PIL import Image as PILImage
            import pytesseract

            scale = 2.0
            pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            image = PILImage.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            language = "+".join(languages or self.languages)
            data: Dict[str, List[Any]] = pytesseract.image_to_data(
                image,
                lang=language,
                output_type=pytesseract.Output.DICT,
            )
            grouped: Dict[Tuple[Any, Any, Any], List[Tuple[str, float, float, float, float, float]]] = {}
            for index, raw_text in enumerate(data.get("text", [])):
                text = str(raw_text or "").strip()
                if not text:
                    continue
                try:
                    confidence = float(data.get("conf", [0])[index]) / 100.0
                except (TypeError, ValueError, IndexError):
                    confidence = 0.0
                x = float(data.get("left", [0])[index]) / scale
                y = float(data.get("top", [0])[index]) / scale
                width = float(data.get("width", [0])[index]) / scale
                height = float(data.get("height", [0])[index]) / scale
                key = (
                    data.get("block_num", [0])[index],
                    data.get("par_num", [0])[index],
                    data.get("line_num", [0])[index],
                )
                grouped.setdefault(key, []).append((text, confidence, x, y, width + x, height + y))
            blocks: List[Block] = []
            confidence_values: List[float] = []
            for block_index, words in enumerate(grouped.values(), start=1):
                words.sort(key=lambda item: item[2])
                text = " ".join(item[0] for item in words).strip()
                confidence = sum(item[1] for item in words) / len(words)
                confidence_values.append(confidence)
                bbox = (
                    min(item[2] for item in words),
                    min(item[3] for item in words),
                    max(item[4] for item in words),
                    max(item[5] for item in words),
                )
                blocks.append(
                    Block(
                        block_id=f"{document_id}_p{page_number}_ocr{block_index}",
                        block_type="paragraph",
                        text=text,
                        page_number=page_number,
                        bbox=bbox,
                        confidence=confidence,
                        reading_order_index=block_index - 1,
                        metadata={"extraction_method": "ocr"},
                    )
                )
            average = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
            warnings = ["OCR_LOW_CONFIDENCE"] if average < self.min_confidence else []
            return blocks, average, warnings
        except Exception:
            return [], 0.0, [self.WARNING_FAILED]

    @staticmethod
    def _empty_document(path: Path, document_id: str, warnings: List[str]) -> ParsedDocument:
        return ParsedDocument(
            document_id=document_id,
            source_path=str(path),
            source_type="pdf",
            extraction_quality=ExtractionQuality(
                quality_status="needs_manual_review",
                warnings=list(dict.fromkeys(warnings)),
            ),
        )
