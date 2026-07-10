"""Lightweight PDF quality gate for ingestion.

The checker samples the first few pages of a PDF before the expensive loader
stage. It rejects files whose text layer is empty or mostly unreadable, which
prevents low-quality chunks from polluting the vector and keyword indexes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import unicodedata

from src.libs.loader.parsed_document import Block, ExtractionQuality, Page, Table

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


DOCUMENT_QUALITY_REJECTION_MESSAGE = "该文档质量不达标，请检查后重新上传"


@dataclass(frozen=True)
class DocumentQualityReport:
    """Result of a document quality preflight check."""

    checked: bool
    passed: bool
    reason: str
    file_path: str
    page_count: int = 0
    sampled_pages: int = 0
    raw_char_count: int = 0
    text_char_count: int = 0
    effective_char_count: int = 0
    effective_char_ratio: float = 0.0
    recognizable_text_density: float = 0.0
    min_effective_char_ratio: float = 0.8
    min_recognizable_text_density: float = 20.0
    text_density: float = 0.0
    garbled_ratio: float = 0.0
    ocr_confidence: float = 0.0
    empty_page_ratio: float = 0.0
    table_extraction_success: bool = True
    duplicate_block_ratio: float = 0.0
    quality_status: str = "accepted"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        value = asdict(self)
        value["warnings"] = list(self.warnings or [])
        return value


class PdfQualityChecker:
    """Preflight quality checker for PDF text layers."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        sample_pages: int = 3,
        min_effective_char_ratio: float = 0.8,
        min_recognizable_text_density: float = 20.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.sample_pages = max(1, int(sample_pages))
        self.min_effective_char_ratio = float(min_effective_char_ratio)
        self.min_recognizable_text_density = float(min_recognizable_text_density)

    def check(self, file_path: str | Path) -> DocumentQualityReport:
        """Check a PDF before full ingestion.

        Non-PDF files are passed through because this checker only evaluates
        PDF text layers; downstream loaders remain responsible for format
        support.
        """
        path = Path(file_path)
        if not self.enabled:
            return self._pass(path, checked=False, reason="disabled")
        if path.suffix.lower() != ".pdf":
            return self._pass(path, checked=False, reason="non_pdf")
        if not PYMUPDF_AVAILABLE:
            return self._pass(path, checked=False, reason="pymupdf_unavailable")

        try:
            with fitz.open(path) as doc:
                page_count = len(doc)
                sampled_pages = min(self.sample_pages, page_count)
                text_parts = [
                    doc[page_index].get_text("text")
                    for page_index in range(sampled_pages)
                ]
        except Exception as exc:
            return DocumentQualityReport(
                checked=True,
                passed=False,
                reason=f"pdf_open_failed: {exc}",
                file_path=str(path),
                min_effective_char_ratio=self.min_effective_char_ratio,
                min_recognizable_text_density=self.min_recognizable_text_density,
            )

        return self.evaluate_text_sample(
            "\n".join(text_parts),
            file_path=path,
            page_count=page_count,
            sampled_pages=sampled_pages,
        )

    def evaluate_text_sample(
        self,
        text: str,
        *,
        file_path: str | Path,
        page_count: int,
        sampled_pages: int,
    ) -> DocumentQualityReport:
        """Evaluate already-extracted sample text."""
        text_chars = [char for char in text if not char.isspace()]
        effective_chars = [char for char in text_chars if self._is_effective_char(char)]
        effective_char_ratio = (
            len(effective_chars) / len(text_chars)
            if text_chars
            else 0.0
        )
        density_pages = max(1, sampled_pages)
        recognizable_text_density = len(effective_chars) / density_pages

        passed = (
            effective_char_ratio >= self.min_effective_char_ratio
            and recognizable_text_density >= self.min_recognizable_text_density
        )
        if passed:
            reason = "passed"
        elif effective_char_ratio < self.min_effective_char_ratio:
            reason = "low_effective_char_ratio"
        else:
            reason = "low_recognizable_text_density"

        return DocumentQualityReport(
            checked=True,
            passed=passed,
            reason=reason,
            file_path=str(Path(file_path)),
            page_count=page_count,
            sampled_pages=sampled_pages,
            raw_char_count=len(text),
            text_char_count=len(text_chars),
            effective_char_count=len(effective_chars),
            effective_char_ratio=round(effective_char_ratio, 4),
            recognizable_text_density=round(recognizable_text_density, 2),
            min_effective_char_ratio=self.min_effective_char_ratio,
            min_recognizable_text_density=self.min_recognizable_text_density,
        )

    def _pass(
        self,
        path: Path,
        *,
        checked: bool,
        reason: str,
    ) -> DocumentQualityReport:
        return DocumentQualityReport(
            checked=checked,
            passed=True,
            reason=reason,
            file_path=str(path),
            min_effective_char_ratio=self.min_effective_char_ratio,
            min_recognizable_text_density=self.min_recognizable_text_density,
        )

    @staticmethod
    def _is_effective_char(char: str) -> bool:
        if char == "\ufffd":
            return False

        category = unicodedata.category(char)
        if category[0] in {"L", "N"}:
            return True
        if category[0] == "P":
            return True

        return char in {
            "%",
            "&",
            "+",
            "-",
            "/",
            "=",
            "@",
            "#",
            "$",
        }


def garbled_ratio(text: str) -> float:
    """Estimate unreadable/encoding-corrupted text without language models."""

    chars = [char for char in str(text or "") if not char.isspace()]
    if not chars:
        return 0.0
    suspicious = 0
    for char in chars:
        category = unicodedata.category(char)
        if char == "\ufffd" or category in {"Cc", "Cf", "Co", "Cs"}:
            suspicious += 1
        elif category == "So" and char not in {"©", "®", "™"}:
            suspicious += 1
        elif char in "ÃÂÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞß":
            suspicious += 1
    return min(1.0, suspicious / len(chars))


def assess_parsed_document_quality(
    pages: Sequence[Page],
    blocks: Sequence[Block],
    tables: Sequence[Table],
    *,
    ocr_confidence: float = 0.0,
    warnings: Iterable[str] = (),
    config: Mapping[str, Any] | None = None,
) -> ExtractionQuality:
    """Build the enhanced quality contract from parsed pages and blocks."""

    config = config or {}
    page_count = max(1, len(pages))
    page_texts = [str(page.text or "") for page in pages]
    combined = "\n".join(page_texts)
    text_density = len("".join(part for part in page_texts if part)) / page_count
    empty_ratio = sum(not text.strip() for text in page_texts) / page_count
    normalized = [" ".join(block.text.split()).lower() for block in blocks if block.text.strip()]
    counts = Counter(normalized)
    duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
    duplicate_ratio = duplicate_count / len(normalized) if normalized else 0.0
    quality_warnings = list(dict.fromkeys(str(item) for item in warnings if item))
    ratio = garbled_ratio(combined)
    min_density = float(config.get("min_text_density", 20.0) or 20.0)
    max_garbled = float(config.get("max_garbled_ratio", 0.20) or 0.20)
    max_empty = float(config.get("max_empty_page_ratio", 0.50) or 0.50)
    max_duplicate = float(config.get("max_duplicate_block_ratio", 0.40) or 0.40)

    if ratio >= max_garbled:
        quality_warnings.append("GARBLE_TEXT_DETECTED")
    if text_density < min_density:
        quality_warnings.append("LOW_TEXT_DENSITY")
    if empty_ratio > 0:
        quality_warnings.append("EMPTY_PAGE_DETECTED")
    if not tables:
        table_success = True
    else:
        table_success = not any("TABLE_EXTRACTION_FAILED" in item for item in quality_warnings)
    if not table_success:
        quality_warnings.append("TABLE_EXTRACTION_FAILED")
    if duplicate_ratio > max_duplicate:
        quality_warnings.append("DUPLICATE_BLOCK_DETECTED")

    if not combined.strip() and not blocks:
        status = "rejected"
    elif ratio >= max_garbled:
        status = "rejected" if bool(config.get("reject_on_garbled", False)) else "needs_manual_review"
    elif (text_density < min_density and ocr_confidence <= 0.0) or empty_ratio > max_empty:
        status = "needs_manual_review"
    elif quality_warnings or text_density < min_density or duplicate_ratio > max_duplicate:
        status = "warning"
    else:
        status = "accepted"

    return ExtractionQuality(
        text_density=round(text_density, 2),
        garbled_ratio=round(ratio, 4),
        ocr_confidence=round(float(ocr_confidence), 4),
        empty_page_ratio=round(empty_ratio, 4),
        table_extraction_success=table_success,
        duplicate_block_ratio=round(duplicate_ratio, 4),
        quality_status=status,
        warnings=list(dict.fromkeys(quality_warnings)),
    )
