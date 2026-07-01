"""Lightweight PDF quality gate for ingestion.

The checker samples the first few pages of a PDF before the expensive loader
stage. It rejects files whose text layer is empty or mostly unreadable, which
prevents low-quality chunks from polluting the vector and keyword indexes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict
import unicodedata

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

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


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
