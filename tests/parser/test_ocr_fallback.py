from __future__ import annotations

import fitz

from src.libs.parser.pdf_parser import PdfParser


class _MissingOCR:
    def parse_page(self, *args, **kwargs):
        return [], 0.0, ["OCR_DEPENDENCY_MISSING"]


def test_missing_ocr_dependency_is_a_quality_warning(tmp_path):
    path = tmp_path / "scan.pdf"
    document = fitz.open()
    document.new_page()
    document.save(path)
    document.close()
    parsed = PdfParser(
        config={"enabled": True, "ocr": {"enabled": True, "min_text_density": 20}},
        ocr_parser=_MissingOCR(),
        extract_images=False,
    ).parse(path)
    assert "OCR_DEPENDENCY_MISSING" in parsed.extraction_quality.warnings
    assert parsed.extraction_quality.quality_status in {"warning", "needs_manual_review", "rejected"}


def test_enhanced_parser_disabled_keeps_basic_text_path(tmp_path):
    path = tmp_path / "basic.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 100), "Basic text remains available")
    document.save(path)
    document.close()
    parsed = PdfParser(config={"enabled": False}).parse(path)
    assert "Basic text remains available" in parsed.text
    assert "ENHANCED_PARSING_DISABLED" in parsed.extraction_quality.warnings
