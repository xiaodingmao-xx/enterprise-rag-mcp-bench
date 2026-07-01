"""Tests for the PDF document quality preflight checker."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.libs.loader.document_quality import PdfQualityChecker

fitz = pytest.importorskip("fitz")


def _write_pdf(path: Path, text: str | None) -> None:
    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


class TestPdfQualityChecker:
    def test_clean_text_pdf_passes(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "clean.pdf"
        _write_pdf(
            pdf_path,
            "This is a clean test document with enough recognizable text. " * 5,
        )

        checker = PdfQualityChecker()
        report = checker.check(pdf_path)

        assert report.checked is True
        assert report.passed is True
        assert report.effective_char_ratio >= 0.8
        assert report.recognizable_text_density >= 20

    def test_blank_pdf_is_rejected(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "blank.pdf"
        _write_pdf(pdf_path, None)

        checker = PdfQualityChecker()
        report = checker.check(pdf_path)

        assert report.checked is True
        assert report.passed is False
        assert report.reason in {
            "low_effective_char_ratio",
            "low_recognizable_text_density",
        }

    def test_low_effective_ratio_is_rejected(self, tmp_path: Path) -> None:
        checker = PdfQualityChecker()

        report = checker.evaluate_text_sample(
            "valid" + ("\ufffd" * 30),
            file_path=tmp_path / "garbled.pdf",
            page_count=1,
            sampled_pages=1,
        )

        assert report.passed is False
        assert report.reason == "low_effective_char_ratio"

    def test_non_pdf_is_passed_through(self, tmp_path: Path) -> None:
        txt_path = tmp_path / "note.txt"
        txt_path.write_text("hello", encoding="utf-8")

        checker = PdfQualityChecker()
        report = checker.check(txt_path)

        assert report.checked is False
        assert report.passed is True
        assert report.reason == "non_pdf"
