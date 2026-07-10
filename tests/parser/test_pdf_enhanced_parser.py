from __future__ import annotations

import fitz

from src.libs.loader.parsed_document import Block
from src.libs.parser.pdf_parser import PdfParser


def _pdf(tmp_path, *, text="A searchable paragraph with enough text for density detection."):
    path = tmp_path / "document.pdf"
    document = fitz.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 100), text)
    document.save(path)
    document.close()
    return path


class _FailIfCalledOCR:
    def __init__(self):
        self.calls = 0

    def parse_page(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("text-layer PDF must not enter OCR")


class _FakeOCR:
    def __init__(self):
        self.calls = 0

    def parse_page(self, pdf_page, *, page_number, document_id):
        self.calls += 1
        return [
            Block(
                block_id=f"{document_id}_ocr_{page_number}",
                block_type="paragraph",
                text="OCR recovered text",
                page_number=page_number,
                bbox=(10.0, 10.0, 100.0, 30.0),
                confidence=0.91,
            )
        ], 0.91, []


def test_text_pdf_does_not_enter_ocr(tmp_path):
    fake = _FailIfCalledOCR()
    parsed = PdfParser(
        config={"enabled": True, "ocr": {"enabled": True, "min_text_density": 20}},
        ocr_parser=fake,
        extract_images=False,
    ).parse(_pdf(tmp_path))
    assert fake.calls == 0
    assert parsed.blocks
    assert parsed.extraction_quality.quality_status in {"accepted", "warning"}


def test_low_density_pdf_enters_fake_ocr(tmp_path):
    fake = _FakeOCR()
    parsed = PdfParser(
        config={"enabled": True, "ocr": {"enabled": True, "min_text_density": 20}},
        ocr_parser=fake,
        extract_images=False,
    ).parse(_pdf(tmp_path, text=""))
    assert fake.calls == 1
    assert any(block.text == "OCR recovered text" for block in parsed.blocks)
    assert parsed.extraction_quality.ocr_confidence == 0.91
