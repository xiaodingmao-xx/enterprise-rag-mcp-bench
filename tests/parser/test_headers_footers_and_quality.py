from __future__ import annotations

import fitz

from src.libs.loader.document_quality import assess_parsed_document_quality
from src.libs.parser.pdf_parser import PdfParser


def test_repeated_header_footer_are_excluded_from_body_text(tmp_path):
    path = tmp_path / "header_footer.pdf"
    document = fitz.open()
    for page_number in range(1, 4):
        page = document.new_page()
        page.insert_text((72, 40), "CONFIDENTIAL HEADER")
        page.insert_text((72, 200), f"Unique body paragraph {page_number}")
        page.insert_text((72, 760), f"Page {page_number}")
    document.save(path)
    document.close()

    parsed = PdfParser(config={"enabled": True}, extract_images=False).parse(path)
    assert parsed.headers
    assert parsed.footers
    assert "CONFIDENTIAL HEADER" not in parsed.text
    assert "Unique body paragraph 2" in parsed.text
    assert parsed.metadata["excluded_headers"]


def test_garbled_quality_is_not_silently_accepted():
    from src.libs.loader.parsed_document import Block, Page

    page = Page(page_number=1, text="� � � ÃÂÐÑ", text_density=10, blocks=[])
    quality = assess_parsed_document_quality(
        [page],
        [Block("b", "paragraph", page.text, 1)],
        [],
        config={"max_garbled_ratio": 0.2, "reject_on_garbled": False},
    )
    assert quality.quality_status in {"warning", "needs_manual_review", "rejected"}
    assert "GARBLE_TEXT_DETECTED" in quality.warnings
