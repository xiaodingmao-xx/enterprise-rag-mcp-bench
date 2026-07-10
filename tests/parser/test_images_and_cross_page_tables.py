from __future__ import annotations

from io import BytesIO

import fitz
from PIL import Image

from src.libs.parser.pdf_parser import PdfParser


def test_image_has_page_and_bbox(tmp_path):
    path = tmp_path / "image.pdf"
    image = Image.new("RGB", (80, 50), "blue")
    stream = BytesIO()
    image.save(stream, format="PNG")
    document = fitz.open()
    page = document.new_page()
    page.insert_image(fitz.Rect(72, 120, 220, 220), stream=stream.getvalue())
    page.insert_text((72, 250), "Figure 1: Blue rectangle")
    document.save(path)
    document.close()
    parsed = PdfParser(config={"enabled": True}, extract_images=False).parse(path)
    assert parsed.images
    assert parsed.images[0].page_number == 1
    assert parsed.images[0].bbox is not None


def test_adjacent_same_shape_tables_are_marked_as_continuation(tmp_path):
    path = tmp_path / "continuation.pdf"
    document = fitz.open()
    for page_number in range(2):
        page = document.new_page()
        values = [["Item", "Qty", "Owner"], ["A", "1", "Lee"]]
        for row_index, row in enumerate(values):
            for col_index, value in enumerate(row):
                rect = fitz.Rect(70 + col_index * 150, 100 + row_index * 30, 220 + col_index * 150, 130 + row_index * 30)
                page.draw_rect(rect)
                page.insert_text((rect.x0 + 5, rect.y0 + 20), value)
    document.save(path)
    document.close()
    parsed = PdfParser(config={"enabled": True}, extract_images=False).parse(path)
    assert len(parsed.tables) >= 2
    assert any(table.metadata.get("possible_continuation") for table in parsed.tables)
    assert "CROSS_PAGE_TABLE_CANDIDATE" in parsed.extraction_quality.warnings
