from __future__ import annotations

import fitz

from src.libs.parser.pdf_parser import PdfParser
from src.libs.parser.table_parser import TableParser


def _table_pdf(tmp_path):
    path = tmp_path / "table.pdf"
    document = fitz.open()
    page = document.new_page()
    x0, y0, width, height = 70, 100, 150, 30
    values = [["Name", "Qty", "Status"], ["Alpha", "1", "Ready"], ["Beta", "2", "Review"]]
    for row_index, row in enumerate(values):
        for col_index, value in enumerate(row):
            rect = fitz.Rect(
                x0 + col_index * width,
                y0 + row_index * height,
                x0 + (col_index + 1) * width,
                y0 + (row_index + 1) * height,
            )
            page.draw_rect(rect)
            page.insert_text((rect.x0 + 5, rect.y0 + 20), value)
    document.save(path)
    document.close()
    return path


def test_table_parser_preserves_structure_and_coordinates(tmp_path):
    parsed = PdfParser(
        config={"enabled": True, "tables": {"enabled": True}},
        extract_images=False,
    ).parse(_table_pdf(tmp_path))
    assert parsed.tables
    table = parsed.tables[0]
    assert table.markdown
    assert table.rows
    assert table.cells
    assert table.cells[0].bbox is not None
    assert table.extraction_method


def test_fake_rows_have_markdown_and_plain_text():
    table = TableParser().from_rows(
        [["A", "B"], ["1", "2"]],
        page_number=2,
        document_id="doc1",
    )
    assert "| A | B |" in table.markdown
    assert "1\t2" in table.plain_text
