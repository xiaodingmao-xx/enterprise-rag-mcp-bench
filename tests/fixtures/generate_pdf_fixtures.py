"""Generate small, deterministic PDFs for enhanced-parser tests.

The generator uses only reportlab/Pillow and never downloads or requires an
OCR engine.  Tests can call individual functions with a temporary directory.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict

try:
    from reportlab.lib.colors import lightblue
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
except ImportError:  # reportlab is a development/fixture dependency
    canvas = None
    letter = None
    ImageReader = None
    lightblue = "lightblue"


def create_text_pdf(output_dir: str | Path) -> Path:
    _require_reportlab()
    path = _path(output_dir, "text_layers.pdf")
    if path.exists():
        return path
    c = canvas.Canvas(str(path), pagesize=letter)
    for page_number in range(1, 3):
        c.setFont("Helvetica-Bold", 18)
        c.drawString(72, 720, "Quarterly Report")
        c.setFont("Helvetica", 11)
        c.drawString(72, 680, f"Page body paragraph {page_number} with searchable text.")
        c.drawString(72, 660, "The parser should preserve page numbers and reading order.")
        c.showPage()
    c.save()
    return path


def create_scanned_pdf(output_dir: str | Path) -> Path:
    _require_reportlab()
    path = _path(output_dir, "scanned.pdf")
    if path.exists():
        return path
    image = _text_image("Scanned enterprise document\nOCR fallback fixture")
    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawImage(ImageReader(image), 72, 500, width=468, height=180)
    c.showPage()
    c.save()
    return path


def create_table_pdf(output_dir: str | Path) -> Path:
    _require_reportlab()
    path = _path(output_dir, "table.pdf")
    if path.exists():
        return path
    c = canvas.Canvas(str(path), pagesize=letter)
    x0, y0, cell_w, cell_h = 72, 560, 150, 28
    font_name, values = _table_font_and_values()
    c.setFont(font_name, 10)
    for row_index, row in enumerate(values):
        for col_index, value in enumerate(row):
            x, y = x0 + col_index * cell_w, y0 - row_index * cell_h
            c.rect(x, y - cell_h, cell_w, cell_h, stroke=1, fill=0)
            c.drawString(x + 8, y - 18, value)
    c.showPage()
    c.save()
    return path


def create_image_pdf(output_dir: str | Path) -> Path:
    _require_reportlab()
    path = _path(output_dir, "image.pdf")
    if path.exists():
        return path
    image = _solid_image()
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica", 11)
    c.drawString(72, 700, "Figure fixture")
    c.drawImage(ImageReader(image), 72, 500, width=180, height=100)
    c.drawString(72, 470, "Figure 1: A blue rectangle used for image association.")
    c.showPage()
    c.save()
    return path


def create_header_footer_pdf(output_dir: str | Path) -> Path:
    _require_reportlab()
    path = _path(output_dir, "header_footer.pdf")
    if path.exists():
        return path
    c = canvas.Canvas(str(path), pagesize=letter)
    for page_number in range(1, 4):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(72, 750, "CONFIDENTIAL - ACME REPORT")
        c.setFont("Helvetica", 11)
        c.drawString(72, 690, f"Different body text for page {page_number}.")
        c.setFont("Helvetica", 9)
        c.drawString(72, 36, f"Page {page_number}")
        c.showPage()
    c.save()
    return path


def create_cross_page_table_pdf(output_dir: str | Path) -> Path:
    _require_reportlab()
    path = _path(output_dir, "cross_page_table.pdf")
    if path.exists():
        return path
    c = canvas.Canvas(str(path), pagesize=letter)
    for page_number, rows in enumerate(
        [
            [["Item", "Qty", "Owner"], ["A", "1", "Lee"], ["B", "2", "Wang"]],
            [["Item", "Qty", "Owner"], ["C", "3", "Chen"], ["D", "4", "Zhao"]],
        ],
        start=1,
    ):
        x0, y0, cell_w, cell_h = 72, 560, 150, 28
        c.setFont("Helvetica", 10)
        for row_index, row in enumerate(rows):
            for col_index, value in enumerate(row):
                x, y = x0 + col_index * cell_w, y0 - row_index * cell_h
                c.rect(x, y - cell_h, cell_w, cell_h, stroke=1, fill=0)
                c.drawString(x + 8, y - 18, value)
        c.drawString(72, 700, f"Table continuation page {page_number}")
        c.showPage()
    c.save()
    return path


def create_garbled_pdf(output_dir: str | Path) -> Path:
    _require_reportlab()
    path = _path(output_dir, "garbled.pdf")
    if path.exists():
        return path
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica", 11)
    c.drawString(72, 700, "��� ���� ÃÂÐÑ §¤※※※")
    c.drawString(72, 680, "� � � ���")
    c.showPage()
    c.save()
    return path


def generate_all(output_dir: str | Path) -> Dict[str, Path]:
    return {
        "text": create_text_pdf(output_dir),
        "scanned": create_scanned_pdf(output_dir),
        "table": create_table_pdf(output_dir),
        "image": create_image_pdf(output_dir),
        "header_footer": create_header_footer_pdf(output_dir),
        "cross_page_table": create_cross_page_table_pdf(output_dir),
        "garbled": create_garbled_pdf(output_dir),
    }


def _path(output_dir: str | Path, name: str) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def _require_reportlab() -> None:
    if canvas is None:
        raise RuntimeError("reportlab is required to generate PDF fixtures; install the dev extra")


def _table_font_and_values() -> tuple[str, list[list[str]]]:
    """Use ReportLab's built-in CJK font when available, with ASCII fallback."""

    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont

        font_name = "STSong-Light"
        if font_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(UnicodeCIDFont(font_name))
        return font_name, [["姓名", "数量", "状态"], ["甲", "12", "就绪"], ["乙", "7", "复核"]]
    except Exception:
        return "Helvetica", [["Name", "Count", "Status"], ["Alpha", "12", "Ready"], ["Beta", "7", "Review"]]


def _text_image(text: str) -> BytesIO:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (900, 350), "white")
    draw = ImageDraw.Draw(image)
    draw.multiline_text((40, 120), text, fill="black", spacing=12)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def _solid_image() -> BytesIO:
    from PIL import Image

    image = Image.new("RGB", (180, 100), lightblue)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


if __name__ == "__main__":
    generated = generate_all(Path(__file__).parent / "generated")
    for name, path in generated.items():
        print(f"{name}: {path}")
