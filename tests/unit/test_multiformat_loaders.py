"""Unit tests for non-PDF document loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.libs.loader.code_loader import CodeLoader
from src.libs.loader.docx_loader import DocxLoader
from src.libs.loader.html_loader import HtmlLoader
from src.libs.loader.markdown_loader import MarkdownLoader
from src.libs.loader.text_loader import TextLoader


def test_markdown_loader_preserves_markdown_and_extracts_title(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Knowledge Guide\n\n- item", encoding="utf-8")

    document = MarkdownLoader().load(path)

    assert document.text == "# Knowledge Guide\n\n- item"
    assert document.metadata["title"] == "Knowledge Guide"
    assert document.metadata["doc_type"] == "markdown"


def test_text_loader_reads_plain_text(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("plain text body", encoding="utf-8")

    document = TextLoader().load(path)

    assert document.text == "plain text body"
    assert document.metadata["title"] == "notes.txt"
    assert document.metadata["doc_type"] == "text"


def test_html_loader_extracts_readable_text_and_removes_scripts(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>Page Title</title><script>hidden()</script></head>"
        "<body><h1>Heading</h1><p>Main body</p><style>.x{}</style></body></html>",
        encoding="utf-8",
    )

    document = HtmlLoader().load(path)

    assert document.metadata["title"] == "Page Title"
    assert "Heading" in document.text
    assert "Main body" in document.text
    assert "hidden" not in document.text


def test_code_loader_preserves_code_and_extracts_python_symbols(tmp_path: Path) -> None:
    path = tmp_path / "service.py"
    path.write_text(
        "class Service:\n"
        "    pass\n\n"
        "async def fetch():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    document = CodeLoader().load(path)

    assert "class Service" in document.text
    assert document.metadata["language"] == "python"
    assert document.metadata["symbols"] == ["Service", "fetch"]


def test_docx_loader_extracts_paragraphs_and_tables(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    path = tmp_path / "report.docx"
    source = docx.Document()
    source.core_properties.title = "Docx Report"
    source.add_paragraph("Intro paragraph")
    table = source.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Key"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Owner"
    table.cell(1, 1).text = "Platform"
    source.save(path)

    document = DocxLoader().load(path)

    assert "Intro paragraph" in document.text
    assert "| Key | Value |" in document.text
    assert "| Owner | Platform |" in document.text
    assert document.metadata["title"] == "Docx Report"
    assert document.metadata["doc_type"] == "docx"
