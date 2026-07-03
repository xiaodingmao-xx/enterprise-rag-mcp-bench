"""Unit tests for LoaderFactory."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.libs.loader.code_loader import CodeLoader
from src.libs.loader.docx_loader import DocxLoader
from src.libs.loader.html_loader import HtmlLoader
from src.libs.loader.loader_factory import LoaderFactory, UnsupportedFileTypeError
from src.libs.loader.markdown_loader import MarkdownLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.text_loader import TextLoader


@pytest.mark.parametrize(
    ("file_name", "expected_type"),
    [
        ("document.pdf", PdfLoader),
        ("README.md", MarkdownLoader),
        ("notes.txt", TextLoader),
        ("page.html", HtmlLoader),
        ("page.htm", HtmlLoader),
        ("report.docx", DocxLoader),
        ("script.py", CodeLoader),
        ("app.js", CodeLoader),
        ("Service.java", CodeLoader),
    ],
)
def test_loader_factory_returns_loader_by_extension(
    file_name: str,
    expected_type: type,
) -> None:
    loader = LoaderFactory.get_loader(Path(file_name))

    assert isinstance(loader, expected_type)


def test_loader_factory_extension_is_case_insensitive() -> None:
    loader = LoaderFactory.get_loader(Path("README.MD"))

    assert isinstance(loader, MarkdownLoader)


def test_loader_factory_unknown_extension_raises_clear_error() -> None:
    with pytest.raises(UnsupportedFileTypeError, match=r"Unsupported file type: \.xyz"):
        LoaderFactory.get_loader(Path("archive.xyz"))


def test_loader_factory_supports_configured_allowlist() -> None:
    class Ingestion:
        supported_extensions = [".md"]

    class Settings:
        ingestion = Ingestion()

    loader = LoaderFactory.get_loader(Path("README.md"), settings=Settings())
    assert isinstance(loader, MarkdownLoader)

    with pytest.raises(UnsupportedFileTypeError, match=r"Unsupported file type: \.txt"):
        LoaderFactory.get_loader(Path("notes.txt"), settings=Settings())
