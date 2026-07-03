"""
Loader Module.

This package contains document loader components:
- Base loader class
- PDF loader
- Markdown/Text/HTML/DOCX/Code loaders
- Loader factory
- File integrity checker
"""

from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.code_loader import CodeLoader
from src.libs.loader.docx_loader import DocxLoader
from src.libs.loader.html_loader import HtmlLoader
from src.libs.loader.loader_factory import (
    DEFAULT_SUPPORTED_EXTENSIONS,
    LoaderFactory,
    UnsupportedFileTypeError,
)
from src.libs.loader.markdown_loader import MarkdownLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.text_loader import TextLoader
from src.libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker
from src.libs.loader.document_quality import (
    DOCUMENT_QUALITY_REJECTION_MESSAGE,
    DocumentQualityReport,
    PdfQualityChecker,
)

__all__ = [
    "BaseLoader",
    "CodeLoader",
    "DocxLoader",
    "HtmlLoader",
    "LoaderFactory",
    "MarkdownLoader",
    "PdfLoader",
    "TextLoader",
    "UnsupportedFileTypeError",
    "DEFAULT_SUPPORTED_EXTENSIONS",
    "FileIntegrityChecker",
    "SQLiteIntegrityChecker",
    "DOCUMENT_QUALITY_REJECTION_MESSAGE",
    "DocumentQualityReport",
    "PdfQualityChecker",
]
