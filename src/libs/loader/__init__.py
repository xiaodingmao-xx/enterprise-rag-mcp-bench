"""
Loader Module.

This package contains document loader components:
- Base loader class
- PDF loader
- File integrity checker
"""

from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker
from src.libs.loader.document_quality import (
    DOCUMENT_QUALITY_REJECTION_MESSAGE,
    DocumentQualityReport,
    PdfQualityChecker,
)

__all__ = [
    "BaseLoader",
    "PdfLoader",
    "FileIntegrityChecker",
    "SQLiteIntegrityChecker",
    "DOCUMENT_QUALITY_REJECTION_MESSAGE",
    "DocumentQualityReport",
    "PdfQualityChecker",
]
