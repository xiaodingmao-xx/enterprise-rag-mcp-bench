"""Optional, structured document parsing enhancements.

Parser implementations are imported lazily so importing the package does not
load OCR, PDF, table, or layout engines.
"""

from src.libs.loader.parsed_document import (
    Block,
    ExtractionQuality,
    Image,
    Page,
    ParsedDocument,
    Table,
    TableCell,
)

__all__ = [
    "BaseParser",
    "BasicTextParser",
    "Block",
    "ExtractionQuality",
    "Image",
    "LayoutParser",
    "OCRParser",
    "Page",
    "ParsedDocument",
    "ParserFactory",
    "PdfParser",
    "Table",
    "TableCell",
    "TableParser",
]


def __getattr__(name: str):
    modules = {
        "BaseParser": "src.libs.parser.base",
        "BasicTextParser": "src.libs.parser.base",
        "LayoutParser": "src.libs.parser.layout_parser",
        "OCRParser": "src.libs.parser.ocr_parser",
        "ParserFactory": "src.libs.parser.parser_factory",
        "PdfParser": "src.libs.parser.pdf_parser",
        "TableParser": "src.libs.parser.table_parser",
    }
    module_name = modules.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = __import__(module_name, fromlist=[name])
    return getattr(module, name)
