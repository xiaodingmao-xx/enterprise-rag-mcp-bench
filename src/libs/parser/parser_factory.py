"""Factory selecting enhanced parsers without changing LoaderFactory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.libs.parser.base import BaseParser, BasicTextParser
from src.libs.parser.pdf_parser import PdfParser


class ParserFactory:
    @classmethod
    def get_parser(cls, file_path: str | Path, settings: Any = None, **kwargs: Any) -> BaseParser:
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            config = cls._enhanced_pdf_config(settings)
            return PdfParser(config=config, **kwargs)
        return BasicTextParser()

    @classmethod
    def create_parser(cls, file_path: str | Path, settings: Any = None, **kwargs: Any) -> BaseParser:
        return cls.get_parser(file_path, settings=settings, **kwargs)

    @classmethod
    def parse(cls, file_path: str | Path, settings: Any = None, **kwargs: Any):
        parser = cls.get_parser(file_path, settings=settings, **kwargs)
        return parser.parse(file_path)

    @staticmethod
    def _enhanced_pdf_config(settings: Any) -> dict[str, Any]:
        ingestion = _value(settings, "ingestion", {})
        parsing = _value(ingestion, "parsing", {})
        enhanced = _value(parsing, "enhanced_pdf", {})
        if not isinstance(enhanced, Mapping):
            enhanced = {}
        config = dict(enhanced)
        config.setdefault("enabled", True)
        return config


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)
