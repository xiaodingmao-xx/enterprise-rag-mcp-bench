"""Factory for selecting document loaders by file extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Type

from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.code_loader import CodeLoader
from src.libs.loader.docx_loader import DocxLoader
from src.libs.loader.html_loader import HtmlLoader
from src.libs.loader.markdown_loader import MarkdownLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.text_loader import TextLoader


DEFAULT_SUPPORTED_EXTENSIONS = (
    ".pdf",
    ".md",
    ".txt",
    ".html",
    ".htm",
    ".docx",
    ".py",
    ".js",
    ".java",
)


class UnsupportedFileTypeError(ValueError):
    """Raised when a file extension is not supported by LoaderFactory."""


class LoaderFactory:
    """Create a document loader based on a file extension."""

    _loader_by_extension: Dict[str, Type[BaseLoader]] = {
        ".pdf": PdfLoader,
        ".md": MarkdownLoader,
        ".txt": TextLoader,
        ".html": HtmlLoader,
        ".htm": HtmlLoader,
        ".docx": DocxLoader,
        ".py": CodeLoader,
        ".js": CodeLoader,
        ".java": CodeLoader,
    }

    @classmethod
    def get_loader(
        cls,
        file_path: str | Path,
        settings: Any = None,
        **kwargs: Any,
    ) -> BaseLoader:
        """Return a loader instance for *file_path*.

        Args:
            file_path: Path whose suffix determines the loader.
            settings: Optional Settings object. If it defines
                ``ingestion.supported_extensions``, that allowlist is honored.
            **kwargs: Loader-specific keyword arguments. Currently used for
                PdfLoader image extraction settings.
        """

        path = Path(file_path)
        suffix = path.suffix.lower()
        supported_extensions = cls.get_supported_extensions(settings)

        if suffix not in supported_extensions:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {suffix or '<none>'}. "
                f"Supported: {', '.join(supported_extensions)}"
            )

        loader_cls = cls._loader_by_extension.get(suffix)
        if loader_cls is None:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {suffix or '<none>'}. "
                f"Supported: {', '.join(cls.supported_extensions())}"
            )

        if loader_cls is PdfLoader:
            kwargs.setdefault("settings", settings)
            return loader_cls(**kwargs)
        return loader_cls()

    @classmethod
    def create_loader(
        cls,
        file_path: str | Path,
        settings: Any = None,
        **kwargs: Any,
    ) -> BaseLoader:
        """Alias for get_loader."""

        return cls.get_loader(file_path=file_path, settings=settings, **kwargs)

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Return all extensions known by the factory."""

        return list(DEFAULT_SUPPORTED_EXTENSIONS)

    @classmethod
    def get_supported_extensions(cls, settings: Any = None) -> list[str]:
        """Return supported extensions from settings or factory defaults."""

        configured = None
        ingestion = getattr(settings, "ingestion", None)
        if ingestion is not None:
            if isinstance(ingestion, dict):
                configured = ingestion.get("supported_extensions")
            else:
                configured = getattr(ingestion, "supported_extensions", None)

        extensions = _normalise_extensions(
            configured if configured is not None else DEFAULT_SUPPORTED_EXTENSIONS
        )
        known = set(cls._loader_by_extension)
        return [extension for extension in extensions if extension in known]


def _normalise_extensions(extensions: Iterable[str]) -> list[str]:
    normalised: list[str] = []
    for extension in extensions:
        ext = str(extension).strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext not in normalised:
            normalised.append(ext)
    return normalised
