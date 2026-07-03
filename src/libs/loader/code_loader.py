"""Source code document loader."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader


class CodeLoader(BaseLoader):
    """Load source code files while preserving the original text."""

    file_type = "code"
    language_by_extension = {
        ".py": "python",
        ".js": "javascript",
        ".java": "java",
    }

    def load(self, file_path: str | Path) -> Document:
        path = self._validate_file(file_path)
        suffix = path.suffix.lower()
        language = self.language_by_extension.get(suffix)
        if language is None:
            raise ValueError(f"Unsupported code file type: {path.suffix}")

        text = path.read_text(encoding="utf-8")
        metadata = {
            "source_path": str(path),
            "file_name": path.name,
            "file_type": self.file_type,
            "doc_type": self.file_type,
            "language": language,
            "title": path.name,
        }

        symbols = self._extract_symbols(text, language)
        if symbols:
            metadata["symbols"] = symbols

        return Document(
            id=self._document_id(path),
            text=text,
            metadata=metadata,
        )

    def _extract_symbols(self, text: str, language: str) -> list[str]:
        if language == "python":
            return self._extract_python_symbols(text)
        if language == "javascript":
            return self._extract_javascript_symbols(text)
        if language == "java":
            return self._extract_java_symbols(text)
        return []

    @staticmethod
    def _extract_python_symbols(text: str) -> list[str]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []

        symbols: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append(node.name)
        return sorted(set(symbols))

    @staticmethod
    def _extract_javascript_symbols(text: str) -> list[str]:
        patterns = [
            r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(",
            r"\bclass\s+([A-Za-z_$][\w$]*)\b",
            r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(",
            r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?function\b",
        ]
        return _extract_regex_symbols(text, patterns)

    @staticmethod
    def _extract_java_symbols(text: str) -> list[str]:
        patterns = [
            r"\b(?:class|interface|enum)\s+([A-Za-z_][\w]*)\b",
            r"\b(?:public|protected|private|static|final|synchronized|native|\s)+"
            r"[\w<>\[\], ?]+\s+([A-Za-z_][\w]*)\s*\(",
        ]
        return _extract_regex_symbols(text, patterns)


def _extract_regex_symbols(text: str, patterns: list[str]) -> list[str]:
    symbols: set[str] = set()
    for pattern in patterns:
        symbols.update(re.findall(pattern, text))
    return sorted(symbols)
