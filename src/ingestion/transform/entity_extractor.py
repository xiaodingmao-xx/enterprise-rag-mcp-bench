"""Rule-based entity extraction for metadata enrichment."""

from __future__ import annotations

import re
from typing import Any


def _dedupe(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        value = str(item).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
        if len(output) >= limit:
            break
    return output


def extract_entities(
    text: str,
    metadata: dict[str, Any] | None = None,
    max_entities: int = 15,
) -> list[str]:
    """Extract common enterprise/document entities from chunk text.

    The extractor intentionally stays lightweight and dependency-free. It
    focuses on entities that are useful for RAG filtering and debugging:
    versions, error codes, API paths, code identifiers, configuration keys,
    and uppercase technical abbreviations.
    """

    if not text or max_entities <= 0:
        return []

    candidates: list[str] = []

    candidates.extend(re.findall(r"\bv?\d+\.\d+(?:\.\d+)?(?:[-_][A-Za-z0-9]+)?\b", text))
    candidates.extend(re.findall(r"\b(?:HTTP\s*)?[45]\d{2}\b", text, flags=re.IGNORECASE))
    candidates.extend(re.findall(r"\bError\s*[A-Za-z]?\d{3,5}\b", text, flags=re.IGNORECASE))
    candidates.extend(re.findall(r"\bE\d{3,5}\b", text))
    candidates.extend(re.findall(r"(?<!\w)/(?:api|v\d+|[A-Za-z0-9_-]+)(?:/[A-Za-z0-9_{}.-]+)+", text))
    candidates.extend(
        re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*\s*(?:=|:)\s*(?:[A-Za-z0-9_.:/-]+)",
            text,
        )
    )
    candidates.extend(re.findall(r"\b[A-Z][A-Za-z0-9]+(?:Service|Client|Config|Factory|Controller)\b", text))
    candidates.extend(re.findall(r"\b[a-zA-Z_][A-Za-z0-9_]*\(\)", text))
    candidates.extend(re.findall(r"\b[A-Z]{2,}(?:\d+)?\b", text))

    metadata = metadata or {}
    for key in ("language", "file_type", "doc_type"):
        if metadata.get(key):
            candidates.append(str(metadata[key]))
    for symbol in metadata.get("symbols", []) if isinstance(metadata.get("symbols"), list) else []:
        candidates.append(str(symbol))

    cleaned = [item.strip().rstrip(".,;:") for item in candidates]
    return _dedupe(cleaned, max_entities)
