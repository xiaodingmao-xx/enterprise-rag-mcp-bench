"""Stable chunk ID generation utilities."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple
import hashlib
import re
import unicodedata


def short_hash(value: str, length: int = 8) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()[:length]


def resolve_doc_hash(*, doc_id: str, metadata: dict[str, Any], text: str) -> str:
    """Resolve a stable document hash for chunk IDs."""
    for key in ("doc_hash", "source_hash", "file_hash"):
        value = metadata.get(key)
        if value:
            return str(value)
    source_path = metadata.get("source_path")
    if source_path:
        return short_hash(str(source_path), 16)
    if doc_id:
        return short_hash(str(doc_id), 16)
    return short_hash(text, 16)


def slugify_section(path: Optional[Iterable[Any]]) -> str:
    """Build a safe section slug from heading/section path."""
    values = [str(item).strip() for item in (path or []) if str(item).strip()]
    if not values:
        return "root"

    raw = "-".join(values)
    normalised = unicodedata.normalize("NFKD", raw)
    ascii_text = normalised.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_text).strip("-").lower()
    if slug:
        return slug[:80]
    return short_hash(raw, 12)


def generate_chunk_id(
    *,
    doc_hash: str,
    text: str,
    chunk_index: int,
    page_range: Optional[Tuple[int, int]] = None,
    section_path: Optional[Iterable[Any]] = None,
    heading_path: Optional[Iterable[Any]] = None,
    chunk_level: Optional[str] = None,
) -> str:
    """Generate a deterministic, traceable chunk ID."""
    level = f"::{chunk_level}" if chunk_level else ""
    if page_range is not None:
        page_start = int(page_range[0])
        section_slug = slugify_section(heading_path or section_path)
        return f"{doc_hash}::p{page_start:03d}::sec{section_slug}{level}::c{chunk_index:04d}"
    return f"{doc_hash}{level}::c{chunk_index:04d}::{short_hash(text)}"
