"""Helpers for building document MCP resource payloads."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

COMMON_METADATA_FIELDS = [
    "source",
    "source_path",
    "source_ref",
    "doc_id",
    "chunk_id",
    "chunk_index",
    "page",
    "page_range",
    "page_range_text",
    "title",
    "summary",
    "tags",
]


def derive_document_id(record: dict[str, Any]) -> str:
    """Return a stable document id for a vector-store record."""
    metadata = _metadata(record)
    for key in ("doc_id", "document_id", "source_ref"):
        value = metadata.get(key)
        if value:
            return str(value)

    source = get_source(metadata)
    if source:
        return _stable_hash_id("doc", source)

    return _stable_hash_id("doc", str(record.get("id", "")))


def build_document_payload(
    collection_name: str,
    document_id: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build JSON-serialisable document resource payload."""
    first = records[0] if records else {}
    metadata = _metadata(first)
    source = get_source(metadata)
    page_range = merge_page_ranges([normalise_page_range(_metadata(record)) for record in records])

    return {
        "type": "document",
        "collection_name": collection_name,
        "document_id": document_id,
        "source": source,
        "title": metadata.get("title") or _title_from_source(source),
        "summary": metadata.get("summary") or _preview(first.get("text", "")),
        "chunk_count": len(records),
        "page_range": page_range,
        "metadata": filter_metadata(metadata),
    }


def get_source(metadata: dict[str, Any]) -> str | None:
    for key in ("source_path", "source", "source_ref"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def tags_from_metadata(metadata: dict[str, Any]) -> list[str]:
    tags = metadata.get("tags")
    if isinstance(tags, list):
        return [str(tag) for tag in tags if str(tag).strip()]
    if isinstance(tags, str):
        return [tag.strip() for tag in tags.split(",") if tag.strip()]
    return []


def normalise_page_range(metadata: dict[str, Any]) -> list[int] | None:
    page_range = metadata.get("page_range")
    parsed = _parse_page_range_value(page_range)
    if parsed is not None:
        return parsed

    page_range_text = metadata.get("page_range_text")
    parsed = _parse_page_range_value(page_range_text)
    if parsed is not None:
        return parsed

    page = metadata.get("page")
    if isinstance(page, int):
        return [page, page]
    if isinstance(page, str) and page.isdigit():
        parsed_page = int(page)
        return [parsed_page, parsed_page]

    return None


def merge_page_ranges(ranges: list[list[int] | None]) -> list[int] | None:
    valid = [page_range for page_range in ranges if page_range]
    if not valid:
        return None
    return [min(item[0] for item in valid), max(item[1] for item in valid)]


def filter_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    excluded = {"text"}
    return {key: value for key, value in metadata.items() if key not in excluded}


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _stable_hash_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _parse_page_range_value(value: Any) -> list[int] | None:
    if isinstance(value, dict):
        start = value.get("start") or value.get("page_start")
        end = value.get("end") or value.get("page_end") or start
        if isinstance(start, int) and isinstance(end, int):
            return [start, end]

    if isinstance(value, (list, tuple)) and value:
        numbers = [int(item) for item in value if isinstance(item, int)]
        if numbers:
            return [min(numbers), max(numbers)]

    if isinstance(value, str):
        numbers = [int(match) for match in re.findall(r"\d+", value)]
        if numbers:
            return [min(numbers), max(numbers)]

    return None


def _preview(text: Any, limit: int = 300) -> str:
    preview = " ".join(str(text or "").split())
    if len(preview) <= limit:
        return preview
    return preview[: limit - 3] + "..."


def _title_from_source(source: str | None) -> str | None:
    if not source:
        return None
    stem = Path(source).stem
    return stem.replace("_", " ").replace("-", " ").title() if stem else None
