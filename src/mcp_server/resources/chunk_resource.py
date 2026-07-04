"""Helpers for building chunk MCP resource payloads."""

from __future__ import annotations

from typing import Any

from src.mcp_server.resources.document_resource import (
    filter_metadata,
    get_source,
    normalise_page_range,
    tags_from_metadata,
)


def build_chunk_payload(collection_name: str, record: dict[str, Any]) -> dict[str, Any]:
    """Build JSON-serialisable chunk resource payload."""
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    page_range = normalise_page_range(metadata)
    page = metadata.get("page")
    if page is None and page_range:
        page = page_range[0]

    return {
        "type": "chunk",
        "collection_name": collection_name,
        "chunk_id": str(record.get("id", "")),
        "text": str(record.get("text", "")),
        "source": get_source(metadata),
        "page": page,
        "page_range": page_range,
        "title": metadata.get("title"),
        "summary": metadata.get("summary"),
        "tags": tags_from_metadata(metadata),
        "metadata": filter_metadata(metadata),
    }
