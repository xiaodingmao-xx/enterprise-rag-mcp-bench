"""Helpers for building collection MCP resource payloads."""

from __future__ import annotations

from typing import Any

from src.mcp_server.resources.document_resource import (
    COMMON_METADATA_FIELDS,
    derive_document_id,
)


def build_collection_payload(
    collection_name: str,
    chunk_count: int,
    records: list[dict[str, Any]],
    sampled: bool = False,
) -> dict[str, Any]:
    """Build JSON-serialisable collection resource payload.

    Chroma exposes total chunk count directly. Distinct document count and
    metadata-field discovery are best-effort because they require scanning
    records; callers may pass a capped sample for large collections.
    """
    document_ids = {derive_document_id(record) for record in records}
    metadata_fields = set(COMMON_METADATA_FIELDS)
    for record in records:
        metadata = record.get("metadata", {})
        if isinstance(metadata, dict):
            metadata_fields.update(str(key) for key in metadata)

    return {
        "type": "collection",
        "collection_name": collection_name,
        "document_count": len(document_ids),
        "chunk_count": chunk_count,
        "available_metadata_fields": sorted(metadata_fields),
        "stats_are_best_effort": sampled,
    }
