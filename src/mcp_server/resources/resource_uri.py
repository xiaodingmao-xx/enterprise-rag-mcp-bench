"""Parser and validation for RAG MCP resource URIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import unquote, urlparse

ResourceKind = Literal["collection", "document", "chunk"]


class ResourceUriError(ValueError):
    """Raised when an MCP resource URI is invalid."""


@dataclass(frozen=True)
class ParsedResourceUri:
    """Parsed representation of a supported RAG resource URI."""

    resource_type: ResourceKind
    collection_name: str
    document_id: str | None = None
    chunk_id: str | None = None


def parse_resource_uri(uri: str) -> ParsedResourceUri:
    """Parse and validate a RAG MCP resource URI.

    Supported formats:
    - rag://collections/{collection_name}
    - rag://collections/{collection_name}/documents/{document_id}
    - rag://collections/{collection_name}/chunks/{chunk_id}
    """
    if not isinstance(uri, str) or not uri.strip():
        raise ResourceUriError("Resource URI must be a non-empty string")

    parsed = urlparse(uri)
    if parsed.scheme != "rag":
        raise ResourceUriError("Resource URI scheme must be 'rag'")
    if parsed.netloc != "collections":
        raise ResourceUriError("Resource URI must start with rag://collections/")
    if parsed.params or parsed.query or parsed.fragment:
        raise ResourceUriError("Resource URI must not contain params, query, or fragment")

    raw_segments = [segment for segment in parsed.path.split("/") if segment]
    if len(raw_segments) not in (1, 3):
        raise ResourceUriError("Resource URI path does not match a supported resource format")

    collection_name = _decode_segment(raw_segments[0], "collection_name")
    if len(raw_segments) == 1:
        return ParsedResourceUri(
            resource_type="collection",
            collection_name=collection_name,
        )

    child_type = raw_segments[1]
    child_id = _decode_segment(raw_segments[2], "resource_id")
    if child_type == "documents":
        return ParsedResourceUri(
            resource_type="document",
            collection_name=collection_name,
            document_id=child_id,
        )
    if child_type == "chunks":
        return ParsedResourceUri(
            resource_type="chunk",
            collection_name=collection_name,
            chunk_id=child_id,
        )

    raise ResourceUriError("Resource URI path child type must be 'documents' or 'chunks'")


def _decode_segment(raw_segment: str, label: str) -> str:
    if not raw_segment:
        raise ResourceUriError(f"{label} cannot be empty")

    lower = raw_segment.lower()
    if "%2f" in lower or "%5c" in lower:
        raise ResourceUriError("Resource URI path traversal is not allowed")

    decoded = unquote(raw_segment)
    if not decoded.strip():
        raise ResourceUriError(f"{label} cannot be empty")
    if decoded in {".", ".."} or ".." in decoded:
        raise ResourceUriError("Resource URI path traversal is not allowed")
    if "/" in decoded or "\\" in decoded:
        raise ResourceUriError("Resource URI segment must not contain path separators")

    return decoded
