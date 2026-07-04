"""Unit tests for MCP RAG resource URI parsing."""

from __future__ import annotations

import pytest

from src.mcp_server.resources.resource_uri import ResourceUriError, parse_resource_uri


def test_parse_collection_uri() -> None:
    parsed = parse_resource_uri("rag://collections/default")

    assert parsed.resource_type == "collection"
    assert parsed.collection_name == "default"
    assert parsed.document_id is None
    assert parsed.chunk_id is None


def test_parse_document_uri() -> None:
    parsed = parse_resource_uri("rag://collections/default/documents/doc_123")

    assert parsed.resource_type == "document"
    assert parsed.collection_name == "default"
    assert parsed.document_id == "doc_123"


def test_parse_chunk_uri() -> None:
    parsed = parse_resource_uri("rag://collections/default/chunks/chunk%3A%3A001")

    assert parsed.resource_type == "chunk"
    assert parsed.collection_name == "default"
    assert parsed.chunk_id == "chunk::001"


def test_rejects_invalid_scheme() -> None:
    with pytest.raises(ResourceUriError, match="scheme"):
        parse_resource_uri("http://collections/default")


def test_rejects_invalid_path() -> None:
    with pytest.raises(ResourceUriError, match="path"):
        parse_resource_uri("rag://collections/default/unknown/item")


def test_rejects_path_traversal_segment() -> None:
    with pytest.raises(ResourceUriError, match="traversal"):
        parse_resource_uri("rag://collections/default/documents/..%2Fsecret")
