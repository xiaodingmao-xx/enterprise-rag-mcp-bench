"""Unit tests for MCP resource resolution against Chroma."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.libs.vector_store.chroma_store import ChromaStore
from src.mcp_server.resources import ResourceResolutionError, ResourceResolver


@pytest.fixture()
def resource_settings(tmp_path: Any) -> SimpleNamespace:
    return SimpleNamespace(
        vector_store=SimpleNamespace(
            provider="chroma",
            collection_name="resolver_test",
            persist_directory=str(tmp_path),
        )
    )


@pytest.fixture()
def populated_collection(resource_settings: SimpleNamespace) -> str:
    collection_name = "resolver_test"
    store = ChromaStore(settings=resource_settings, collection_name=collection_name)
    try:
        store.upsert(
            [
                {
                    "id": "chunk-one",
                    "vector": [0.1, 0.2, 0.3],
                    "metadata": {
                        "doc_id": "doc-alpha",
                        "source": "docs/example.pdf",
                        "title": "Example Document",
                        "summary": "A short summary.",
                        "tags": "rag,mcp",
                        "page": 1,
                        "chunk_id": "chunk-one",
                    },
                },
                {
                    "id": "chunk-two",
                    "vector": [0.2, 0.3, 0.4],
                    "metadata": {
                        "doc_id": "doc-alpha",
                        "source": "docs/example.pdf",
                        "page": 2,
                        "chunk_id": "chunk-two",
                    },
                },
            ]
        )
        yield collection_name
    finally:
        try:
            store.client.delete_collection(name=collection_name)
        except Exception:
            pass
        store.close()


def test_list_resource_descriptors_includes_collection_and_document(
    resource_settings: SimpleNamespace,
    populated_collection: str,
) -> None:
    resolver = ResourceResolver(
        settings=resource_settings,
        documents_per_collection=100,
    )

    descriptors = resolver.list_resource_descriptors()
    uris = {descriptor.uri for descriptor in descriptors}

    assert f"rag://collections/{populated_collection}" in uris
    assert f"rag://collections/{populated_collection}/documents/doc-alpha" in uris


def test_read_collection_resource(
    resource_settings: SimpleNamespace,
    populated_collection: str,
) -> None:
    resolver = ResourceResolver(settings=resource_settings)

    payload = resolver.read_resource(f"rag://collections/{populated_collection}")

    assert payload["type"] == "collection"
    assert payload["collection_name"] == populated_collection
    assert payload["chunk_count"] == 2
    assert payload["document_count"] == 1


def test_read_document_resource(
    resource_settings: SimpleNamespace,
    populated_collection: str,
) -> None:
    resolver = ResourceResolver(settings=resource_settings)

    payload = resolver.read_resource(
        f"rag://collections/{populated_collection}/documents/doc-alpha"
    )

    assert payload["type"] == "document"
    assert payload["document_id"] == "doc-alpha"
    assert payload["chunk_count"] == 2
    assert payload["source"] == "docs/example.pdf"


def test_read_chunk_resource(
    resource_settings: SimpleNamespace,
    populated_collection: str,
) -> None:
    resolver = ResourceResolver(settings=resource_settings)

    payload = resolver.read_resource(
        f"rag://collections/{populated_collection}/chunks/chunk-one"
    )

    assert payload["type"] == "chunk"
    assert payload["chunk_id"] == "chunk-one"
    assert payload["text"]
    assert payload["tags"] == ["rag", "mcp"]


def test_missing_chunk_raises_controlled_error(
    resource_settings: SimpleNamespace,
    populated_collection: str,
) -> None:
    resolver = ResourceResolver(settings=resource_settings)

    with pytest.raises(ResourceResolutionError) as exc_info:
        resolver.read_resource(
            f"rag://collections/{populated_collection}/chunks/missing"
        )

    assert exc_info.value.error_code == "CHUNK_NOT_FOUND"
