"""Resolve RAG MCP resource URIs into JSON-serialisable payloads."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from src.core.settings import Settings, load_settings
from src.libs.vector_store.chroma_store import ChromaStore
from src.mcp_server.resources.chunk_resource import build_chunk_payload
from src.mcp_server.resources.collection_resource import build_collection_payload
from src.mcp_server.resources.document_resource import (
    build_document_payload,
    derive_document_id,
)
from src.mcp_server.resources.resource_uri import ResourceUriError, parse_resource_uri
from src.mcp_server.tools.list_collections import ListCollectionsTool
from src.security.acl_filter import ACLFilter
from src.security.context import RequestContext, resolve_request_context
from src.security.policy import ACLPolicy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResourceDescriptor:
    """Small, SDK-neutral resource item representation."""

    uri: str
    name: str
    description: str
    mime_type: str = "application/json"


class ResourceResolutionError(RuntimeError):
    """User-safe resource resolution error."""

    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        self.message = message
        super().__init__(message)

    def to_payload(self) -> dict[str, str]:
        return {"error_code": self.error_code, "message": self.message}


class ResourceResolver:
    """Read-only resolver for collection, document, and chunk resources."""

    def __init__(
        self,
        settings: Settings | None = None,
        collection_limit: int = 100,
        documents_per_collection: int = 0,
        scan_limit: int = 5000,
        request_context: RequestContext | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.collection_limit = collection_limit
        self.documents_per_collection = documents_per_collection
        self.scan_limit = scan_limit
        self.request_context = request_context

    def _acl_policy(self) -> ACLPolicy:
        security_enabled = bool(getattr(getattr(self.settings, "security", None), "enabled", True))
        return ACLPolicy(enabled=security_enabled)

    def list_resource_descriptors(self, context: RequestContext | None = None) -> list[ResourceDescriptor]:
        """List collection and a capped set of document resources."""
        request_context = self._resolve_context(context)
        descriptors: list[ResourceDescriptor] = []
        for collection_name in self._list_collection_names(request_context)[: self.collection_limit]:
            descriptors.append(
                ResourceDescriptor(
                    uri=_collection_uri(collection_name),
                    name=f"Collection: {collection_name}",
                    description=f"RAG collection {collection_name}",
                )
            )
            if self.documents_per_collection <= 0:
                continue

            try:
                records = self._list_records(collection_name, limit=self.scan_limit, context=request_context)
            except Exception:
                logger.exception("Failed to list document resources for collection %s", collection_name)
                continue

            seen: set[str] = set()
            for record in records:
                document_id = derive_document_id(record)
                if document_id in seen:
                    continue
                seen.add(document_id)
                descriptors.append(
                    ResourceDescriptor(
                        uri=_document_uri(collection_name, document_id),
                        name=f"Document: {document_id}",
                        description=f"Document {document_id} in collection {collection_name}",
                    )
                )
                if len(seen) >= self.documents_per_collection:
                    break

        return descriptors

    def read_resource(self, uri: str, context: RequestContext | None = None) -> dict[str, Any]:
        """Read a resource payload by URI or raise ResourceResolutionError."""
        request_context = self._resolve_context(context)
        try:
            parsed = parse_resource_uri(uri)
        except ResourceUriError as exc:
            raise ResourceResolutionError("INVALID_RESOURCE_URI", str(exc)) from exc

        if parsed.resource_type == "collection":
            return self._read_collection(parsed.collection_name, request_context)
        if parsed.resource_type == "document":
            assert parsed.document_id is not None
            return self._read_document(parsed.collection_name, parsed.document_id, request_context)
        if parsed.resource_type == "chunk":
            assert parsed.chunk_id is not None
            return self._read_chunk(parsed.collection_name, parsed.chunk_id, request_context)

        raise ResourceResolutionError("INVALID_RESOURCE_URI", "Unsupported resource type")

    def _read_collection(self, collection_name: str, context: RequestContext) -> dict[str, Any]:
        self._ensure_collection_exists(collection_name, context)
        store = self._create_store(collection_name)
        try:
            chunk_count = store.collection.count()
            records = (
                self._records_from_collection_get(
                    store.collection.get(
                        limit=min(chunk_count, self.scan_limit),
                        include=["metadatas", "documents"],
                    )
                )
                if chunk_count
                else []
            )
            records = ACLFilter(self._acl_policy()).filter_records(records, context).results
            if not records:
                raise ResourceResolutionError("ACCESS_DENIED", "Collection is not accessible")
            return build_collection_payload(
                collection_name=collection_name,
                chunk_count=len(records),
                records=records,
                sampled=chunk_count > len(records),
            )
        finally:
            store.close()

    def _read_document(self, collection_name: str, document_id: str, context: RequestContext) -> dict[str, Any]:
        self._ensure_collection_exists(collection_name, context)
        records = self._list_records(collection_name, limit=self.scan_limit, context=context)
        document_records = [
            record for record in records if derive_document_id(record) == document_id
        ]
        if not document_records:
            raise ResourceResolutionError(
                "DOCUMENT_NOT_FOUND",
                f"Document resource not found: {_document_uri(collection_name, document_id)}",
            )
        return build_document_payload(collection_name, document_id, document_records)

    def _read_chunk(self, collection_name: str, chunk_id: str, context: RequestContext) -> dict[str, Any]:
        self._ensure_collection_exists(collection_name, context)
        store = self._create_store(collection_name)
        try:
            records = store.get_by_ids([chunk_id])
        finally:
            store.close()

        record = records[0] if records else {}
        if not record:
            raise ResourceResolutionError(
                "CHUNK_NOT_FOUND",
                f"Chunk resource not found: {_chunk_uri(collection_name, chunk_id)}",
            )
        if not self._acl_policy().can_access(record.get("metadata", {}), context):
            raise ResourceResolutionError("CHUNK_NOT_FOUND", "Chunk resource not found")
        return build_chunk_payload(collection_name, record)

    def _list_collection_names(self, context: RequestContext | None = None) -> list[str]:
        tool = ListCollectionsTool(settings=self.settings)
        return [info.name for info in tool.list_collections(include_stats=False, context=context)]

    def _ensure_collection_exists(self, collection_name: str, context: RequestContext | None = None) -> None:
        if collection_name not in set(self._list_collection_names(context)):
            raise ResourceResolutionError(
                "COLLECTION_NOT_FOUND",
                f"Collection resource not found: {_collection_uri(collection_name)}",
            )

    def _list_records(self, collection_name: str, limit: int, context: RequestContext | None = None) -> list[dict[str, Any]]:
        self._ensure_collection_exists(collection_name, context)
        store = self._create_store(collection_name)
        try:
            count = store.collection.count()
            if count <= 0:
                return []
            kwargs: dict[str, Any] = {
                "limit": min(count, limit),
                "include": ["metadatas", "documents"],
            }
            if context is not None and context.auth_source != "local-dev":
                native = ACLFilter.native_filters(context)
                if native:
                    kwargs["where"] = native
            records = self._records_from_collection_get(store.collection.get(**kwargs))
            return ACLFilter(self._acl_policy()).filter_records(records, context).results if context else records
        finally:
            store.close()

    def _create_store(self, collection_name: str) -> ChromaStore:
        return ChromaStore(settings=self.settings, collection_name=collection_name)

    def _resolve_context(self, context: RequestContext | None) -> RequestContext:
        return resolve_request_context(
            self.settings,
            context=context or self.request_context,
        )

    @staticmethod
    def _records_from_collection_get(results: dict[str, Any]) -> list[dict[str, Any]]:
        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        records: list[dict[str, Any]] = []
        for index, record_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            text = documents[index] if index < len(documents) and documents[index] else ""
            records.append({"id": record_id, "text": text, "metadata": metadata})
        return records


def _collection_uri(collection_name: str) -> str:
    return f"rag://collections/{quote(collection_name, safe='')}"


def _document_uri(collection_name: str, document_id: str) -> str:
    return f"{_collection_uri(collection_name)}/documents/{quote(document_id, safe='')}"


def _chunk_uri(collection_name: str, chunk_id: str) -> str:
    return f"{_collection_uri(collection_name)}/chunks/{quote(chunk_id, safe='')}"
