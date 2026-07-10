"""Tenant-aware document lifecycle facade for REST and MCP HTTP."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from src.api.errors import APIError
from src.core.permission_service import PermissionService
from src.security.context import RequestContext


class DocumentService:
    def __init__(
        self,
        settings: Any = None,
        *,
        ingestion_service: Any = None,
        permission_service: PermissionService | None = None,
        document_manager: Any = None,
        version_store: Any = None,
    ) -> None:
        self.settings = settings
        self.ingestion_service = ingestion_service
        self.permission_service = permission_service or PermissionService()
        self.document_manager = document_manager
        self.version_store = version_store
        self._documents: dict[str, dict[str, Any]] = {}

    async def create(self, payload: dict[str, Any], context: RequestContext) -> dict[str, Any]:
        self.permission_service.require_context(context)
        document_id = uuid.uuid4().hex
        collection_id = str(payload.get("collection_id") or "default")
        record = {
            "document_id": document_id,
            "tenant_id": str(context.tenant_id),
            "title": payload.get("title") or payload.get("external_document_id") or document_id,
            "source_uri": payload.get("source_uri") or "",
            "source_type": payload.get("source_type") or "text",
            "metadata": dict(payload.get("metadata") or {}),
            "collection_id": collection_id,
            "external_document_id": payload.get("external_document_id"),
            "status": "created",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._documents[document_id] = record
        job_id = None
        if payload.get("content") and self.ingestion_service is not None:
            job = await self.ingestion_service.create_job(
                {
                    "content": payload["content"],
                    "filename": payload.get("title") or f"{document_id}.txt",
                    "collection_id": collection_id,
                    "metadata": payload.get("metadata") or {},
                },
                context,
            )
            job_id = job.get("job_id")
            record["status"] = "ingestion_queued"
        return {
            "document_id": document_id,
            "version_id": None,
            "job_id": job_id,
            "status": record["status"],
        }

    async def get(self, document_id: str, context: RequestContext) -> dict[str, Any]:
        record = self.permission_service.ensure_record(
            self._documents.get(document_id), context, not_found_code="DOCUMENT_NOT_FOUND"
        )
        return record

    async def delete(self, document_id: str, context: RequestContext) -> dict[str, Any]:
        record = await self.get(document_id, context)
        if self.document_manager is not None:
            delete = getattr(self.document_manager, "delete_document", None)
            if callable(delete) and record.get("source_uri"):
                await __import__("asyncio").to_thread(delete, record["source_uri"], record.get("collection_id", "default"))
        self._documents.pop(document_id, None)
        return {"document_id": document_id, "status": "deleted"}

    async def versions(self, document_id: str, context: RequestContext) -> list[dict[str, Any]]:
        await self.get(document_id, context)
        if self.version_store is None:
            raise APIError("NOT_IMPLEMENTED", "Document version listing is not configured")
        return [item.to_dict() for item in self.version_store.list_versions(document_id)]

    async def rollback(self, document_id: str, version_id: str, context: RequestContext) -> dict[str, Any]:
        await self.get(document_id, context)
        if self.version_store is None:
            raise APIError("NOT_IMPLEMENTED", "Document rollback is not configured")
        result = self.version_store.rollback_to_version(
            document_id, version_id, tenant_id=str(context.tenant_id), actor=str(context.user_id or "")
        )
        return result.to_dict() if hasattr(result, "to_dict") else dict(result)

    async def summary(self, document_id: str, context: RequestContext) -> dict[str, Any]:
        record = await self.get(document_id, context)
        return {
            "doc_id": document_id,
            "title": record.get("title", ""),
            "summary": record.get("metadata", {}).get("summary", ""),
            "tags": record.get("metadata", {}).get("tags", []),
            "source_uri": record.get("source_uri", ""),
            "collection_id": record.get("collection_id", "default"),
        }
