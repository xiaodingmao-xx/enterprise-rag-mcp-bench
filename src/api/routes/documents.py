"""Tenant-scoped document lifecycle endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import call_service, get_request_context, get_services
from src.api.middleware.timeout import run_with_timeout, timeout_seconds
from src.api.schemas.documents import DocumentCreateRequest, DocumentRollbackRequest
from src.security.context import RequestContext

router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.post("")
async def create_document(
    payload: DocumentCreateRequest,
    request: Request,
    context: RequestContext = Depends(get_request_context),
    services=Depends(get_services),
) -> dict:
    result = await run_with_timeout(
        call_service(services.documents, "create", payload.model_dump(), context),
        timeout_seconds(request.app.state.settings, "ingestion", 300.0),
        "INGESTION_TIMEOUT",
    )
    return dict(result)


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    context: RequestContext = Depends(get_request_context),
    services=Depends(get_services),
) -> dict:
    return dict(await call_service(services.documents, "get", document_id, context))


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    context: RequestContext = Depends(get_request_context),
    services=Depends(get_services),
) -> dict:
    return dict(await call_service(services.documents, "delete", document_id, context))


@router.get("/{document_id}/versions")
async def document_versions(
    document_id: str,
    context: RequestContext = Depends(get_request_context),
    services=Depends(get_services),
) -> dict:
    return {"document_id": document_id, "versions": await call_service(services.documents, "versions", document_id, context)}


@router.post("/{document_id}/rollback")
async def rollback_document(
    document_id: str,
    payload: DocumentRollbackRequest,
    context: RequestContext = Depends(get_request_context),
    services=Depends(get_services),
) -> dict:
    return dict(
        await call_service(
            services.documents,
            "rollback",
            document_id,
            payload.version_id,
            context,
        )
    )
