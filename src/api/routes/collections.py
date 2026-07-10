"""Tenant-scoped collection endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import call_service, get_request_context, get_services
from src.security.context import RequestContext

router = APIRouter(prefix="/v1", tags=["collections"])


@router.get("/collections")
async def list_collections(
    context: RequestContext = Depends(get_request_context),
    services=Depends(get_services),
) -> dict:
    result = await call_service(services.collections, "list_collections", context)
    return {"collections": result}
