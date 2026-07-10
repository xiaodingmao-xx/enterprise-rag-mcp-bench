"""Ingestion job endpoints backed by the existing task queue facade."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import call_service, get_request_context, get_services
from src.api.middleware.timeout import run_with_timeout, timeout_seconds
from src.api.schemas.ingestion import IngestionJobCreateRequest
from src.security.context import RequestContext

router = APIRouter(prefix="/v1/ingestion/jobs", tags=["ingestion"])


@router.post("")
async def create_ingestion_job(
    payload: IngestionJobCreateRequest,
    request: Request,
    context: RequestContext = Depends(get_request_context),
    services=Depends(get_services),
) -> dict:
    return dict(
        await run_with_timeout(
            call_service(services.ingestion, "create_job", payload.model_dump(), context),
            timeout_seconds(request.app.state.settings, "ingestion", 300.0),
            "INGESTION_TIMEOUT",
        )
    )


@router.get("")
async def list_ingestion_jobs(
    context: RequestContext = Depends(get_request_context),
    services=Depends(get_services),
) -> dict:
    return {"jobs": await call_service(services.ingestion, "list_jobs", context)}


@router.get("/{job_id}")
async def get_ingestion_job(
    job_id: str,
    context: RequestContext = Depends(get_request_context),
    services=Depends(get_services),
) -> dict:
    return dict(await call_service(services.ingestion, "get_job", job_id, context))
