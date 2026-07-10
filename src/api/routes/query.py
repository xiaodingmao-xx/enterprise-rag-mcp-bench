"""REST query endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import call_service, get_request_context, get_services, get_settings
from src.api.middleware.timeout import run_with_timeout, timeout_seconds
from src.api.schemas.query import QueryRequest
from src.security.context import RequestContext

router = APIRouter(prefix="/v1", tags=["query"])


@router.post("/query")
async def query(
    payload: QueryRequest,
    request: Request,
    context: RequestContext = Depends(get_request_context),
    services=Depends(get_services),
    settings=Depends(get_settings),
) -> dict:
    result = await run_with_timeout(
        call_service(services.query, "query", payload.model_dump(), context),
        timeout_seconds(settings, "query", 30.0),
        "RETRIEVAL_TIMEOUT",
    )
    response = dict(result) if isinstance(result, dict) else {"answer": result}
    response.setdefault("request_id", context.request_id)
    return response
