"""Safe, request-correlated errors shared by REST and MCP HTTP routes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


ERROR_STATUS: dict[str, int] = {
    "AUTH_REQUIRED": 401,
    "INVALID_TOKEN": 401,
    "TOKEN_EXPIRED": 401,
    "TENANT_REQUIRED": 401,
    "FORBIDDEN": 403,
    "TENANT_MISMATCH": 403,
    "INVALID_QUERY": 400,
    "VALIDATION_ERROR": 400,
    "REQUEST_TOO_LARGE": 413,
    "DOCUMENT_NOT_FOUND": 404,
    "COLLECTION_NOT_FOUND": 404,
    "INGESTION_JOB_NOT_FOUND": 404,
    "INGESTION_FAILED": 500,
    "INGESTION_TIMEOUT": 504,
    "RETRIEVAL_TIMEOUT": 504,
    "RERANK_TIMEOUT": 504,
    "LLM_TIMEOUT": 504,
    "RATE_LIMITED": 429,
    "MCP_TOOL_NOT_FOUND": 404,
    "MCP_SCHEMA_MISMATCH": 400,
    "NOT_IMPLEMENTED": 501,
    "INTERNAL_ERROR": 500,
}

RETRYABLE_CODES = {
    "RATE_LIMITED",
    "INGESTION_TIMEOUT",
    "RETRIEVAL_TIMEOUT",
    "RERANK_TIMEOUT",
    "LLM_TIMEOUT",
    "INGESTION_FAILED",
    "INTERNAL_ERROR",
}


@dataclass
class APIError(Exception):
    """An expected error that is safe to expose to an API client."""

    error_code: str
    message: str
    status_code: int | None = None
    retryable: bool | None = None
    headers: dict[str, str] | None = None

    def __post_init__(self) -> None:
        self.error_code = str(self.error_code).upper()
        self.status_code = self.status_code or ERROR_STATUS.get(self.error_code, 500)
        if self.retryable is None:
            self.retryable = self.error_code in RETRYABLE_CODES
        super().__init__(self.message)


def request_id_from(request: Request) -> str:
    context = getattr(request.state, "context", None)
    request_id = getattr(context, "request_id", None)
    return str(request_id or request.headers.get("X-Request-ID") or "unknown")


def error_payload(request: Request, error: APIError) -> dict[str, Any]:
    return {
        "request_id": request_id_from(request),
        "error_code": error.error_code,
        "message": error.message,
        "retryable": bool(error.retryable),
    }


def error_response(request: Request, error: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=int(error.status_code or 500),
        content=error_payload(request, error),
        headers=error.headers or {},
    )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return error_response(request, exc)


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Pydantic's raw error details may include implementation-specific input
    # values.  Keep the public response stable and non-sensitive.
    logger.debug("Request validation failed for %s", request.url.path)
    return error_response(
        request,
        APIError("VALIDATION_ERROR", "Request validation failed", status_code=400),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Detailed traces are opt-in debug diagnostics; the public response and
    # normal logs never contain arbitrary exception text or secrets.
    logger.error(
        "Unhandled API error on %s (%s)",
        request.url.path,
        type(exc).__name__,
        exc_info=logger.isEnabledFor(logging.DEBUG),
    )
    return error_response(
        request,
        APIError("INTERNAL_ERROR", "Internal server error", status_code=500),
    )


def api_error_from_mapping(value: Mapping[str, Any], *, default_code: str = "INTERNAL_ERROR") -> APIError:
    """Convert service-layer error mappings without leaking arbitrary fields."""

    code = str(value.get("error_code") or default_code).upper()
    message = str(value.get("message") or "Request failed")
    return APIError(code, message)
