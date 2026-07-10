"""Request identity propagation for REST and MCP HTTP requests."""

from __future__ import annotations

import logging
import uuid
from dataclasses import replace

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.security.context import RequestContext

logger = logging.getLogger(__name__)


def _header_id(request: Request, name: str, fallback: str) -> str:
    value = request.headers.get(name, "").strip()
    # Prevent unbounded user-controlled identifiers from entering logs or
    # response headers while retaining normal distributed tracing IDs.
    return value[:128] if value else fallback


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Create a per-request context before authentication and rate limiting."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = _header_id(request, "X-Request-ID", str(uuid.uuid4()))
        trace_id = _header_id(request, "X-Trace-ID", request_id)
        client_host = request.client.host if request.client else ""
        context = RequestContext(
            request_id=request_id,
            trace_id=trace_id,
            client_host=client_host,
            user_agent=request.headers.get("User-Agent", "")[:512],
        )
        request.state.context = context
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id
        logger.info(
            "HTTP request completed: %s %s %s",
            request.method,
            request.url.path,
            response.status_code,
            extra={"request_id": request_id, "trace_id": trace_id},
        )
        return response


def update_transport_context(request: Request, context: RequestContext) -> RequestContext:
    """Preserve request IDs and transport metadata on an authenticated context."""

    base = getattr(request.state, "context", None)
    if base is None:
        return context
    return replace(
        context,
        request_id=base.request_id,
        trace_id=base.trace_id,
        client_host=base.client_host,
        user_agent=base.user_agent,
    )
