"""Early request-body size enforcement."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.api.config import api_section, get_value
from src.api.errors import APIError, error_response


class BodyLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, settings, max_bytes: int | None = None) -> None:
        super().__init__(app)
        self.settings = settings
        configured = get_value(api_section(settings, "limits", None), "max_request_body_bytes", None)
        self.max_bytes = int(max_bytes or configured or 10 * 1024 * 1024)

    async def dispatch(self, request: Request, call_next) -> Response:
        content_length = request.headers.get("content-length")
        try:
            too_large = content_length is not None and int(content_length) > self.max_bytes
        except ValueError:
            too_large = False
        if too_large:
            return error_response(
                request,
                APIError("REQUEST_TOO_LARGE", "Request body exceeds the configured limit"),
            )

        # For chunked requests, wrap receive so the body is rejected as soon
        # as the cumulative bytes cross the limit instead of being buffered in
        # full before validation.
        received = 0
        original_receive = request._receive

        async def limited_receive():
            nonlocal received
            message = await original_receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise APIError("REQUEST_TOO_LARGE", "Request body exceeds the configured limit")
            return message

        request._receive = limited_receive  # type: ignore[method-assign]
        try:
            return await call_next(request)
        except APIError as exc:
            return error_response(request, exc)
