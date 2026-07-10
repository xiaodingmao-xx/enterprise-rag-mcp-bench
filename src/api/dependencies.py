"""FastAPI dependencies shared by REST and MCP gateway routes."""

from __future__ import annotations

import inspect
from typing import Any

from fastapi import Request

from src.api.errors import APIError
from src.security.context import RequestContext


def get_request_context(request: Request) -> RequestContext:
    context = getattr(request.state, "context", None)
    if not isinstance(context, RequestContext) or not context.authenticated:
        raise APIError("AUTH_REQUIRED", "Authentication is required")
    return context


def get_settings(request: Request) -> Any:
    return request.app.state.settings


def get_services(request: Request) -> Any:
    return request.app.state.services


async def call_service(service: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(service, method_name, None)
    if not callable(method):
        raise APIError("INTERNAL_ERROR", f"Service operation '{method_name}' is unavailable")
    result = method(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result
