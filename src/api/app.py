"""FastAPI application factory for REST and MCP HTTP access."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from src.api.config import auth_mode, environment, get_value
from src.api.errors import api_error_handler, unhandled_error_handler, validation_error_handler
from src.api.health import router as health_router
from src.api.middleware.auth import AuthenticationMiddleware
from src.api.middleware.body_limit import BodyLimitMiddleware
from src.api.middleware.rate_limit import RateLimiterBackend, RateLimitMiddleware
from src.api.middleware.request_context import RequestContextMiddleware
from src.api.routes.collections import router as collections_router
from src.api.routes.documents import router as documents_router
from src.api.routes.ingestion_jobs import router as ingestion_router
from src.api.routes.mcp_gateway import router as mcp_router
from src.api.routes.query import router as query_router
from src.core.collection_service import CollectionService
from src.core.document_service import DocumentService
from src.core.ingestion_service import IngestionService
from src.core.permission_service import PermissionService
from src.core.query_service import QueryService
from src.core.settings import Settings, SettingsError, load_settings


def _resolve_settings(settings: Any) -> Any:
    if settings is None:
        return load_settings()
    if isinstance(settings, (str, Path)):
        return load_settings(settings)
    if isinstance(settings, Mapping):
        return Settings.from_dict(dict(settings))
    return settings


def _build_services(settings: Any, provided: Any = None) -> Any:
    supplied = provided if provided is not None else SimpleNamespace()

    def take(name: str, default: Any) -> Any:
        if isinstance(supplied, Mapping):
            value = supplied.get(name)
        else:
            value = getattr(supplied, name, None)
        return default if value is None else value

    permissions = take("permissions", PermissionService())
    ingestion = take("ingestion", IngestionService(settings))
    documents = take(
        "documents",
        DocumentService(settings, ingestion_service=ingestion, permission_service=permissions),
    )
    return SimpleNamespace(
        permissions=permissions,
        ingestion=ingestion,
        documents=documents,
        collections=take("collections", CollectionService(settings)),
        query=take("query", QueryService(settings)),
    )


async def _default_readiness() -> dict[str, Any]:
    return {"ready": True, "dependencies": {"configuration": True}}


def create_app(
    settings: Any = None,
    *,
    services: Any = None,
    readiness_check: Callable[[], Any] | None = None,
    rate_limiter_backend: RateLimiterBackend | None = None,
) -> FastAPI:
    """Create an isolated app instance suitable for uvicorn and TestClient."""

    resolved_settings = _resolve_settings(settings)
    api = get_value(resolved_settings, "api", None)
    if environment(resolved_settings) in {"production", "prod"} and auth_mode(resolved_settings) in {
        "local-dev",
        "development",
    }:
        raise SettingsError("production API cannot use local-dev authentication")

    app = FastAPI(
        title="Modular RAG API",
        version="0.1.0",
        description="Tenant-aware REST API and MCP HTTP Gateway for the Modular RAG server.",
    )
    app.state.settings = resolved_settings
    app.state.services = _build_services(resolved_settings, services)
    app.state.readiness_check = readiness_check or _default_readiness
    app.state.mcp_gateway = None

    # Starlette inserts newly registered middleware at the front of its stack;
    # register in reverse execution order so request identity is outermost,
    # followed by body protection, authentication, and rate limiting.
    app.add_middleware(
        RateLimitMiddleware,
        settings=resolved_settings,
        backend=rate_limiter_backend,
    )
    app.add_middleware(AuthenticationMiddleware, settings=resolved_settings)
    app.add_middleware(BodyLimitMiddleware, settings=resolved_settings)
    app.add_middleware(RequestContextMiddleware)

    app.add_exception_handler(Exception, unhandled_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    from src.api.errors import APIError

    app.add_exception_handler(APIError, api_error_handler)

    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(documents_router)
    app.include_router(collections_router)
    app.include_router(ingestion_router)
    mcp_enabled = get_value(get_value(api, "mcp_gateway", None), "enabled", True)
    if mcp_enabled:
        app.include_router(mcp_router)
    return app


app = create_app()

__all__ = ["app", "create_app"]
