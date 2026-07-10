"""Shared tenant and document authorization checks for REST and MCP HTTP."""

from __future__ import annotations

from typing import Any, Mapping

from src.api.errors import APIError
from src.security.context import RequestContext


class PermissionService:
    """Keep tenant isolation independent from transport-specific routes."""

    def require_context(self, context: RequestContext) -> None:
        if not context.authenticated:
            raise APIError("AUTH_REQUIRED", "Authentication is required")
        if not context.tenant_id:
            raise APIError("TENANT_REQUIRED", "A tenant identity is required")

    def ensure_tenant(self, resource_tenant_id: str | None, context: RequestContext) -> None:
        self.require_context(context)
        resource_tenant = str(resource_tenant_id or "")
        if resource_tenant and resource_tenant != str(context.tenant_id):
            raise APIError("TENANT_MISMATCH", "The resource belongs to another tenant")
        if not resource_tenant and context.auth_source != "local-dev":
            raise APIError("FORBIDDEN", "The resource has no trusted tenant boundary")

    def ensure_record(self, record: Mapping[str, Any] | None, context: RequestContext, *, not_found_code: str) -> dict[str, Any]:
        if record is None:
            raise APIError(not_found_code, "Resource not found")
        data = dict(record)
        self.ensure_tenant(data.get("tenant_id"), context)
        return data
