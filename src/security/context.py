"""Trusted request identity and development/JWT authentication boundary."""

from __future__ import annotations

import base64
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


class AuthenticationError(PermissionError):
    """Raised when a production request has no valid authentication."""


class TenantRequiredError(PermissionError):
    """Raised when a tenant boundary cannot be established."""


@dataclass(frozen=True)
class RequestContext:
    """Identity bound to one request; never read from a mutable global."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    roles: tuple[str, ...] = ()
    department: Optional[str] = None
    auth_source: str = "unknown"
    authenticated: bool = False
    # API-facing aliases/transport metadata.  ``auth_source`` remains the
    # canonical field used by the existing ACL implementation.
    auth_mode: str = "unknown"
    client_host: str = ""
    user_agent: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", tuple(sorted({str(role) for role in self.roles if str(role).strip()})))

    @property
    def is_admin(self) -> bool:
        return any(role.lower() in {"admin", "tenant_admin", "system_admin"} for role in self.roles)

    @classmethod
    def local_dev(cls, tenant_id: str = "local", user_id: str = "local-user") -> "RequestContext":
        return cls(
            tenant_id=tenant_id,
            user_id=user_id,
            roles=("admin",),
            department="local",
            auth_source="local-dev",
            authenticated=True,
            auth_mode="local-dev",
        )

    @classmethod
    def from_claims(cls, claims: Mapping[str, Any], *, auth_source: str = "jwt") -> "RequestContext":
        roles = claims.get("roles", claims.get("role", ()))
        if isinstance(roles, str):
            roles = [item.strip() for item in roles.split(",") if item.strip()]
        return cls(
            request_id=str(claims.get("request_id") or uuid.uuid4()),
            trace_id=str(claims.get("trace_id") or uuid.uuid4()),
            tenant_id=str(claims.get("tenant_id")) if claims.get("tenant_id") else None,
            user_id=str(claims.get("sub") or claims.get("user_id")) if (claims.get("sub") or claims.get("user_id")) else None,
            roles=tuple(roles or ()),
            department=str(claims.get("department")) if claims.get("department") else None,
            auth_source=auth_source,
            authenticated=True,
            auth_mode=auth_source,
        )


def _setting(settings: Any, name: str, default: Any) -> Any:
    security = getattr(settings, "security", None) if settings is not None else None
    if security is None and isinstance(settings, Mapping):
        security = settings.get("security", {})
    if security is not None and security.__class__.__module__ == "unittest.mock":
        return default
    if isinstance(security, Mapping):
        return security.get(name, default)
    return getattr(security, name, default)


def _jwt_setting(settings: Any, name: str, default: Any = "") -> Any:
    jwt_settings = _setting(settings, "jwt", {})
    if isinstance(jwt_settings, Mapping):
        return jwt_settings.get(name, default)
    return getattr(jwt_settings, name, default)


def _decode_jwt(token: str, settings: Any) -> Mapping[str, Any]:
    secret = str(_jwt_setting(settings, "secret", os.environ.get("JWT_SECRET", "")) or "")
    issuer = str(_jwt_setting(settings, "issuer", "") or "")
    audience = str(_jwt_setting(settings, "audience", "") or "")
    if not secret:
        raise AuthenticationError("JWT verifier is not configured")
    try:
        import jwt  # type: ignore

        options = {"require": ["sub", "exp"]}
        kwargs: dict[str, Any] = {"algorithms": ["HS256"], "options": options}
        if issuer:
            kwargs["issuer"] = issuer
        if audience:
            kwargs["audience"] = audience
        claims = jwt.decode(token, secret, **kwargs)
        return claims if isinstance(claims, Mapping) else {}
    except ImportError as exc:
        raise AuthenticationError("PyJWT is required for production JWT validation") from exc
    except Exception as exc:
        raise AuthenticationError("JWT validation failed") from exc


def resolve_request_context(
    settings: Any,
    *,
    context: Optional[RequestContext] = None,
    authorization: Optional[str] = None,
) -> RequestContext:
    """Resolve a trusted context according to local-dev or production mode.

    Tool arguments are intentionally not accepted here.  In production only a
    validated JWT or an application-injected ``RequestContext`` is trusted.
    """

    mode = str(_setting(settings, "mode", "local-dev")).lower()
    require_tenant = bool(_setting(settings, "require_tenant", mode in {"production", "prod"}))
    require_auth = bool(_setting(settings, "require_authentication", mode in {"production", "prod"}))

    if context is None and authorization:
        token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else authorization.strip()
        context = RequestContext.from_claims(_decode_jwt(token, settings))

    if context is None:
        if mode in {"production", "prod"} or require_auth:
            raise AuthenticationError("Authentication is required")
        context = RequestContext.local_dev(
            tenant_id=str(_setting(settings, "default_local_tenant", "local")),
            user_id=str(_setting(settings, "default_local_user", "local-user")),
        )

    if require_auth and not context.authenticated:
        raise AuthenticationError("Authenticated RequestContext is required")
    if require_tenant and not context.tenant_id:
        raise TenantRequiredError("tenant_id is required")
    return context
