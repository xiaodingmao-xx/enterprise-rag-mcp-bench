"""Local-development and self-contained HS256 JWT authentication."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Mapping
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.api.config import auth_mode, auth_value, environment
from src.api.errors import APIError, error_response
from src.api.middleware.request_context import update_transport_context
from src.security.context import RequestContext


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _decode_hs256(token: str, *, secret: str, issuer: str, audience: str, leeway: int) -> Mapping[str, Any]:
    parts = token.split(".")
    if len(parts) != 3 or not secret:
        raise APIError("INVALID_TOKEN", "Invalid authentication token")
    try:
        header = json.loads(_b64decode(parts[0]).decode("utf-8"))
        claims = json.loads(_b64decode(parts[1]).decode("utf-8"))
        signature = _b64decode(parts[2])
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise APIError("INVALID_TOKEN", "Invalid authentication token") from exc
    if not isinstance(header, dict) or header.get("alg") != "HS256":
        raise APIError("INVALID_TOKEN", "Invalid authentication token")
    expected = hmac.new(secret.encode("utf-8"), f"{parts[0]}.{parts[1]}".encode("ascii"), hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise APIError("INVALID_TOKEN", "Invalid authentication token")
    if not isinstance(claims, dict):
        raise APIError("INVALID_TOKEN", "Invalid authentication token")

    exp = claims.get("exp")
    if exp is None:
        raise APIError("INVALID_TOKEN", "Invalid authentication token")
    try:
        if float(exp) < time.time() - leeway:
            raise APIError("TOKEN_EXPIRED", "Authentication token has expired")
    except (TypeError, ValueError) as exc:
        raise APIError("INVALID_TOKEN", "Invalid authentication token") from exc

    if issuer and claims.get("iss") != issuer:
        raise APIError("INVALID_TOKEN", "Invalid authentication token")
    if audience:
        token_aud = claims.get("aud")
        valid_aud = token_aud == audience or (
            isinstance(token_aud, list) and audience in token_aud
        )
        if not valid_aud:
            raise APIError("INVALID_TOKEN", "Invalid authentication token")
    if not claims.get("sub") and not claims.get("user_id"):
        raise APIError("INVALID_TOKEN", "Invalid authentication token")
    return claims


def _decode_token(token: str, settings: Any) -> Mapping[str, Any]:
    algorithm = str(auth_value(settings, "algorithm", "HS256")).upper()
    if algorithm != "HS256":
        raise APIError("INVALID_TOKEN", "Unsupported authentication algorithm")
    return _decode_hs256(
        token,
        secret=str(auth_value(settings, "secret", "") or ""),
        issuer=str(auth_value(settings, "issuer", "") or ""),
        audience=str(auth_value(settings, "audience", "") or ""),
        leeway=int(auth_value(settings, "leeway_seconds", 30) or 0),
    )


def _roles_from_header(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in (value or "").split(",") if item.strip())


def authenticate_request(request: Request, settings: Any) -> RequestContext:
    """Authenticate one request without ever returning the raw token."""

    mode = auth_mode(settings)
    if mode in {"local-dev", "development"}:
        if environment(settings) in {"production", "prod"}:
            raise APIError("INVALID_TOKEN", "Production API cannot use local-dev authentication")
        allow_without_auth = bool(
            auth_value(settings, "allow_local_dev_without_auth", True)
        )
        authorization = request.headers.get("Authorization", "")
        if authorization and authorization.lower().startswith("bearer "):
            claims = _decode_token(authorization[7:].strip(), settings)
            context = RequestContext.from_claims(claims, auth_source="jwt")
        elif not allow_without_auth:
            raise APIError("AUTH_REQUIRED", "Authentication is required")
        else:
            tenant = request.headers.get("X-Tenant-ID") or "local"
            user = request.headers.get("X-User-ID") or "local-dev-user"
            roles = _roles_from_header(request.headers.get("X-Roles")) or ("admin",)
            context = RequestContext(
                tenant_id=tenant,
                user_id=user,
                roles=roles,
                department=request.headers.get("X-Department"),
                auth_source="local-dev",
                auth_mode="local-dev",
                authenticated=True,
            )
        return update_transport_context(request, context)

    if mode not in {"jwt", "production", "prod"}:
        raise APIError("INVALID_TOKEN", "Unsupported authentication mode")
    authorization = request.headers.get("Authorization", "")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise APIError("AUTH_REQUIRED", "Authentication is required")
    token = authorization[7:].strip()
    if not token:
        raise APIError("AUTH_REQUIRED", "Authentication is required")
    context = RequestContext.from_claims(_decode_token(token, settings), auth_source="jwt")
    if not context.tenant_id:
        raise APIError("TENANT_REQUIRED", "A tenant identity is required")
    return update_transport_context(request, context)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authenticate protected routes and make the trusted context available."""

    def __init__(self, app, *, settings: Any, excluded_paths: set[str] | None = None) -> None:
        super().__init__(app)
        self.settings = settings
        self.excluded_paths = excluded_paths or {
            "/health/live",
            "/health/ready",
            "/openapi.json",
            "/docs",
            "/docs/oauth2-redirect",
            "/redoc",
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        try:
            request.state.context = authenticate_request(request, self.settings)
            return await call_next(request)
        except APIError as exc:
            return error_response(request, exc)
