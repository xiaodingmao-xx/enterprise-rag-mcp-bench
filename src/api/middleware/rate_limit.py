"""Per-tenant and per-user rate limiting with an in-memory default backend."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from threading import Lock
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.api.config import api_section, get_value
from src.api.errors import APIError, error_response


class RateLimiterBackend(ABC):
    @abstractmethod
    async def allow(self, key: str, limit: int, window_seconds: int) -> bool:
        """Return whether one request is allowed for *key*."""

    def retry_after(self, key: str, window_seconds: int) -> int:
        return max(1, int(window_seconds))


class InMemoryRateLimiterBackend(RateLimiterBackend):
    """Fixed-window/sliding-window hybrid suitable for one process."""

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = {}
        self._lock = Lock()

    async def allow(self, key: str, limit: int, window_seconds: int) -> bool:
        now = time.monotonic()
        with self._lock:
            events = self._events.setdefault(key, deque())
            cutoff = now - max(1, int(window_seconds))
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= max(1, int(limit)):
                return False
            events.append(now)
            return True

    def retry_after(self, key: str, window_seconds: int) -> int:
        now = time.monotonic()
        with self._lock:
            events = self._events.get(key)
            if not events:
                return 1
            return max(1, int(events[0] + max(1, int(window_seconds)) - now + 0.999))


class RedisRateLimiterBackend(RateLimiterBackend):
    """Reserved backend contract; Redis is intentionally not mandatory."""

    async def allow(self, key: str, limit: int, window_seconds: int) -> bool:
        raise NotImplementedError("Redis rate limiting backend is not installed")


def _rule(settings: Any, identity: str) -> tuple[int, int]:
    configured = api_section(settings, "rate_limit", None)
    rule = get_value(configured, identity, None)
    return (
        int(get_value(rule, "requests", 120 if identity == "tenant" else 60)),
        int(get_value(rule, "window_seconds", 60)),
    )


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, settings: Any, backend: RateLimiterBackend | None = None) -> None:
        super().__init__(app)
        self.settings = settings
        config = api_section(settings, "rate_limit", None)
        self.enabled = bool(get_value(config, "enabled", True))
        backend_name = str(get_value(config, "backend", "memory")).lower()
        if backend is not None:
            self.backend = backend
        elif backend_name == "memory":
            self.backend = InMemoryRateLimiterBackend()
        elif backend_name == "redis":
            self.backend = RedisRateLimiterBackend()
        else:
            raise ValueError(f"Unsupported rate limiter backend: {backend_name}")

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.enabled or request.url.path in {"/health/live", "/health/ready"}:
            return await call_next(request)
        context = getattr(request.state, "context", None)
        if context is None:
            return await call_next(request)

        identities = [
            ("tenant", str(getattr(context, "tenant_id", None) or "anonymous")),
            ("user", str(getattr(context, "user_id", None) or "anonymous")),
        ]
        for identity, value in identities:
            limit, window = _rule(self.settings, identity)
            # Keys contain only trusted, non-secret identity values.
            key = f"{identity}:{value}"
            try:
                allowed = await self.backend.allow(key, limit, window)
            except NotImplementedError:
                return error_response(
                    request,
                    APIError("INTERNAL_ERROR", "Configured rate limiter is unavailable"),
                )
            if not allowed:
                retry_after = self.backend.retry_after(key, window)
                return error_response(
                    request,
                    APIError(
                        "RATE_LIMITED",
                        "Rate limit exceeded",
                        headers={"Retry-After": str(retry_after)},
                    ),
                )
        return await call_next(request)
