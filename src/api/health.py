"""Liveness and readiness endpoints."""

from __future__ import annotations

import inspect
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "live"}


@router.get("/ready")
async def ready(request: Request) -> Any:
    checker = request.app.state.readiness_check
    try:
        result = checker()
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, dict):
            is_ready = bool(result.get("ready", result.get("status") == "ready"))
            details = {"status": "ready" if is_ready else "not_ready"}
            # Only expose stable dependency status labels, never paths or
            # exception text supplied by a backend.
            dependencies = result.get("dependencies")
            if isinstance(dependencies, dict):
                details["dependencies"] = {
                    str(key): bool(value) for key, value in dependencies.items()
                }
        else:
            is_ready = bool(result)
            details = {"status": "ready" if is_ready else "not_ready"}
        if not is_ready:
            return JSONResponse(status_code=503, content=details)
        return details
    except Exception:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
