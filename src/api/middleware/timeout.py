"""Cancellation-safe stage timeout helpers shared by API services."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from src.api.config import api_section, get_value
from src.api.errors import APIError


async def run_with_timeout(
    operation: Awaitable[Any],
    seconds: float,
    error_code: str = "RETRIEVAL_TIMEOUT",
) -> Any:
    """Cancel the downstream awaitable on timeout and preserve cancellation."""

    try:
        return await asyncio.wait_for(operation, timeout=max(0.001, float(seconds)))
    except asyncio.TimeoutError as exc:
        raise APIError(error_code, "The operation timed out") from exc
    except asyncio.CancelledError:
        # Never turn client disconnect cancellation into a successful request.
        raise


def timeout_seconds(settings: Any, stage: str, default: float) -> float:
    configured = get_value(api_section(settings, "timeout", None), f"{stage}_seconds", None)
    try:
        return max(0.001, float(configured if configured is not None else default))
    except (TypeError, ValueError):
        return default


def stage_error_code(stage: str) -> str:
    return {
        "retrieval": "RETRIEVAL_TIMEOUT",
        "rerank": "RERANK_TIMEOUT",
        "llm": "LLM_TIMEOUT",
        "ingestion": "INGESTION_TIMEOUT",
        "query": "RETRIEVAL_TIMEOUT",
    }.get(stage, "INTERNAL_ERROR")
