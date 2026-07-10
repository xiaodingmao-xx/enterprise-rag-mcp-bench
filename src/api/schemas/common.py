"""Common API schema helpers."""

from __future__ import annotations

from typing import Any


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        return result if isinstance(result, dict) else {"value": result}
    if hasattr(value, "model_dump"):
        result = value.model_dump()
        return result if isinstance(result, dict) else {"value": result}
    if hasattr(value, "dict"):
        result = value.dict()
        return result if isinstance(result, dict) else {"value": result}
    return {"value": value}
