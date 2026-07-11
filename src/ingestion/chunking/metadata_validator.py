"""Stage-aware validation for canonical chunk metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from src.ingestion.chunking.chunk_metadata import LIST_FIELDS, normalize_chunk_metadata, to_vector_metadata


class MetadataValidationError(ValueError):
    """Raised when metadata cannot satisfy the current pipeline contract."""


def validate_chunk_metadata(metadata: Dict[str, Any], stage: str = "chunk") -> Dict[str, Any]:
    raw = dict(metadata or {})
    normalized = normalize_chunk_metadata(raw)
    errors = []
    if raw.get("chunk_id") is not None and not isinstance(raw.get("chunk_id"), str):
        errors.append("chunk_id must be a string")
    if not normalized.get("chunk_id") and stage in {"chunk", "upsert", "vector"}:
        errors.append("chunk_id must be a non-empty string")
    for field in LIST_FIELDS:
        if not isinstance(normalized.get(field), list) or not all(isinstance(item, str) for item in normalized[field]):
            errors.append(f"{field} must be list[str]")
    if normalized.get("page_start") is not None and not isinstance(normalized["page_start"], int):
        errors.append("page_start must be int or None")
    if normalized.get("page_end") is not None and not isinstance(normalized["page_end"], int):
        errors.append("page_end must be int or None")
    if normalized.get("page_start") and normalized.get("page_end") and normalized["page_end"] < normalized["page_start"]:
        errors.append("page_end must not be smaller than page_start")
    page_fields = ("page", "page_num", "page_number", "page_start", "page_end", "page_range")
    if any(field in raw and raw.get(field) not in (None, "") for field in page_fields):
        if normalized.get("page_start") is None or normalized.get("page_end") is None:
            errors.append("page fields must contain positive integers")
    bbox = normalized.get("bbox")
    if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(item, float) for item in bbox)):
        errors.append("bbox must be list[float] with four values or None")
    if raw.get("bbox") not in (None, "") and bbox is None:
        errors.append("bbox must contain four numeric values")
    if normalized.get("acl_visibility") and normalized["acl_visibility"] not in {
        "public", "tenant", "private", "restricted", "unknown", "users", "roles", "department"
    }:
        errors.append("acl_visibility is unsupported")
    if normalized.get("created_at"):
        try:
            datetime.fromisoformat(str(normalized["created_at"]).replace("Z", "+00:00"))
        except ValueError:
            errors.append("created_at must be ISO-like")
    if errors:
        raise MetadataValidationError(f"Invalid chunk metadata at stage={stage}: {'; '.join(errors)}")
    if stage in {"vector", "upsert"}:
        return to_vector_metadata(normalized)
    return normalized


def sanitize_retrieval_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return normalize_chunk_metadata(metadata)
    except Exception:
        return dict(metadata or {})
