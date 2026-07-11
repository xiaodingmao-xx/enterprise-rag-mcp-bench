"""Offline trust metrics for mock answers and retrieval contexts."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


def compute_trust_metrics(
    *,
    citation_verification: Any = None,
    refused: bool = False,
    expected_refusal: bool | None = None,
    confidence: Any = None,
) -> dict[str, Any]:
    """Return JSON-serializable P1 metrics without external providers."""
    valid = len(getattr(citation_verification, "valid_citations", []) or []) if citation_verification else 0
    invalid = len(getattr(citation_verification, "invalid_citations", []) or []) if citation_verification else 0
    supported = int(getattr(citation_verification, "supported_claim_count", 0) or 0) if citation_verification else 0
    unsupported = int(getattr(citation_verification, "unsupported_claim_count", 0) or 0) if citation_verification else 0
    claims = supported + unsupported
    return {
        "citation_precision": valid / (valid + invalid) if valid + invalid else 0.0,
        "citation_coverage": float(getattr(citation_verification, "citation_coverage", 0.0) or 0.0),
        "unsupported_claim_ratio": unsupported / claims if claims else 0.0,
        "refusal_accuracy": (
            float(refused == expected_refusal) if expected_refusal is not None else None
        ),
        "invalid_citation_rate": invalid / (valid + invalid) if valid + invalid else 0.0,
        "answer_confidence_distribution": {
            str(getattr(confidence, "label", "unknown")): 1
        } if confidence is not None else {},
    }

