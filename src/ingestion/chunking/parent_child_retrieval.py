"""Helpers for retrieving a child while retaining parent context."""

from __future__ import annotations

import copy
from typing import Any, Iterable, Optional


def build_parent_index(chunks: Iterable[Any]) -> dict[str, Any]:
    return {str(getattr(chunk, "id", "")): chunk for chunk in chunks if getattr(chunk, "id", None)}


def get_parent_context(child: Any, chunks: Iterable[Any] | dict[str, Any], max_chars: int = 2000) -> Optional[Any]:
    index = chunks if isinstance(chunks, dict) else build_parent_index(chunks)
    parent_id = getattr(child, "parent_chunk_id", None) or getattr(child, "metadata", {}).get("parent_chunk_id")
    parent = index.get(str(parent_id)) if parent_id else None
    if parent is not None and max_chars > 0 and hasattr(parent, "text"):
        # Retrieval enrichment must not mutate the indexed parent object.
        parent = copy.copy(parent)
        parent.text = str(parent.text)[:max_chars]
    return parent
