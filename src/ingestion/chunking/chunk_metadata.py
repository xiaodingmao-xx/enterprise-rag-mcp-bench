"""Canonical chunk metadata and vector-safe serialization helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


LIST_FIELDS = ("heading_path", "section_path", "block_ids", "table_ids", "image_ids", "acl_users", "acl_roles")
VECTOR_TEXT_FIELDS = {
    "heading_path": "heading_path_text",
    "section_path": "section_path_text",
    "block_ids": "block_ids_text",
    "table_ids": "table_ids_text",
    "image_ids": "image_ids_text",
    "acl_users": "acl_users_text",
    "acl_roles": "acl_roles_text",
    "bbox": "bbox_text",
}


def _list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, list):
                return [str(item).strip() for item in decoded if str(item).strip()]
        except (TypeError, ValueError):
            pass
        return [item.strip() for item in re.split(r"[,;\n]+", raw) if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _bbox(value: Any) -> Optional[List[float]]:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            value = re.split(r"[,\s]+", value.strip())
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _page_pair(value: Any) -> tuple[Optional[int], Optional[int]]:
    if value is None or value == "":
        return None, None
    if isinstance(value, dict):
        start, end = value.get("start"), value.get("end", value.get("start"))
    elif isinstance(value, (list, tuple)) and value:
        start, end = value[0], value[-1]
    else:
        start = end = value
    try:
        start_i, end_i = int(start), int(end)
    except (TypeError, ValueError):
        return None, None
    if start_i <= 0 or end_i <= 0:
        return None, None
    return min(start_i, end_i), max(start_i, end_i)


@dataclass
class ChunkMetadata:
    tenant_id: Optional[str] = None
    document_id: Optional[str] = None
    version_id: Optional[str] = None
    chunk_id: str = ""
    parent_chunk_id: Optional[str] = None
    source_id: Optional[str] = None
    source_uri: Optional[str] = None
    source_type: Optional[str] = None
    title: Optional[str] = None
    heading_path: List[str] = field(default_factory=list)
    section_path: List[str] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    bbox: Optional[List[float]] = None
    block_ids: List[str] = field(default_factory=list)
    table_ids: List[str] = field(default_factory=list)
    image_ids: List[str] = field(default_factory=list)
    language: Optional[str] = None
    content_hash: Optional[str] = None
    parser_version: Optional[str] = None
    chunker_version: Optional[str] = None
    embedding_model: Optional[str] = None
    acl_visibility: Optional[str] = None
    acl_users: List[str] = field(default_factory=list)
    acl_roles: List[str] = field(default_factory=list)
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, metadata: Dict[str, Any]) -> "ChunkMetadata":
        return cls(**{key: value for key, value in normalize_chunk_metadata(metadata).items() if key in cls.__dataclass_fields__})

    @classmethod
    def from_document_and_draft(
        cls,
        document: Any,
        draft: Any,
        chunk_id: str,
        chunk_index: int,
        settings: Any = None,
    ) -> "ChunkMetadata":
        metadata = dict(getattr(document, "metadata", {}) or {})
        metadata.update(getattr(draft, "metadata", {}) or {})
        metadata.update(
            {
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "parent_chunk_id": getattr(draft, "parent_chunk_id", None),
                "heading_path": getattr(draft, "heading_path", None) or metadata.get("heading_path"),
                "section_path": getattr(draft, "section_path", None) or metadata.get("section_path"),
                "page_range": getattr(draft, "page_range", None) or metadata.get("page_range"),
                "source_ref": getattr(draft, "source_ref", None) or metadata.get("source_ref"),
            }
        )
        if settings is not None:
            ingestion = getattr(settings, "ingestion", None)
            chunking = getattr(ingestion, "chunking", None)
            metadata.setdefault("chunker_version", getattr(chunking, "strategy", None))
            metadata.setdefault("embedding_model", getattr(getattr(settings, "embedding", None), "model", None))
        return cls.from_dict(metadata)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_chunk_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy aliases into one canonical metadata shape."""

    raw = dict(metadata or {})
    source_uri = raw.get("source_uri") or raw.get("source_path") or raw.get("source")
    source_type = raw.get("source_type") or raw.get("doc_type")
    if not source_type and source_uri:
        source_type = Path(str(source_uri)).suffix.lower().lstrip(".") or None
    document_id = raw.get("document_id") or raw.get("doc_id") or raw.get("source_doc_id")
    page_start, page_end = _page_pair(raw.get("page_range"))
    page_value = raw.get("page_start", raw.get("page", raw.get("page_num", raw.get("page_number"))))
    if page_value is not None:
        page_start, page_end = _page_pair(page_value)
    if raw.get("page_end") is not None:
        try:
            page_end = int(raw["page_end"])
        except (TypeError, ValueError):
            page_end = None

    result = dict(raw)
    result.update(
        {
            "tenant_id": str(raw["tenant_id"]) if raw.get("tenant_id") is not None else None,
            "document_id": str(document_id) if document_id is not None else None,
            "version_id": str(raw["version_id"]) if raw.get("version_id") is not None else None,
            "chunk_id": str(raw.get("chunk_id") or ""),
            "parent_chunk_id": str(raw["parent_chunk_id"]) if raw.get("parent_chunk_id") else None,
            "source_id": str(raw["source_id"]) if raw.get("source_id") is not None else None,
            "source_uri": str(source_uri) if source_uri is not None else None,
            "source_type": str(source_type) if source_type is not None else None,
            "heading_path": _list(raw.get("heading_path")),
            "section_path": _list(raw.get("section_path")) or _list(raw.get("heading_path")),
            "page_start": page_start,
            "page_end": page_end,
            "bbox": _bbox(raw.get("bbox")),
            "block_ids": _list(raw.get("block_ids")),
            "table_ids": _list(raw.get("table_ids")),
            "image_ids": _list(raw.get("image_ids", raw.get("image_refs"))),
            "acl_users": _list(raw.get("acl_users", raw.get("allowed_users"))),
            "acl_roles": _list(raw.get("acl_roles", raw.get("allowed_roles"))),
            "created_at": str(raw.get("created_at") or datetime.now(timezone.utc).isoformat()),
        }
    )
    for key in ("title", "language", "content_hash", "parser_version", "chunker_version", "embedding_model", "acl_visibility"):
        if result.get(key) is not None:
            result[key] = str(result[key])
    result["heading_path_text"] = " > ".join(result["heading_path"])
    result["section_path_text"] = " > ".join(result["section_path"])
    result["block_ids_text"] = ",".join(result["block_ids"])
    result["table_ids_text"] = ",".join(result["table_ids"])
    result["image_ids_text"] = ",".join(result["image_ids"])
    result["acl_users_text"] = ",".join(result["acl_users"])
    result["acl_roles_text"] = ",".join(result["acl_roles"])
    result["bbox_text"] = ",".join(str(item) for item in result["bbox"]) if result["bbox"] else ""
    return result


def to_vector_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten canonical metadata to primitive values accepted by vector stores."""

    normalized = normalize_chunk_metadata(metadata)
    output: Dict[str, Any] = {}
    for key, value in normalized.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            if value is not None:
                output[key] = value
        elif isinstance(value, (list, tuple, dict)):
            output[key] = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return output


def content_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()
