"""Unified, security-aware retrieval filters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Optional


class InvalidRetrievalFilterError(ValueError):
    pass


@dataclass
class RetrievalFilter:
    tenant_id: Optional[str] = None
    document_ids: Optional[list[str]] = None
    source_types: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    created_after: Optional[str] = None
    created_before: Optional[str] = None
    version_id: Optional[str] = None
    acl: Optional[dict[str, Any]] = None
    department: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RetrievalFilter":
        if data is None:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise InvalidRetrievalFilterError(f"Unknown retrieval filter fields: {sorted(unknown)}")
        values = dict(data)
        for key in ("document_ids", "source_types", "tags"):
            if values.get(key) is not None:
                if not isinstance(values[key], list) or not all(isinstance(item, str) for item in values[key]):
                    raise InvalidRetrievalFilterError(f"{key} must be list[str]")
        for key in ("created_after", "created_before"):
            if values.get(key):
                try:
                    datetime.fromisoformat(str(values[key]).replace("Z", "+00:00"))
                except ValueError as exc:
                    raise InvalidRetrievalFilterError(f"{key} must be ISO-like") from exc
        return cls(**values).validate()

    def validate(self) -> "RetrievalFilter":
        if self.created_after and self.created_before and self.created_after > self.created_before:
            raise InvalidRetrievalFilterError("created_after must not be after created_before")
        return self

    def merge_with_request_context(self, context: Any) -> "RetrievalFilter":
        values = self.to_dict()
        if getattr(context, "tenant_id", None):
            values["tenant_id"] = str(context.tenant_id)
        acl = dict(values.get("acl") or {})
        if getattr(context, "user_id", None):
            acl["user_id"] = str(context.user_id)
        if getattr(context, "roles", None):
            acl["roles"] = list(context.roles)
        if getattr(context, "department", None):
            values["department"] = str(context.department)
            acl["department"] = str(context.department)
        values["acl"] = acl or None
        return RetrievalFilter.from_dict(values)

    def to_metadata_filter(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.tenant_id:
            result["tenant_id"] = self.tenant_id
        if self.document_ids:
            result["document_id"] = {"$in": self.document_ids}
        if self.source_types and len(self.source_types) == 1:
            result["source_type"] = self.source_types[0]
        if self.version_id:
            result["version_id"] = self.version_id
        if self.department:
            result["acl_department"] = self.department
        return result

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        value = metadata or {}
        if self.tenant_id and str(value.get("tenant_id", "")) != str(self.tenant_id):
            return False
        if self.document_ids and str(value.get("document_id", value.get("doc_id", ""))) not in self.document_ids:
            return False
        if self.source_types and str(value.get("source_type", value.get("doc_type", ""))) not in self.source_types:
            return False
        if self.version_id and str(value.get("version_id", "")) != str(self.version_id):
            return False
        if self.department and str(value.get("acl_department", value.get("department", ""))) != self.department:
            return False
        if self.tags:
            raw_tags = value.get("tags", [])
            tags = set(raw_tags if isinstance(raw_tags, list) else str(raw_tags).replace(",", " ").split())
            if not set(self.tags).issubset(tags):
                return False
        if self.created_after and str(value.get("created_at", "")) < self.created_after:
            return False
        if self.created_before and str(value.get("created_at", "")) > self.created_before:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value is not None}


def validate_retrieval_filter(filters: RetrievalFilter | Mapping[str, Any] | None) -> RetrievalFilter:
    return filters if isinstance(filters, RetrievalFilter) else RetrievalFilter.from_dict(filters)
