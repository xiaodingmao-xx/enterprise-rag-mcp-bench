"""Canonical ACL metadata model shared by ingestion and retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional


Visibility = Literal["private", "users", "roles", "department", "tenant", "public"]
VALID_VISIBILITIES = {"private", "users", "roles", "department", "tenant", "public"}


def _list_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, list):
                return [str(item) for item in decoded if str(item).strip()]
        except (TypeError, ValueError):
            pass
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


@dataclass(frozen=True)
class DocumentACL:
    """Document-level access metadata inherited by every chunk."""

    tenant_id: str
    document_id: str
    version_id: str = ""
    owner_id: str = ""
    allowed_users: List[str] = field(default_factory=list)
    allowed_roles: List[str] = field(default_factory=list)
    visibility: Visibility = "tenant"
    source_system: str = "local"
    department: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("DocumentACL.tenant_id is required")
        if not self.document_id:
            raise ValueError("DocumentACL.document_id is required")
        visibility = str(self.visibility).strip().lower()
        if visibility not in VALID_VISIBILITIES:
            raise ValueError(f"Unsupported ACL visibility: {visibility}")
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "allowed_users", _list_value(self.allowed_users))
        object.__setattr__(self, "allowed_roles", _list_value(self.allowed_roles))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any], *, fallback_document_id: str = "") -> "DocumentACL":
        """Create an ACL from canonical keys or legacy aliases."""

        document_id = str(
            metadata.get("document_id")
            or metadata.get("doc_id")
            or metadata.get("source_ref")
            or fallback_document_id
            or "unknown"
        )
        return cls(
            tenant_id=str(metadata.get("tenant_id", "")),
            document_id=document_id,
            version_id=str(metadata.get("version_id", "")),
            owner_id=str(metadata.get("owner_id", "")),
            allowed_users=_list_value(metadata.get("acl_users", metadata.get("allowed_users"))),
            allowed_roles=_list_value(metadata.get("acl_roles", metadata.get("allowed_roles"))),
            visibility=str(metadata.get("acl_visibility", metadata.get("visibility", "tenant"))).lower(),
            source_system=str(metadata.get("source_system", "local")),
            department=str(metadata.get("acl_department", metadata.get("department", ""))),
            created_at=str(metadata.get("created_at", "")),
            updated_at=str(metadata.get("updated_at", "")),
        )

    def to_metadata(self) -> Dict[str, Any]:
        """Return canonical metadata suitable for Document and Chunk objects."""

        return {
            "tenant_id": self.tenant_id,
            "document_id": self.document_id,
            "version_id": self.version_id,
            "owner_id": self.owner_id,
            "acl_users": list(self.allowed_users),
            "acl_roles": list(self.allowed_roles),
            "acl_visibility": self.visibility,
            "source_system": self.source_system,
            "acl_department": self.department,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
