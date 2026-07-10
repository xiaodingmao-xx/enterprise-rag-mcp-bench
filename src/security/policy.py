"""Deterministic ACL policy evaluation independent of any LLM prompt."""

from __future__ import annotations

from typing import Any, Mapping

from src.security.context import RequestContext
from src.security.models import DocumentACL


class ACLPolicy:
    """Evaluate document visibility for a trusted request context."""

    def __init__(self, *, enabled: bool = True, allow_legacy_local: bool = True) -> None:
        self.enabled = enabled
        self.allow_legacy_local = allow_legacy_local

    def can_access(self, metadata: Mapping[str, Any], context: RequestContext, *, action: str = "read") -> bool:
        if not self.enabled:
            return True
        tenant_id = str(metadata.get("tenant_id", "") or "")
        if tenant_id and tenant_id != str(context.tenant_id or ""):
            return False
        if not tenant_id:
            # Existing local data predates ACL metadata. It is tolerated only
            # in local-dev mode and never in a tenant-authenticated deployment.
            if not (self.allow_legacy_local and context.auth_source == "local-dev"):
                return False
            return True

        if context.is_admin:
            return True

        try:
            acl = DocumentACL.from_metadata(metadata, fallback_document_id=str(metadata.get("document_id", "")))
        except (TypeError, ValueError):
            return False
        visibility = acl.visibility
        user_id = str(context.user_id or "")
        roles = {role.lower() for role in context.roles}

        if visibility in {"public", "tenant"}:
            return True
        if visibility == "private":
            return bool(acl.owner_id and acl.owner_id == user_id)
        if visibility == "users":
            return user_id in set(acl.allowed_users) or (acl.owner_id and acl.owner_id == user_id)
        if visibility == "roles":
            return bool(roles.intersection({role.lower() for role in acl.allowed_roles}))
        if visibility == "department":
            return bool(acl.department and acl.department == (context.department or ""))
        return False

    def filter_records(self, records: list[dict[str, Any]], context: RequestContext) -> list[dict[str, Any]]:
        return [
            record for record in records
            if self.can_access(record.get("metadata", {}) if isinstance(record, dict) else {}, context)
        ]

    def filter_retrieval_results(self, results: list[Any], context: RequestContext) -> list[Any]:
        filtered = []
        for result in results:
            metadata = getattr(result, "metadata", None)
            if not isinstance(metadata, Mapping) and isinstance(result, dict):
                metadata = result.get("metadata", {})
            if self.can_access(metadata or {}, context):
                filtered.append(result)
        return filtered
