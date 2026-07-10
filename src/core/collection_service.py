"""Tenant-scoped collection facade."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.core.settings import Settings
from src.security.context import RequestContext


class CollectionService:
    def __init__(self, settings: Settings | Any = None) -> None:
        self.settings = settings
        self._collections: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        configured = getattr(getattr(settings, "vector_store", None), "collection_name", "knowledge_hub")
        self.default_collection = str(configured or "knowledge_hub")

    async def list_collections(self, context: RequestContext) -> list[dict[str, Any]]:
        tenant = str(context.tenant_id or "local")
        collections = self._collections[tenant]
        if not collections:
            collections[self.default_collection] = {
                "id": self.default_collection,
                "name": self.default_collection,
                "tenant_id": tenant,
                "document_count": 0,
            }
        return [dict(item) for item in collections.values()]

    def register(self, collection_id: str, context: RequestContext) -> dict[str, Any]:
        tenant = str(context.tenant_id or "local")
        item = self._collections[tenant].setdefault(
            collection_id,
            {
                "id": collection_id,
                "name": collection_id,
                "tenant_id": tenant,
                "document_count": 0,
            },
        )
        return dict(item)
