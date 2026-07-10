"""MCP SDK adapter for RAG resources."""

from __future__ import annotations

import json
import logging

from mcp import types
from mcp.server.lowlevel.helper_types import ReadResourceContents

from src.mcp_server.resources.resource_resolver import (
    ResourceResolutionError,
    ResourceResolver,
)
from src.security.context import RequestContext

logger = logging.getLogger(__name__)


class RagResourceRegistry:
    """Expose RAG resources through MCP SDK types."""

    def __init__(self, resolver: ResourceResolver | None = None) -> None:
        self.resolver = resolver or ResourceResolver()

    def list_resources(self, context: RequestContext | None = None) -> list[types.Resource]:
        try:
            descriptors = self.resolver.list_resource_descriptors(context=context)
        except Exception:
            logger.exception("Failed to list MCP resources")
            return []

        return [
            types.Resource(
                uri=descriptor.uri,
                name=descriptor.name,
                description=descriptor.description,
                mimeType=descriptor.mime_type,
            )
            for descriptor in descriptors
        ]

    def read_resource(self, uri: str, context: RequestContext | None = None) -> list[ReadResourceContents]:
        try:
            payload = self.resolver.read_resource(uri, context=context)
        except ResourceResolutionError as exc:
            payload = exc.to_payload()
        except Exception:
            logger.exception("Failed to read MCP resource: %s", uri)
            payload = {
                "error_code": "INTERNAL_ERROR",
                "message": "Failed to read resource",
            }

        return [
            ReadResourceContents(
                content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
                mime_type="application/json",
            )
        ]
