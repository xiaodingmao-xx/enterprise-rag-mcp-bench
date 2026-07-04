"""MCP resource support for RAG collections, documents, and chunks."""

from src.mcp_server.resources.resource_registry import RagResourceRegistry
from src.mcp_server.resources.resource_resolver import (
    ResourceDescriptor,
    ResourceResolutionError,
    ResourceResolver,
)
from src.mcp_server.resources.resource_uri import (
    ParsedResourceUri,
    ResourceUriError,
    parse_resource_uri,
)

__all__ = [
    "ParsedResourceUri",
    "RagResourceRegistry",
    "ResourceDescriptor",
    "ResourceResolutionError",
    "ResourceResolver",
    "ResourceUriError",
    "parse_resource_uri",
]
