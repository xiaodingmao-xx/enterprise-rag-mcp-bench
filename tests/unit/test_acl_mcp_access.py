"""MCP collection/resource ACL boundary tests."""

from __future__ import annotations

import sys
import types
import importlib.machinery
from types import SimpleNamespace

# The repository's optional MCP dependency is not required for these pure
# boundary tests. Provide the tiny SDK surface imported by the tool modules.
if "mcp" not in sys.modules:
    mcp = types.ModuleType("mcp")
    mcp.__spec__ = importlib.machinery.ModuleSpec("mcp", loader=None)
    mcp.types = SimpleNamespace(TextContent=object, CallToolResult=object, Tool=object)
    server = types.ModuleType("mcp.server")
    lowlevel = types.ModuleType("mcp.server.lowlevel")
    helper_types = types.ModuleType("mcp.server.lowlevel.helper_types")
    helper_types.ReadResourceContents = object
    sys.modules["mcp"] = mcp
    sys.modules["mcp.server"] = server
    sys.modules["mcp.server.lowlevel"] = lowlevel
    sys.modules["mcp.server.lowlevel.helper_types"] = helper_types

import pytest

from src.mcp_server.resources.resource_resolver import ResourceResolutionError, ResourceResolver
from src.mcp_server.tools.list_collections import ListCollectionsTool
from src.security.context import RequestContext
from src.security.models import DocumentACL


class FakeCollection:
    def __init__(self, name: str, records: list[dict]) -> None:
        self.name = name
        self.metadata = {}
        self.records = records

    def count(self) -> int:
        return len(self.records)

    def get(self, **kwargs):
        return {"metadatas": [record["metadata"] for record in self.records]}


class FakeClient:
    def __init__(self, collections):
        self.collections = collections

    def list_collections(self):
        return self.collections


def _context(user: str) -> RequestContext:
    return RequestContext(
        tenant_id="tenant-a",
        user_id=user,
        roles=("member",),
        auth_source="jwt",
        authenticated=True,
    )


def _settings():
    return SimpleNamespace(
        vector_store=SimpleNamespace(persist_directory="./data/db/chroma", collection_name="docs"),
        security=SimpleNamespace(
            mode="local-dev",
            require_tenant=False,
            require_authentication=False,
        ),
    )


def test_list_collections_hides_collections_without_access() -> None:
    visible = {"metadata": DocumentACL(tenant_id="tenant-a", document_id="d1", allowed_users=["alice"], visibility="users").to_metadata()}
    hidden = {"metadata": DocumentACL(tenant_id="tenant-a", document_id="d2", allowed_users=["bob"], visibility="users").to_metadata()}
    tool = ListCollectionsTool(settings=_settings())
    tool._get_chroma_client = lambda: FakeClient([FakeCollection("visible", [visible]), FakeCollection("hidden", [hidden])])

    result = tool.list_collections(include_stats=True, context=_context("alice"))

    assert [item.name for item in result] == ["visible"]


def test_document_resource_denies_unauthorized_user() -> None:
    denied_record = {
        "id": "chunk-1",
        "text": "secret",
        "metadata": DocumentACL(
            tenant_id="tenant-a",
            document_id="doc-1",
            allowed_users=["alice"],
            visibility="users",
        ).to_metadata(),
    }

    class Store:
        collection = FakeCollection("docs", [denied_record])

        def get_by_ids(self, ids):
            return [denied_record]

        def close(self):
            pass

    resolver = ResourceResolver(settings=_settings())
    resolver._list_collection_names = lambda context=None: ["docs"]
    resolver._create_store = lambda collection_name: Store()

    with pytest.raises(ResourceResolutionError) as exc_info:
        resolver.read_resource("rag://collections/docs/chunks/chunk-1", context=_context("bob"))

    assert exc_info.value.error_code == "CHUNK_NOT_FOUND"
