"""Tests for query_knowledge_hub LRU+TTL cache integration."""

import importlib.util
import asyncio
import sys
from types import ModuleType, SimpleNamespace

if importlib.util.find_spec("mcp") is None:
    fake_mcp = ModuleType("mcp")

    class TextContent:
        def __init__(self, type: str, text: str) -> None:
            self.type = type
            self.text = text

    class ImageContent:
        pass

    class CallToolResult:
        def __init__(self, content, isError: bool = False) -> None:
            self.content = content
            self.isError = isError

    fake_mcp.types = SimpleNamespace(
        TextContent=TextContent,
        ImageContent=ImageContent,
        CallToolResult=CallToolResult,
        Tool=object,
    )
    sys.modules["mcp"] = fake_mcp

from src.core.settings import PerformanceSettings, QueryCacheSettings
from src.core.types import RetrievalResult
from src.mcp_server.tools import query_knowledge_hub as tool_module
from src.mcp_server.tools.query_knowledge_hub import (
    QueryKnowledgeHubConfig,
    QueryKnowledgeHubTool,
)


class DummyTraceCollector:
    def collect(self, trace) -> None:
        pass


def test_query_tool_returns_cached_response_without_research(monkeypatch) -> None:
    asyncio.run(_run_query_tool_cache_assertions(monkeypatch))


async def _run_query_tool_cache_assertions(monkeypatch) -> None:
    settings = SimpleNamespace(
        embedding=SimpleNamespace(provider="fake", model="fake-embedding"),
        rerank=SimpleNamespace(enabled=False, provider="none", model="none", top_k=5),
        performance=PerformanceSettings(
            query_cache=QueryCacheSettings(enabled=True, max_size=8, ttl_seconds=60),
        ),
    )
    tool = QueryKnowledgeHubTool(
        settings=settings,
        config=QueryKnowledgeHubConfig(enable_rerank=False),
    )
    search_calls = {"count": 0}

    def fake_search(query, top_k, trace=None):
        search_calls["count"] += 1
        return [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.9,
                text="cached answer",
                metadata={"source_path": "doc.pdf"},
            )
        ]

    monkeypatch.setattr(tool, "_ensure_initialized", lambda collection: None)
    monkeypatch.setattr(tool, "_perform_search", fake_search)
    monkeypatch.setattr(tool_module, "TraceCollector", DummyTraceCollector)

    first = await tool.execute("what is cached", top_k=1, collection="default")
    second = await tool.execute("what is cached", top_k=1, collection="default")

    assert search_calls["count"] == 1
    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True
    assert second.content == first.content
