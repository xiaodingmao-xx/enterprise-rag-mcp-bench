"""Core query facade shared by REST and the HTTP MCP gateway."""

from __future__ import annotations

import time
from typing import Any, Mapping

from src.api.errors import APIError
from src.core.settings import Settings
from src.security.context import RequestContext


class QueryService:
    """Adapt the existing query_knowledge_hub implementation to an API DTO."""

    def __init__(self, settings: Settings | Any = None, *, tool: Any = None) -> None:
        self.settings = settings
        self.tool = tool

    def _get_tool(self, *, use_rerank: bool) -> Any:
        if self.tool is None:
            from src.mcp_server.tools.query_knowledge_hub import (
                QueryKnowledgeHubConfig,
                QueryKnowledgeHubTool,
            )

            self.tool = QueryKnowledgeHubTool(
                settings=self.settings,
                config=QueryKnowledgeHubConfig(enable_rerank=use_rerank),
            )
        return self.tool

    async def query(self, payload: Mapping[str, Any], context: RequestContext) -> dict[str, Any]:
        query = str(payload.get("query", "")).strip()
        if not query:
            raise APIError("INVALID_QUERY", "query must not be blank")
        started = time.perf_counter()
        tool = self._get_tool(use_rerank=bool(payload.get("use_rerank", True)))
        result = await tool.execute(
            query=query,
            top_k=int(payload.get("top_k", 5)),
            collection=payload.get("collection_id") or payload.get("collection"),
            mode="answer" if bool(payload.get("use_llm", True)) else "contexts",
            context=context,
        )
        normalized = self._normalize_result(result)
        normalized.setdefault("request_id", context.request_id)
        normalized.setdefault("answer", normalized.get("content", ""))
        normalized.setdefault("citations", [])
        normalized.setdefault("chunks", [])
        normalized["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return normalized

    @staticmethod
    def _normalize_result(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return dict(result)
        to_dict = getattr(result, "to_dict", None)
        if callable(to_dict):
            raw = to_dict()
            if isinstance(raw, dict):
                structured = raw.get("structuredContent", {})
                if isinstance(structured, dict):
                    return {
                        "answer": raw.get("content", ""),
                        "content": raw.get("content", ""),
                        "citations": structured.get("citations", []),
                        "chunks": structured.get("metadata", {}).get("results", []),
                        "metadata": structured.get("metadata", {}),
                    }
                return raw
        content = getattr(result, "content", None)
        if isinstance(content, str):
            return {"answer": content, "content": content}
        return {"answer": str(result)}
