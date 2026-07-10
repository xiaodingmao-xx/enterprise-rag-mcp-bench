"""HTTP adapter for the existing stdio MCP tool registry."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import call_service, get_request_context
from src.api.errors import APIError
from src.security.context import RequestContext


class HTTPMCPGateway:
    """Expose stdio MCP schemas while routing execution through API services."""

    def __init__(self, settings: Any, services: Any) -> None:
        self.settings = settings
        self.services = services
        self._protocol_handler = None

    @property
    def protocol_handler(self) -> Any:
        if self._protocol_handler is None:
            try:
                from src.mcp_server.protocol_handler import ProtocolHandler, _register_default_tools

                handler = ProtocolHandler(
                    server_name="modular-rag-mcp-server-http",
                    server_version="0.1.0",
                    settings=self.settings,
                )
                _register_default_tools(handler)
                self._protocol_handler = handler
            except ImportError as exc:
                missing_module = str(getattr(exc, "name", "") or "")
                if not (missing_module == "mcp" or missing_module.startswith("mcp.")):
                    raise
                # The MCP dependency is optional for REST-only deployments
                # and for isolated API tests.  Keep the fallback schemas in
                # sync with the stdio tool contracts so the gateway remains
                # inspectable without installing the transport package.
                self._protocol_handler = SimpleNamespace(
                    tools={
                        "query_knowledge_hub": SimpleNamespace(
                            name="query_knowledge_hub",
                            description="Search the knowledge base for relevant documents.",
                            input_schema={
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                                    "collection": {"type": "string"},
                                    "mode": {"type": "string", "enum": ["contexts", "answer"]},
                                    "answer_style": {"type": "string", "enum": ["concise", "detailed", "bullet"]},
                                    "language": {"type": "string", "enum": ["auto", "zh", "en"]},
                                    "include_sources": {"type": "boolean"},
                                    "include_citations": {"type": "boolean"},
                                },
                                "required": ["query"],
                            },
                        ),
                        "list_collections": SimpleNamespace(
                            name="list_collections",
                            description="List all available document collections in the knowledge base.",
                            input_schema={
                                "type": "object",
                                "properties": {"include_stats": {"type": "boolean"}},
                                "required": [],
                            },
                        ),
                        "get_document_summary": SimpleNamespace(
                            name="get_document_summary",
                            description="Get summary and metadata for a specific document.",
                            input_schema={
                                "type": "object",
                                "properties": {
                                    "doc_id": {"type": "string"},
                                    "collection": {"type": "string"},
                                },
                                "required": ["doc_id"],
                            },
                        ),
                    }
                )
        return self._protocol_handler

    def tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": definition.name,
                "description": definition.description,
                "inputSchema": definition.input_schema,
            }
            for definition in self.protocol_handler.tools.values()
        ]

    def _validate_arguments(self, name: str, arguments: dict[str, Any]) -> None:
        definition = self.protocol_handler.tools.get(name)
        if definition is None:
            raise APIError("MCP_TOOL_NOT_FOUND", f"MCP tool '{name}' was not found")
        schema = definition.input_schema or {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [field for field in required if field not in arguments]
        unknown = [field for field in arguments if field not in properties]
        if missing or unknown:
            raise APIError("MCP_SCHEMA_MISMATCH", "MCP tool arguments do not match the tool schema")

    async def call(self, name: str, arguments: dict[str, Any], context: RequestContext) -> dict[str, Any]:
        self._validate_arguments(name, arguments)
        args = dict(arguments)
        if name == "query_knowledge_hub":
            result = await call_service(
                self.services.query,
                "query",
                {
                    "query": args.get("query"),
                    "top_k": args.get("top_k", 5),
                    "collection_id": args.get("collection"),
                    "use_rerank": True,
                    "use_llm": args.get("mode", "contexts") == "answer",
                    "filters": {},
                },
                context,
            )
        elif name == "list_collections":
            result = await call_service(self.services.collections, "list_collections", context)
        elif name == "get_document_summary":
            result = await call_service(
                self.services.documents,
                "summary",
                str(args.get("doc_id")),
                context,
            )
        else:
            raise APIError("MCP_TOOL_NOT_FOUND", f"MCP tool '{name}' was not found")
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False, default=str),
                }
            ],
            "isError": False,
        }


router = APIRouter(tags=["mcp"])


def _gateway(request: Request) -> HTTPMCPGateway:
    gateway = getattr(request.app.state, "mcp_gateway", None)
    if gateway is None:
        gateway = HTTPMCPGateway(request.app.state.settings, request.app.state.services)
        request.app.state.mcp_gateway = gateway
    return gateway


@router.get("/mcp/tools")
@router.get("/v1/mcp/tools")
async def list_mcp_tools(
    request: Request,
    context: RequestContext = Depends(get_request_context),
) -> dict[str, Any]:
    return {"tools": _gateway(request).tools()}


@router.post("/mcp/tools/call")
@router.post("/v1/mcp/tools/call")
async def call_mcp_tool(
    body: dict[str, Any],
    request: Request,
    context: RequestContext = Depends(get_request_context),
) -> dict[str, Any]:
    name = body.get("name")
    arguments = body.get("arguments", {})
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise APIError("MCP_SCHEMA_MISMATCH", "MCP tool call requires name and object arguments")
    return await _gateway(request).call(name, arguments, context)


@router.post("/mcp")
@router.post("/v1/mcp")
async def mcp_json_rpc(
    body: dict[str, Any],
    request: Request,
    context: RequestContext = Depends(get_request_context),
) -> dict[str, Any]:
    method = body.get("method")
    rpc_id = body.get("id")
    if body.get("jsonrpc") not in {None, "2.0"} or not isinstance(method, str):
        raise APIError("MCP_SCHEMA_MISMATCH", "Invalid MCP JSON-RPC request")
    gateway = _gateway(request)
    if method == "tools/list":
        result: Any = {"tools": gateway.tools()}
    elif method == "tools/call":
        params = body.get("params")
        if not isinstance(params, dict):
            raise APIError("MCP_SCHEMA_MISMATCH", "MCP tools/call params must be an object")
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise APIError("MCP_SCHEMA_MISMATCH", "MCP tool call requires name and object arguments")
        result = await gateway.call(name, arguments, context)
    else:
        raise APIError("MCP_TOOL_NOT_FOUND", "Unsupported MCP method")
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}
