# MCP Client Setup Guide

## 1. Overview

This project exposes a local RAG knowledge base through the Model Context Protocol
(MCP) over stdio.

Supported MCP capabilities:

- Tools: query and inspect the knowledge base.
- Resources: read collections, documents, and chunks as structured resources.

Current limitations:

- Prompts are not supported.
- Sampling is not supported.

## 2. Prerequisites

- Python 3.10 or newer.
- Project dependencies installed in your active environment.
- A configured `config/settings.yaml`.
- API keys set through environment variables when you want to call LLM,
  embedding, or vision providers.

Install the project in editable mode when you want to use the console script:

```bash
python -m pip install -e .
```

Do not put real API keys in repository files. Use environment variables such as:

```bash
OPENAI_API_KEY=your-api-key
EMBEDDING_API_KEY=your-api-key
LLM_API_KEY=your-api-key
VISION_LLM_API_KEY=your-api-key
```

## 3. Start MCP Server

From the project root, start the MCP stdio server with:

```bash
python -m src.mcp_server.server
```

After installing the project, you can also use:

```bash
mcp-server
```

The server writes protocol messages to stdout and logs to stderr. This is
important because MCP stdio clients expect stdout to contain only JSON-RPC
messages.

## 4. Claude Desktop Configuration

Claude Desktop uses a JSON configuration file. Point the server at the project
root and start it with Python module mode:

```json
{
  "mcpServers": {
    "modular-rag": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "env": {
        "OPENAI_API_KEY": "your-api-key"
      }
    }
  }
}
```

If your Python executable is not on PATH, use the absolute path to the virtual
environment Python executable. On Windows, that often looks like:

```json
{
  "mcpServers": {
    "modular-rag": {
      "command": "H:\\Learning\\postgraduate\\0Resume\\project\\MODULAR-RAG-MCP-SERVER-main\\.venv\\Scripts\\python.exe",
      "args": ["-m", "src.mcp_server.server"],
      "env": {
        "OPENAI_API_KEY": "your-api-key"
      }
    }
  }
}
```

## 5. Generic MCP Client Configuration

Any MCP client that supports stdio can start the server with:

```json
{
  "name": "modular-rag",
  "transport": {
    "type": "stdio",
    "command": "python",
    "args": ["-m", "src.mcp_server.server"],
    "env": {
      "OPENAI_API_KEY": "your-api-key"
    }
  }
}
```

When the project is installed, the command can be simplified to:

```json
{
  "name": "modular-rag",
  "transport": {
    "type": "stdio",
    "command": "mcp-server",
    "args": []
  }
}
```

## 6. Available Tools

### query_knowledge_hub

Search the knowledge base with hybrid retrieval.

Example arguments:

```json
{
  "query": "How does the RAG pipeline work?",
  "top_k": 5,
  "collection": "default"
}
```

### list_collections

List available Chroma collections.

Example arguments:

```json
{
  "include_stats": true
}
```

### get_document_summary

Read a document summary from a collection.

Example arguments:

```json
{
  "doc_id": "doc_abc123",
  "collection": "default"
}
```

## 7. Available Resources

Resources are read-only JSON payloads exposed through MCP `resources/list` and
`resources/read`.

Supported URI formats:

- `rag://collections/{collection_name}`
- `rag://collections/{collection_name}/documents/{document_id}`
- `rag://collections/{collection_name}/chunks/{chunk_id}`

Example collection read:

```json
{
  "uri": "rag://collections/default"
}
```

Example document read:

```json
{
  "uri": "rag://collections/default/documents/doc_abc123"
}
```

Example chunk read:

```json
{
  "uri": "rag://collections/default/chunks/chunk_001"
}
```

Document IDs prefer metadata fields such as `doc_id`, `document_id`, or
`source_ref`. If those are unavailable, the server derives a stable ID from
the source path.

## 8. Example Workflows

### 1. List collections

1. Send MCP `initialize`.
2. Send `resources/list`.
3. Pick a resource URI like `rag://collections/default`.
4. Send `resources/read` for that URI.

### 2. Query the knowledge base

1. Send `tools/list` and confirm `query_knowledge_hub` is available.
2. Call `query_knowledge_hub` with a user question.
3. Inspect returned citations, source metadata, and chunk IDs.

### 3. Read a returned chunk resource

1. Run `query_knowledge_hub`.
2. Copy a returned `chunk_id`.
3. Read `rag://collections/default/chunks/{chunk_id}` with `resources/read`.
4. Use the returned `text`, `source`, `page`, and metadata for grounding.

## 9. Troubleshooting

### MCP client cannot find Python

Use an absolute Python path in the MCP client configuration. For a local virtual
environment on Windows, use `.venv\\Scripts\\python.exe`.

### Environment variables are missing

Set provider-specific API keys in the MCP client `env` block. Restart the MCP
client after changing environment variables.

### Collection does not exist

Run `list_collections` or `resources/list` first. Check that ingestion wrote to
the same `vector_store.collection_name` and `vector_store.persist_directory`.

### Chroma data is empty

Run an ingestion command first, then restart the MCP client if it keeps a long
running server process.

### Windows path issues

Escape backslashes in JSON strings or use double backslashes. Avoid putting
unescaped Windows paths directly inside JSON.

### Server starts but client receives no response

Make sure the MCP server is started through stdio and that stdout is not polluted
by print statements. Logs should go to stderr.

### JSON-RPC initialization fails

Verify the client sends a valid MCP `initialize` request before calling tools or
resources. Then send `notifications/initialized`.

## 10. Limitations

- Resources are read-only.
- Prompts are not supported.
- Sampling is not supported.
- Document IDs may be derived from source path hashes when stable document
  metadata is unavailable.
- `resources/list` does not enumerate every chunk by default because large
  knowledge bases may contain thousands of chunks.
- Collection `document_count` and metadata fields are best-effort when the
  collection is large, because exact values require scanning stored records.
