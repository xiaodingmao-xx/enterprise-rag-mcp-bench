"""E2E tests for MCP resources over stdio."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from src.core.settings import load_settings
from src.libs.vector_store.chroma_store import ChromaStore

PROJECT_ROOT = Path(__file__).parent.parent.parent
COLLECTION_NAME = "mcp_resources_e2e"
DOCUMENT_ID = "doc-alpha"
CHUNK_ID = "chunk-one"


INIT_REQUEST: dict[str, Any] = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "resource-e2e-client", "version": "0.1.0"},
    },
}

INITIALIZED_NOTIFICATION: dict[str, Any] = {
    "jsonrpc": "2.0",
    "method": "notifications/initialized",
}


@pytest.fixture()
def resource_collection() -> str:
    settings = load_settings()
    store = ChromaStore(settings=settings, collection_name=COLLECTION_NAME)
    try:
        try:
            store.client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
        store.collection = store.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        store.upsert(
            [
                {
                    "id": CHUNK_ID,
                    "vector": [0.1, 0.2, 0.3],
                    "metadata": {
                        "doc_id": DOCUMENT_ID,
                        "source": "docs/example.pdf",
                        "title": "Example Document",
                        "summary": "Resource e2e summary.",
                        "tags": "rag,mcp",
                        "page": 3,
                        "chunk_id": CHUNK_ID,
                    },
                }
            ]
        )
        yield COLLECTION_NAME
    finally:
        try:
            store.client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
        store.close()


@pytest.fixture()
def mcp_server(resource_collection: str) -> subprocess.Popen:
    process = _start_server()
    yield process
    _terminate(process)


def _start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    for key in (
        "EMBEDDING_API_KEY",
        "OPENAI_API_KEY",
        "DASHSCOPE_API_KEY",
        "LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "VISION_LLM_API_KEY",
    ):
        env.pop(key, None)
    return subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
        env=env,
    )


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _send_jsonrpc(
    process: subprocess.Popen,
    messages: list[dict[str, Any]],
    expected_responses: int,
    timeout: float = 20.0,
) -> list[dict[str, Any]]:
    assert process.stdin is not None
    assert process.stdout is not None

    responses: list[dict[str, Any]] = []
    stop_event = threading.Event()

    def _reader() -> None:
        while not stop_event.is_set():
            line = process.stdout.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                response = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if "id" in response and ("result" in response or "error" in response):
                responses.append(response)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    for message in messages:
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

    deadline = time.time() + timeout
    while len(responses) < expected_responses and time.time() < deadline:
        if process.poll() is not None:
            break
        time.sleep(0.1)

    stop_event.set()
    return responses


def _find_response(responses: list[dict[str, Any]], request_id: int) -> dict[str, Any] | None:
    return next((response for response in responses if response.get("id") == request_id), None)


def _resource_payload(response: dict[str, Any]) -> dict[str, Any]:
    text = response["result"]["contents"][0]["text"]
    return json.loads(text)


@pytest.mark.e2e
def test_resources_list_returns_collection_resource(mcp_server: subprocess.Popen) -> None:
    responses = _send_jsonrpc(
        mcp_server,
        [
            INIT_REQUEST,
            INITIALIZED_NOTIFICATION,
            {"jsonrpc": "2.0", "id": 2, "method": "resources/list", "params": {}},
        ],
        expected_responses=2,
    )

    list_response = _find_response(responses, 2)
    assert list_response is not None
    resources = list_response["result"]["resources"]
    uris = {resource["uri"] for resource in resources}
    assert f"rag://collections/{COLLECTION_NAME}" in uris


@pytest.mark.e2e
def test_resources_read_collection_resource(mcp_server: subprocess.Popen) -> None:
    responses = _send_jsonrpc(
        mcp_server,
        [
            INIT_REQUEST,
            INITIALIZED_NOTIFICATION,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "resources/read",
                "params": {"uri": f"rag://collections/{COLLECTION_NAME}"},
            },
        ],
        expected_responses=2,
    )

    read_response = _find_response(responses, 2)
    assert read_response is not None
    payload = _resource_payload(read_response)
    assert payload["type"] == "collection"
    assert payload["collection_name"] == COLLECTION_NAME
    assert payload["chunk_count"] == 1


@pytest.mark.e2e
def test_resources_read_chunk_resource(mcp_server: subprocess.Popen) -> None:
    responses = _send_jsonrpc(
        mcp_server,
        [
            INIT_REQUEST,
            INITIALIZED_NOTIFICATION,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "resources/read",
                "params": {
                    "uri": f"rag://collections/{COLLECTION_NAME}/chunks/{CHUNK_ID}"
                },
            },
        ],
        expected_responses=2,
    )

    read_response = _find_response(responses, 2)
    assert read_response is not None
    payload = _resource_payload(read_response)
    assert payload["type"] == "chunk"
    assert payload["chunk_id"] == CHUNK_ID
    assert "chunk-one" in payload["text"]


@pytest.mark.e2e
def test_resources_read_missing_resource_returns_controlled_error(
    mcp_server: subprocess.Popen,
) -> None:
    responses = _send_jsonrpc(
        mcp_server,
        [
            INIT_REQUEST,
            INITIALIZED_NOTIFICATION,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "resources/read",
                "params": {
                    "uri": f"rag://collections/{COLLECTION_NAME}/chunks/missing"
                },
            },
        ],
        expected_responses=2,
    )

    read_response = _find_response(responses, 2)
    assert read_response is not None
    payload = _resource_payload(read_response)
    assert payload["error_code"] == "CHUNK_NOT_FOUND"
