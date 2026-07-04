"""E2E tests for MCP server startup entry points.

These tests guard the packaging/runtime contract:

- ``python -m src.mcp_server.server`` starts the real stdio MCP server.
- ``mcp-server`` starts the same server when the console script is installed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
REQUIRED_TOOLS = {
    "query_knowledge_hub",
    "list_collections",
    "get_document_summary",
}

INIT_REQUEST: dict[str, Any] = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {
            "name": "entrypoint-e2e-client",
            "version": "0.1.0",
        },
    },
}

INITIALIZED_NOTIFICATION: dict[str, Any] = {
    "jsonrpc": "2.0",
    "method": "notifications/initialized",
}

TOOLS_LIST_REQUEST: dict[str, Any] = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {},
}


def _start_process(command: list[str]) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(
        command,
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
        try:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            pytest.fail(f"MCP entrypoint exited before accepting JSON-RPC: {exc}")

    deadline = time.time() + timeout
    while len(responses) < expected_responses and time.time() < deadline:
        if process.poll() is not None:
            break
        time.sleep(0.1)

    stop_event.set()
    return responses


def _find_response(responses: list[dict[str, Any]], request_id: int) -> dict[str, Any] | None:
    return next((response for response in responses if response.get("id") == request_id), None)


def _assert_initialize_and_tools_list(command: list[str]) -> None:
    process = _start_process(command)
    try:
        responses = _send_jsonrpc(
            process,
            [INIT_REQUEST, INITIALIZED_NOTIFICATION, TOOLS_LIST_REQUEST],
            expected_responses=2,
        )

        init_response = _find_response(responses, 1)
        assert init_response is not None, f"Missing initialize response: {responses}"
        assert "result" in init_response, f"initialize failed: {init_response}"

        init_result = init_response["result"]
        assert "serverInfo" in init_result
        assert "capabilities" in init_result
        assert init_result["capabilities"].get("tools") is not None

        tools_response = _find_response(responses, 2)
        assert tools_response is not None, f"Missing tools/list response: {responses}"
        assert "result" in tools_response, f"tools/list failed: {tools_response}"

        tools = tools_response["result"].get("tools", [])
        assert isinstance(tools, list)
        tool_names = {tool.get("name") for tool in tools}
        assert REQUIRED_TOOLS.issubset(tool_names)
    finally:
        _terminate(process)


def _find_console_script() -> str | None:
    script_name = "mcp-server.exe" if os.name == "nt" else "mcp-server"
    sibling_script = Path(sys.executable).resolve().parent / script_name
    if sibling_script.exists():
        return str(sibling_script)
    return shutil.which("mcp-server")


@pytest.mark.e2e
def test_pyproject_console_script_points_to_real_server() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'mcp-server = "src.mcp_server.server:main"' in pyproject
    assert 'mcp-server = "main:main"' not in pyproject


@pytest.mark.e2e
def test_module_entrypoint_starts_mcp_server_and_lists_tools() -> None:
    _assert_initialize_and_tools_list([sys.executable, "-m", "src.mcp_server.server"])


@pytest.mark.e2e
def test_console_script_starts_mcp_server_and_lists_tools() -> None:
    script = _find_console_script()
    if script is None:
        pytest.skip("mcp-server console script is not installed in this environment")

    _assert_initialize_and_tools_list([script])
