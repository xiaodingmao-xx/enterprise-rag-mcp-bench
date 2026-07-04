"""Compatibility entry point for the MCP stdio server."""

from __future__ import annotations

import sys

from src.mcp_server.server import main

if __name__ == "__main__":
    sys.exit(main())
