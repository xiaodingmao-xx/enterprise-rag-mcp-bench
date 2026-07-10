"""Start the production REST API and MCP HTTP Gateway."""

from __future__ import annotations

import uvicorn

from src.core.settings import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run(
        "src.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        factory=False,
    )


if __name__ == "__main__":
    main()
