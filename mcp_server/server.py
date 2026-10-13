"""MCP server: SSE + streamable-HTTP transports behind a FastAPI app.

    python -m mcp_server.server            # serves on MCP_HOST:MCP_PORT
      SSE endpoint:             /sse           (MCP_URL for the agent)
      streamable HTTP endpoint: /mcp
      health:                   /healthz
      tool schemas (debug):     /tools
"""
from __future__ import annotations

import contextlib
import os

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

from .registry import register_all


def build_mcp() -> tuple[FastMCP, list]:
    mcp = FastMCP(
        "coding-agent-tools",
        host=os.getenv("MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("MCP_PORT", "8765")),
    )
    specs = register_all(mcp)
    return mcp, specs


def build_app() -> FastAPI:
    mcp, specs = build_mcp()
    sse_app = mcp.sse_app()
    http_app = mcp.streamable_http_app()

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        # the streamable-HTTP transport needs its session manager running
        async with mcp.session_manager.run():
            yield

    app = FastAPI(title="coding-agent MCP", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "tools": len(specs)}

    @app.get("/tools")
    async def tools():
        return [{"name": s.name, "description": s.description, "inputSchema": s.input_schema} for s in specs]

    # Order matters: Starlette matches mounts first-come, and a Mount("/") swallows every path.
    # Our own routes (/healthz, /tools) are registered above; the specific mount goes before "/".
    app.mount("/http", http_app)     # exposes /http/mcp  (streamable HTTP)
    app.mount("/", sse_app)          # exposes /sse and /messages/  (SSE)
    return app


def main() -> None:
    import uvicorn

    uvicorn.run(build_app(), host=os.getenv("MCP_HOST", "127.0.0.1"), port=int(os.getenv("MCP_PORT", "8765")), log_level="info")


if __name__ == "__main__":
    main()
