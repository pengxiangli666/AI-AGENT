"""Tool access for graph nodes.

Nodes never import sandbox/browser code directly; they talk to a ``ToolClient``.
In production that is an MCP client speaking SSE/HTTP to the MCP server (which
is what decouples the agent core from the privileged terminal/browser
services).  Tests inject a fake that implements the same two methods.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Protocol


class ToolClient(Protocol):
    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class MCPToolClient:
    """Opens a short-lived MCP session per call (simple and restart-safe)."""

    def __init__(self, url: str | None = None, transport: str = "sse"):
        self.url = url or os.getenv("MCP_URL", "http://localhost:8765/sse")
        self.transport = transport

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[Any]:
        from mcp import ClientSession

        if self.transport == "sse":
            from mcp.client.sse import sse_client

            async with sse_client(self.url) as (r, w):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    yield s
        else:
            from mcp.client.streamable_http import streamablehttp_client

            async with streamablehttp_client(self.url) as (r, w, _):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    yield s

    async def list_tools(self) -> list[dict[str, Any]]:
        async with self._session() as s:
            res = await s.list_tools()
            return [{"name": t.name, "description": t.description, "inputSchema": t.inputSchema} for t in res.tools]

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with self._session() as s:
            res = await s.call_tool(name, arguments)
        if res.isError:
            text = "".join(getattr(c, "text", "") for c in res.content)
            raise RuntimeError(f"MCP tool {name} failed: {text}")
        structured = getattr(res, "structuredContent", None)
        if structured:
            return structured.get("result", structured) if isinstance(structured, dict) else {"result": structured}
        text = "".join(getattr(c, "text", "") for c in res.content)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}
