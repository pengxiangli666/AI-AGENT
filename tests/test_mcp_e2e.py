"""Real MCP round trip: server subprocess <-SSE / streamable-HTTP-> agent's MCPToolClient.

Runs the sandbox in SANDBOX_MODE=local (no Docker needed); the Docker path is covered by
argument-level tests and by scripts/smoke_docker.sh on a machine with Docker.
"""
import sys

import httpx
import pytest

from agent.tools import MCPToolClient


@pytest.mark.asyncio
async def test_sse_lists_tools_and_runs_code(server):
    c = MCPToolClient(f"http://127.0.0.1:{server}/sse")
    names = {t["name"] for t in await c.list_tools()}
    assert {"run_in_sandbox", "deploy_faas", "browse_page", "terminal_start"} <= names

    res = await c.call("run_in_sandbox", {"files": {"t.py": "print(6*7)"}, "cmd": f"{sys.executable} t.py"})
    assert res["exit_code"] == 0 and res["stdout"].strip() == "42"

    bad = await c.call("run_in_sandbox", {"files": {"t.py": "raise SystemExit(5)"}, "cmd": f"{sys.executable} t.py"})
    assert bad["exit_code"] == 5


@pytest.mark.asyncio
async def test_streamable_http_transport_also_works(server):
    c = MCPToolClient(f"http://127.0.0.1:{server}/http/mcp", transport="http")
    res = await c.call("run_in_sandbox", {"files": {}, "cmd": "echo via-http"})
    assert res["exit_code"] == 0 and "via-http" in res["stdout"]


@pytest.mark.asyncio
async def test_tool_errors_surface_as_exceptions(server):
    c = MCPToolClient(f"http://127.0.0.1:{server}/sse")
    with pytest.raises(RuntimeError, match="refused"):
        await c.call("browse_page", {"url": "http://169.254.169.254/latest/meta-data/"})
    with pytest.raises(RuntimeError):
        await c.call("run_in_sandbox", {"files": {"../x": "y"}, "cmd": "true"})


def test_http_debug_endpoints(server):
    tools = httpx.get(f"http://127.0.0.1:{server}/tools").json()
    run = next(t for t in tools if t["name"] == "run_in_sandbox")
    assert run["inputSchema"]["required"] == ["files", "cmd"]
