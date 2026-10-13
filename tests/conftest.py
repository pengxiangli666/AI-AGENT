import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server():
    """A real MCP server process (SSE + streamable HTTP) with the sandbox in local mode."""
    port = _free_port()
    env = {**os.environ, "MCP_HOST": "127.0.0.1", "MCP_PORT": str(port), "SANDBOX_MODE": "local", "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen([sys.executable, "-m", "mcp_server.server"], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            proc.terminate()
            raise RuntimeError("server did not start:\n" + (proc.stdout.read().decode() if proc.stdout else ""))
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=10)
