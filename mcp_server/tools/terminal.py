"""Terminal tools. All commands run inside an ephemeral sandbox (see sandbox/runner.py)."""
from __future__ import annotations

from typing import Any

from mcp_server.registry import mcp_tool
from sandbox.runner import SandboxConfig, job_manager, run_in_sandbox as _run


@mcp_tool
async def run_in_sandbox(files: dict[str, str], cmd: str, runtime: str = "python", timeout: int = 60) -> dict[str, Any]:
    """Write files into a fresh sandbox, run one shell command, and return its result.

    Args:
        files: Mapping of relative file path to full text content.
        cmd: Shell command to run from the project root (e.g. "pytest -q").
        runtime: Language runtime the project needs, "python" or "node".
        timeout: Wall-clock limit in seconds before the run is killed.
    """
    cfg = SandboxConfig.from_env()
    if runtime not in ("python", "node"):
        raise ValueError("runtime must be 'python' or 'node'")
    timeout = max(1, min(int(timeout), 600))
    res = await _run(files, cmd, cfg, timeout=timeout)
    return res.as_dict()


@mcp_tool
async def terminal_start(files: dict[str, str], cmd: str, timeout: int = 600) -> dict[str, Any]:
    """Start a long-running command in a sandbox and return immediately with a job id and PID.

    Args:
        files: Mapping of relative file path to full text content.
        cmd: Shell command to run from the project root.
        timeout: Wall-clock limit in seconds before the job is killed.
    """
    job = await job_manager().start(files, cmd, timeout=max(1, min(int(timeout), 3600)))
    return {"job_id": job.id, "pid": job.pid}


@mcp_tool
async def terminal_logs(job_id: str, offset: int = 0) -> dict[str, Any]:
    """Fetch output produced since ``offset`` for a job started with terminal_start.

    Args:
        job_id: Id returned by terminal_start.
        offset: Value of next_offset from the previous call (0 for the first call).
    """
    return job_manager().logs(job_id, offset)


@mcp_tool
async def terminal_kill(job_id: str) -> dict[str, Any]:
    """Kill a running job started with terminal_start.

    Args:
        job_id: Id returned by terminal_start.
    """
    return {"killed": await job_manager().kill(job_id)}
