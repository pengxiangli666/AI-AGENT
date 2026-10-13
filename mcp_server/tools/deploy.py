"""FaaS deployment tool: run generated code as an app container and publish it behind Nginx."""
from __future__ import annotations

from typing import Any

from mcp_server.registry import mcp_tool
from sandbox.faas import FaasConfig, deploy as _deploy, list_deployments as _list, teardown as _teardown


@mcp_tool
async def deploy_faas(files: dict[str, str], runtime: str = "python") -> dict[str, Any]:
    """Deploy a project as a sandboxed app and return the URL it is served at.

    Args:
        files: Mapping of relative file path to full text content. Must contain the app entrypoint.
        runtime: Language runtime the project needs, "python" or "node".
    """
    if runtime not in ("python", "node"):
        raise ValueError("runtime must be 'python' or 'node'")
    d = await _deploy(files, runtime, FaasConfig.from_env())
    return {"url": d.url, "app_id": d.app_id, "host_port": d.host_port}


@mcp_tool
async def list_deployments() -> dict[str, Any]:
    """List currently running deployed apps."""
    return {"deployments": [d.__dict__ for d in _list()]}


@mcp_tool
async def undeploy_faas(app_id: str) -> dict[str, Any]:
    """Stop a deployed app and remove its Nginx route.

    Args:
        app_id: The app_id returned by deploy_faas.
    """
    return {"removed": await _teardown(app_id, FaasConfig.from_env())}
