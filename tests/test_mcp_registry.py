import pytest
from mcp.server.fastmcp import FastMCP

from mcp_server.registry import autodiscover, build_spec, mcp_tool, parse_docstring, register_all


def test_parse_docstring_extracts_params_and_summary():
    def f(a: int, b: str = "x"):
        """Do a thing.

        Args:
            a: The first number,
                continued here.
            b: A label.

        Returns:
            nothing
        """

    summary, params = parse_docstring(f.__doc__)
    assert summary == "Do a thing."
    assert params == {"a": "The first number, continued here.", "b": "A label."}


def test_build_spec_schema_required_defaults_and_descriptions():
    @mcp_tool
    async def sample(path: str, count: int = 3, tags: list[str] | None = None) -> dict:
        """Sample tool.

        Args:
            path: Where to look.
            count: How many.
            tags: Optional labels.
        """

    spec = build_spec(sample)
    s = spec.input_schema
    assert spec.name == "sample" and spec.description == "Sample tool."
    assert s["required"] == ["path"]
    assert s["properties"]["path"]["description"] == "Where to look."
    assert s["properties"]["count"]["default"] == 3
    assert "title" not in s and "title" not in s["properties"]["path"]


def test_autodiscover_finds_all_builtin_tools():
    names = {s.name for s in autodiscover()}
    assert {
        "run_in_sandbox", "terminal_start", "terminal_logs", "terminal_kill",
        "browse_page", "ui_check", "deploy_faas", "list_deployments", "undeploy_faas",
    } <= names


@pytest.mark.asyncio
async def test_generated_schema_matches_what_fastmcp_advertises():
    mcp = FastMCP("t")
    specs = {s.name: s for s in register_all(mcp)}
    advertised = {t.name: t for t in await mcp.list_tools()}
    assert set(advertised) == set(specs)
    for name, spec in specs.items():
        theirs = advertised[name].inputSchema
        assert set(theirs.get("properties", {})) == set(spec.input_schema.get("properties", {})), name
        assert sorted(theirs.get("required", [])) == sorted(spec.input_schema.get("required", [])), name
    # docstring descriptions reached the wire schema
    cmd = advertised["run_in_sandbox"].inputSchema["properties"]["cmd"]
    assert "shell command" in cmd.get("description", "").lower()
