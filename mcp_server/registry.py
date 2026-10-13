"""Zero-boilerplate tool registration with generated JSON Schema.

Adding a tool = drop a file in ``mcp_server/tools/`` and decorate an (async)
function with ``@mcp_tool``.  At startup ``autodiscover`` imports every module
in the package, and ``register_all`` builds, for each function:

  * a pydantic model from its signature (types, defaults, required-ness),
  * per-parameter descriptions parsed from the Google-style ``Args:`` docstring,
  * the JSON Schema served to MCP clients.

The docstring descriptions are pushed into the signature as
``Annotated[T, Field(description=...)]`` so the schema FastMCP advertises is the
same one we generate (checked in tests).
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
import typing
from dataclasses import dataclass
from typing import Annotated, Any, Callable

from pydantic import Field, create_model

_TOOLS: dict[str, "ToolSpec"] = {}


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    input_schema: dict[str, Any]


def mcp_tool(fn: Callable[..., Any] | None = None, *, name: str | None = None):
    """Mark a function as an MCP tool (collected by ``register_all``)."""

    def deco(f: Callable[..., Any]) -> Callable[..., Any]:
        f.__mcp_tool_name__ = name or f.__name__  # type: ignore[attr-defined]
        return f

    return deco(fn) if fn is not None else deco


_ARGS = re.compile(r"^\s*Args:\s*$", re.M)
_PARAM = re.compile(r"^\s{2,}(\w+)(?:\s*\([^)]*\))?:\s*(.+)$")


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Return (summary, {param: description}) from a Google-style docstring."""
    if not doc:
        return "", {}
    doc = inspect.cleandoc(doc)
    m = _ARGS.search(doc)
    head = doc[: m.start()].strip() if m else doc.strip()
    params: dict[str, str] = {}
    if m:
        current: str | None = None
        for line in doc[m.end():].splitlines():
            if not line.strip():
                if current:
                    continue
                continue
            pm = _PARAM.match(line)
            if pm and (len(line) - len(line.lstrip())) <= 4:
                current = pm.group(1)
                params[current] = pm.group(2).strip()
            elif current and line.startswith(" " * 6):
                params[current] += " " + line.strip()
            elif not line.startswith(" "):
                break  # next section (Returns:, Raises:, ...)
    return head, params


def build_spec(fn: Callable[..., Any]) -> ToolSpec:
    name = getattr(fn, "__mcp_tool_name__", fn.__name__)
    summary, pdocs = parse_docstring(fn.__doc__)
    hints = typing.get_type_hints(fn)
    sig = inspect.signature(fn)

    fields: dict[str, Any] = {}
    new_params: list[inspect.Parameter] = []
    for pname, p in sig.parameters.items():
        if pname in ("self", "ctx"):
            continue
        ann = hints.get(pname, Any)
        desc = pdocs.get(pname)
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[pname] = (ann, Field(default, description=desc) if desc else default)
        annotated = Annotated[ann, Field(description=desc)] if desc else ann
        new_params.append(p.replace(annotation=annotated))

    model = create_model(f"{name}_args", **fields)  # type: ignore[call-overload]
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)

    # expose the enriched signature so FastMCP emits the same schema
    fn.__signature__ = sig.replace(parameters=new_params)  # type: ignore[attr-defined]
    return ToolSpec(name=name, description=summary or name, fn=fn, input_schema=schema)


def autodiscover(package: str = "mcp_server.tools") -> list[ToolSpec]:
    pkg = importlib.import_module(package)
    specs: list[ToolSpec] = []
    for mod_info in pkgutil.iter_modules(pkg.__path__):
        mod = importlib.import_module(f"{package}.{mod_info.name}")
        for _, obj in inspect.getmembers(mod, inspect.isfunction):
            if hasattr(obj, "__mcp_tool_name__") and obj.__module__ == mod.__name__:
                specs.append(build_spec(obj))
    names = [s.name for s in specs]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise RuntimeError(f"duplicate MCP tool names: {sorted(dupes)}")
    return sorted(specs, key=lambda s: s.name)


def register_all(mcp: Any, package: str = "mcp_server.tools") -> list[ToolSpec]:
    specs = autodiscover(package)
    for s in specs:
        mcp.add_tool(s.fn, name=s.name, description=s.description)
        _TOOLS[s.name] = s
    return specs


def registered() -> dict[str, ToolSpec]:
    return dict(_TOOLS)
