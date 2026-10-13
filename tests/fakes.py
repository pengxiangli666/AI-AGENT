"""Scripted fakes so the graph can be exercised without OpenAI, MCP or Docker."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


class ScriptedLLM:
    """Returns queued replies in order; records the prompts it saw."""

    def __init__(self, replies: list[Any]):
        self.replies = list(replies)
        self.seen: list[Any] = []

    async def ainvoke(self, messages):
        self.seen.append(messages)
        r = self.replies.pop(0)
        text = r if isinstance(r, str) else json.dumps(r)
        return SimpleNamespace(content=text, usage_metadata={"input_tokens": 10, "output_tokens": 5})


class FakeTools:
    def __init__(self, exec_results: list[dict[str, Any]] | None = None):
        self.exec_results = list(exec_results or [{"exit_code": 0, "stdout": "ok", "stderr": ""}])
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "run_in_sandbox":
            return self.exec_results.pop(0) if len(self.exec_results) > 1 else self.exec_results[0]
        if name == "deploy_faas":
            return {"url": "http://app-1234.localhost:8080"}
        raise KeyError(name)
