"""Tiered model routing (OpenAI).

Two tiers:
  * ``router``    - cheap/fast model for planning, review and other logic-heavy
                    but low-token work.
  * ``generator`` - stronger model, used only for code generation.

Routing is a plain, explicit lookup table (``ROLE_TIER``) rather than a
LangChain history wrapper: the choice depends on *which node is calling*, not
on conversation history.  Every call is also counted so you can measure the
real token split between tiers (see ``Usage``) instead of guessing a saving.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

Tier = str  # "router" | "generator"

ROLE_TIER: dict[str, Tier] = {
    "planner": "router",
    "reviewer": "router",
    "deployer": "router",
    "coder": "generator",
}


class ChatModel(Protocol):
    async def ainvoke(self, messages: list[Any]) -> Any: ...


@dataclass
class Usage:
    """Per-tier token accounting, filled from response metadata when present."""
    calls: dict[str, int] = field(default_factory=dict)
    input_tokens: dict[str, int] = field(default_factory=dict)
    output_tokens: dict[str, int] = field(default_factory=dict)

    def record(self, tier: str, response: Any) -> None:
        self.calls[tier] = self.calls.get(tier, 0) + 1
        meta = getattr(response, "usage_metadata", None) or {}
        self.input_tokens[tier] = self.input_tokens.get(tier, 0) + int(meta.get("input_tokens", 0))
        self.output_tokens[tier] = self.output_tokens.get(tier, 0) + int(meta.get("output_tokens", 0))


class ModelRouter:
    def __init__(self, models: dict[Tier, ChatModel] | None = None):
        self._models = models or {}
        self.usage = Usage()

    @classmethod
    def from_env(cls) -> "ModelRouter":
        from langchain_openai import ChatOpenAI

        base = os.getenv("OPENAI_BASE_URL")
        common: dict[str, Any] = {"temperature": 0}
        if base:
            common["base_url"] = base
        return cls(
            {
                "router": ChatOpenAI(model=os.getenv("MODEL_ROUTER", "gpt-4o-mini"), **common),
                "generator": ChatOpenAI(model=os.getenv("MODEL_GENERATOR", "gpt-4o"), **common),
            }
        )

    async def ask(self, role: str, system: str, user: str) -> str:
        tier = ROLE_TIER[role]
        model = self._models[tier]
        resp = await model.ainvoke([("system", system), ("user", user)])
        self.usage.record(tier, resp)
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    async def ask_json(self, role: str, system: str, user: str) -> dict[str, Any]:
        raw = await self.ask(role, system, user)
        return extract_json(raw)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of an LLM reply (tolerates code fences / chatter)."""
    text = text.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"no JSON object in model reply: {text[:200]!r}")
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model reply JSON is not an object")
    return obj
