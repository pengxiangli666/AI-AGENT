"""StateGraph assembly, checkpointer factory and rollback helper."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from langgraph.graph import END, START, StateGraph

from .llm import ModelRouter
from .nodes import after_exec, after_review, give_up, make_nodes
from .state import AgentState
from .tools import ToolClient


def build_graph(router: ModelRouter, tools: ToolClient, checkpointer: Any, *, hitl: bool = True):
    n = make_nodes(router, tools)
    g = StateGraph(AgentState)
    for name in ("planner", "coder", "reviewer", "executor", "deployer"):
        g.add_node(name, n[name])
    g.add_node("give_up", give_up)

    g.add_edge(START, "planner")
    g.add_edge("planner", "coder")
    g.add_edge("coder", "reviewer")
    g.add_conditional_edges("reviewer", after_review, {"executor": "executor", "coder": "coder", "give_up": "give_up"})
    g.add_conditional_edges("executor", after_exec, {"deployer": "deployer", "coder": "coder", "give_up": "give_up"})
    g.add_edge("deployer", END)
    g.add_edge("give_up", END)

    return g.compile(checkpointer=checkpointer, interrupt_before=["deployer"] if hitl else [])


@asynccontextmanager
async def postgres_checkpointer(uri: str | None = None) -> AsyncIterator[Any]:
    """AsyncPostgresSaver so a crashed/long-running task resumes from its last checkpoint."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    uri = uri or os.environ["POSTGRES_URI"]
    async with AsyncPostgresSaver.from_conn_string(uri) as saver:
        await saver.setup()  # idempotent: creates checkpoint tables on first run
        yield saver


def thread_config(thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
    if checkpoint_id:
        cfg["configurable"]["checkpoint_id"] = checkpoint_id
    return cfg


async def list_checkpoints(graph: Any, thread_id: str) -> list[dict[str, Any]]:
    """Newest first: [{checkpoint_id, next, status, attempt}]."""
    out = []
    async for snap in graph.aget_state_history(thread_config(thread_id)):
        out.append(
            {
                "checkpoint_id": snap.config["configurable"]["checkpoint_id"],
                "next": list(snap.next),
                "status": snap.values.get("status"),
                "attempt": snap.values.get("attempt"),
            }
        )
    return out


async def rollback(graph: Any, thread_id: str, checkpoint_id: str, patch: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fork the thread from an earlier checkpoint (optionally patching state) and return the new config.

    Resume with ``await graph.ainvoke(None, new_config)``.
    """
    target = None
    async for snap in graph.aget_state_history(thread_config(thread_id)):
        if snap.config["configurable"]["checkpoint_id"] == checkpoint_id:
            target = snap.config  # carries checkpoint_ns etc. that the saver requires
            break
    if target is None:
        raise KeyError(f"checkpoint {checkpoint_id} not found in thread {thread_id}")
    return await graph.aupdate_state(target, patch or {})


async def approve(graph: Any, thread_id: str) -> None:
    await graph.aupdate_state(thread_config(thread_id), {"approved_by_human": True})


async def reject(graph: Any, thread_id: str) -> None:
    await graph.aupdate_state(thread_config(thread_id), {"approved_by_human": False})
