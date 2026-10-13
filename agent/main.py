"""CLI: run a requirement through the agent with a human approval gate before deploy.

    python -m agent.main "Build a FastAPI service with GET /health" --thread demo1
    python -m agent.main --thread demo1 --resume        # continue after a crash / restart
"""
from __future__ import annotations

import argparse
import asyncio
import os
import uuid

from .graph import (
    approve,
    build_graph,
    list_checkpoints,
    postgres_checkpointer,
    reject,
    rollback,
    thread_config,
)
from .llm import ModelRouter
from .tools import MCPToolClient


async def run(args: argparse.Namespace) -> None:
    router = ModelRouter.from_env()
    tools = MCPToolClient(os.getenv("MCP_URL"))
    async with postgres_checkpointer() as saver:
        graph = build_graph(router, tools, saver)
        cfg = thread_config(args.thread)

        if args.resume:
            await graph.ainvoke(None, cfg)
        else:
            await graph.ainvoke({"requirement": args.requirement, "max_attempts": args.max_attempts}, cfg)

        while True:
            snap = await graph.aget_state(cfg)
            if not snap.next:
                break
            if "deployer" in snap.next:
                v = snap.values
                print("\n=== Awaiting approval to deploy ===")
                print("files:", ", ".join(v.get("files", {})))
                print("last run exit code:", (v.get("exec_result") or {}).get("exit_code"))
                choice = input("[a]pprove / [r]eject / [b]ack (rollback to a checkpoint): ").strip().lower()
                if choice == "a":
                    await approve(graph, args.thread)
                elif choice == "b":
                    cps = await list_checkpoints(graph, args.thread)
                    for i, c in enumerate(cps):
                        print(i, c)
                    idx = int(input("checkpoint index: "))
                    new_cfg = await rollback(graph, args.thread, cps[idx]["checkpoint_id"])
                    cfg = new_cfg
                else:
                    await reject(graph, args.thread)
            await graph.ainvoke(None, cfg)

        final = (await graph.aget_state(cfg)).values
        print("\nstatus:", final.get("status"))
        if final.get("deploy_url"):
            print("deployed at:", final["deploy_url"])
        if final.get("failure_reason"):
            print("failure:", final["failure_reason"])
        print("token usage by tier:", router.usage)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("requirement", nargs="?", default="")
    p.add_argument("--thread", default=f"t-{uuid.uuid4().hex[:8]}")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not args.resume and not args.requirement:
        p.error("requirement is required unless --resume")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
