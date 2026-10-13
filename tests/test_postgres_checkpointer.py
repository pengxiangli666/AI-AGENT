"""PostgresSaver durability: a crashed run resumes from its last checkpoint in a *new process*-like setup.

Needs a real Postgres:
    TEST_POSTGRES_URI=postgresql://agent:agent@localhost:5432/agent pytest tests/test_postgres_checkpointer.py
(docker compose up -d postgres)  Skipped when the variable is unset.
"""
import os
import uuid

import pytest

from agent.graph import approve, build_graph, list_checkpoints, postgres_checkpointer, thread_config
from agent.llm import ModelRouter
from tests.fakes import FakeTools, ScriptedLLM

URI = os.getenv("TEST_POSTGRES_URI")
pytestmark = [pytest.mark.skipif(not URI, reason="TEST_POSTGRES_URI not set"), pytest.mark.asyncio]

PLAN = {"plan": ["do it"], "runtime": "python", "run_cmd": "pytest -q"}
FILES = {"files": {"a.py": "x = 1", "test_a.py": "def test(): pass"}}
OK = {"approved": True, "issues": [], "feedback": ""}


class CrashOnceTools(FakeTools):
    def __init__(self):
        super().__init__([{"exit_code": 0, "stdout": "ok", "stderr": ""}])
        self.crashed = False

    async def call(self, name, arguments):
        if name == "run_in_sandbox" and not self.crashed:
            self.crashed = True
            raise ConnectionError("sandbox worker died")
        return await super().call(name, arguments)


def routers(router_replies, gen_replies):
    return ModelRouter({"router": ScriptedLLM(router_replies), "generator": ScriptedLLM(gen_replies)})


async def test_crash_mid_run_then_resume_does_not_redo_earlier_steps():
    tid = f"crash-{uuid.uuid4().hex[:8]}"
    cfg = thread_config(tid)
    tools = CrashOnceTools()

    # --- process 1: planner, coder, reviewer succeed; executor crashes
    r1 = routers([PLAN, OK], [FILES])
    async with postgres_checkpointer(URI) as saver:
        g1 = build_graph(r1, tools, saver)
        with pytest.raises(ConnectionError):
            await g1.ainvoke({"requirement": "x"}, cfg)

    # --- process 2: fresh graph, fresh connection, LLMs with NO replies queued -> any LLM call would blow up
    r2 = routers([], [])
    async with postgres_checkpointer(URI) as saver:
        g2 = build_graph(r2, tools, saver)
        snap = await g2.aget_state(cfg)
        assert snap.next == ("executor",) and snap.values["files"]       # state survived the "restart"
        await g2.ainvoke(None, cfg)                                        # resume
        snap = await g2.aget_state(cfg)
        assert snap.next == ("deployer",) and snap.values["exec_result"]["exit_code"] == 0
        assert r2.usage.calls == {}                                        # nothing was re-planned / re-coded

        await approve(g2, tid)
        out = await g2.ainvoke(None, cfg)
        assert out["status"] == "deployed"
        assert len(await list_checkpoints(g2, tid)) >= 5


async def test_pending_approval_survives_restart():
    tid = f"hitl-{uuid.uuid4().hex[:8]}"
    cfg = thread_config(tid)
    async with postgres_checkpointer(URI) as saver:
        g = build_graph(routers([PLAN, OK], [FILES]), FakeTools(), saver)
        await g.ainvoke({"requirement": "x"}, cfg)
    async with postgres_checkpointer(URI) as saver:                        # e.g. approver comes back next day
        g = build_graph(routers([], []), FakeTools(), saver)
        assert (await g.aget_state(cfg)).next == ("deployer",)
