"""Whole pipeline with a REAL MCP server and REAL sandboxed execution; only the LLM is scripted.

Proves the loop end to end: coder writes buggy code -> real test run fails with a real traceback ->
traceback is fed back to the coder -> fixed code passes -> graph pauses for human approval.
"""
import sys

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import build_graph, thread_config
from agent.llm import ModelRouter
from agent.tools import MCPToolClient
from tests.fakes import ScriptedLLM

TEST_PY = "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\nif __name__ == '__main__':\n    unittest.main()\n"
PLAN = {"plan": ["write add()", "unit test it"], "runtime": "python", "run_cmd": f"{sys.executable} -m unittest -q test_calc.py"}
BUGGY = {"files": {"calc.py": "def add(a, b):\n    return a - b\n", "test_calc.py": TEST_PY}}
FIXED = {"files": {"calc.py": "def add(a, b):\n    return a + b\n", "test_calc.py": TEST_PY}}
OK = {"approved": True, "issues": [], "feedback": ""}


@pytest.mark.asyncio
async def test_real_execution_feeds_traceback_back_to_coder(server):
    router_llm = ScriptedLLM([PLAN, OK, OK])          # planner, reviewer x2
    gen_llm = ScriptedLLM([BUGGY, FIXED])             # coder x2
    router = ModelRouter({"router": router_llm, "generator": gen_llm})
    graph = build_graph(router, MCPToolClient(f"http://127.0.0.1:{server}/sse"), InMemorySaver())
    cfg = thread_config("e2e-1")

    await graph.ainvoke({"requirement": "write add(a, b)", "max_attempts": 3}, cfg)
    snap = await graph.aget_state(cfg)

    assert snap.next == ("deployer",), snap.values.get("history")
    assert snap.values["attempt"] == 2
    assert snap.values["exec_result"]["exit_code"] == 0
    second_coder_prompt = str(gen_llm.seen[1])
    assert "AssertionError" in second_coder_prompt and ("-1 != 5" in second_coder_prompt or "5 != -1" in second_coder_prompt)
    assert snap.values["files"]["calc.py"].strip().endswith("return a + b")
