import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import approve, build_graph, list_checkpoints, reject, rollback, thread_config
from agent.llm import ModelRouter, extract_json
from agent.nodes import validate_files
from tests.fakes import FakeTools, ScriptedLLM

PLAN = {"plan": ["write add()", "test it"], "runtime": "python", "run_cmd": "pytest -q"}
FILES_V1 = {"files": {"app.py": "def add(a,b): return a-b", "test_app.py": "from app import add\ndef test(): assert add(1,2)==3"}}
FILES_V2 = {"files": {"app.py": "def add(a,b): return a+b", "test_app.py": "from app import add\ndef test(): assert add(1,2)==3"}}
OK = {"approved": True, "issues": [], "feedback": ""}
BAD = {"approved": False, "issues": ["add subtracts"], "feedback": "fix add"}


def make(router_replies, gen_replies, exec_results, hitl=True):
    router_llm, gen_llm = ScriptedLLM(router_replies), ScriptedLLM(gen_replies)
    router = ModelRouter({"router": router_llm, "generator": gen_llm})
    tools = FakeTools(exec_results)
    graph = build_graph(router, tools, InMemorySaver(), hitl=hitl)
    return graph, router, tools, router_llm, gen_llm


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('noise {"a": 3} trailing') == {"a": 3}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_validate_files_blocks_unsafe_paths():
    clean, problems = validate_files({"../evil.py": "x", "/etc/passwd": "x", "ok/a.py": "print(1)"})
    assert set(clean) == {"ok/a.py"} and len(problems) == 2
    assert validate_files({})[1]


@pytest.mark.asyncio
async def test_happy_path_pauses_before_deploy_then_deploys_after_approval():
    # router tier calls: planner, reviewer ; generator tier: coder
    graph, router, tools, *_ = make([PLAN, OK], [FILES_V2], [{"exit_code": 0, "stdout": "1 passed", "stderr": ""}])
    cfg = thread_config("t1")
    await graph.ainvoke({"requirement": "add fn"}, cfg)
    snap = await graph.aget_state(cfg)
    assert snap.next == ("deployer",)                       # paused for HITL
    assert not any(c[0] == "deploy_faas" for c in tools.calls)

    await approve(graph, "t1")
    out = await graph.ainvoke(None, cfg)
    assert out["status"] == "deployed" and out["deploy_url"].startswith("http://app-")
    # tier split: 2 cheap calls, 1 strong call
    assert router.usage.calls == {"router": 2, "generator": 1}


@pytest.mark.asyncio
async def test_resume_without_approval_never_deploys():
    graph, _, tools, *_ = make([PLAN, OK], [FILES_V2], [{"exit_code": 0, "stdout": "", "stderr": ""}])
    cfg = thread_config("t2")
    await graph.ainvoke({"requirement": "x"}, cfg)
    await reject(graph, "t2")
    out = await graph.ainvoke(None, cfg)
    assert out["status"] == "rejected"
    assert not any(c[0] == "deploy_faas" for c in tools.calls)


@pytest.mark.asyncio
async def test_self_correction_loop_review_then_exec_failure_then_success():
    graph, router, tools, router_llm, gen_llm = make(
        [PLAN, BAD, OK, OK],                                     # planner, reviewer(bad), reviewer(ok), reviewer(ok)
        [FILES_V1, FILES_V1, FILES_V2],                          # 3 coder passes
        [{"exit_code": 1, "stdout": "", "stderr": "AssertionError"}, {"exit_code": 0, "stdout": "ok", "stderr": ""}],
    )
    cfg = thread_config("t3")
    await graph.ainvoke({"requirement": "add fn", "max_attempts": 5}, cfg)
    snap = await graph.aget_state(cfg)
    assert snap.next == ("deployer",)
    assert snap.values["attempt"] == 3
    # the coder saw reviewer feedback on pass 2 and the stderr on pass 3
    assert "fix add" in str(gen_llm.seen[1])
    assert "AssertionError" in str(gen_llm.seen[2])


@pytest.mark.asyncio
async def test_attempt_cap_prevents_infinite_loop():
    graph, *_ = make([PLAN, BAD, BAD], [FILES_V1, FILES_V1], [{"exit_code": 1, "stdout": "", "stderr": "x"}])
    cfg = thread_config("t4")
    out = await graph.ainvoke({"requirement": "x", "max_attempts": 2}, cfg)
    assert out["status"] == "failed" and "gave up after 2 attempts" in out["failure_reason"]


@pytest.mark.asyncio
async def test_invalid_file_set_loops_back_without_calling_reviewer_llm():
    bad_files = {"files": {"../escape.py": "x"}}
    graph, _, _, router_llm, _ = make([PLAN, OK], [bad_files, FILES_V2], [{"exit_code": 0, "stdout": "", "stderr": ""}])
    cfg = thread_config("t5")
    await graph.ainvoke({"requirement": "x", "max_attempts": 3}, cfg)
    snap = await graph.aget_state(cfg)
    assert snap.next == ("deployer",)
    assert len(router_llm.seen) == 2        # planner + ONE reviewer call (first pass skipped)


@pytest.mark.asyncio
async def test_rollback_forks_from_earlier_checkpoint():
    graph, *_ = make([PLAN, OK], [FILES_V2], [{"exit_code": 0, "stdout": "", "stderr": ""}])
    cfg = thread_config("t6")
    await graph.ainvoke({"requirement": "x"}, cfg)
    cps = await list_checkpoints(graph, "t6")
    before_exec = next(c for c in cps if c["next"] == ["executor"])
    new_cfg = await rollback(graph, "t6", before_exec["checkpoint_id"])
    snap = await graph.aget_state(new_cfg)
    assert snap.next == ("executor",)
