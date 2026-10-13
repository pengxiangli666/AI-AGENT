"""Graph nodes: Planner -> Coder -> Reviewer -> Executor -> Deployer.

Nodes are built by ``make_nodes`` so the LLM router and tool client are injected
(real OpenAI + MCP in production, scripted fakes in tests).
"""
from __future__ import annotations

import posixpath
from typing import Any

from .llm import ModelRouter
from .state import AgentState
from .tools import ToolClient

MAX_FILES = 40
MAX_FILE_BYTES = 200_000

PLANNER_SYS = (
    "You are the Planner of a software-engineering agent. Break the requirement into "
    "a few concrete implementation steps and choose how the result will be tested.\n"
    'Reply with ONLY a JSON object: {"plan": [str, ...], "runtime": "python"|"node", '
    '"run_cmd": "<single shell command that runs the tests/entrypoint and exits 0 on success>"}'
)

CODER_SYS = (
    "You are the Coder. Produce the COMPLETE project as files, including its tests, so that "
    "run_cmd exits 0 when everything works. If review/execution feedback is given, fix exactly "
    "those problems and return the full updated file set.\n"
    'Reply with ONLY a JSON object: {"files": {"relative/path": "full file content", ...}}. '
    "Paths must be relative, no '..'."
)

REVIEWER_SYS = (
    "You are the Reviewer. Check the files against the requirement and plan: correctness, "
    "missing pieces, obvious bugs, unsafe operations (network exfiltration, deleting outside the "
    "project, hard-coded secrets).\n"
    'Reply with ONLY a JSON object: {"approved": bool, "issues": [str, ...], "feedback": str}'
)


def validate_files(files: Any) -> tuple[dict[str, str], list[str]]:
    """Sanity-check model-produced files before they go anywhere near a sandbox."""
    problems: list[str] = []
    clean: dict[str, str] = {}
    if not isinstance(files, dict) or not files:
        return {}, ["coder returned no files"]
    if len(files) > MAX_FILES:
        problems.append(f"too many files ({len(files)} > {MAX_FILES})")
    for path, content in files.items():
        norm = posixpath.normpath(str(path))
        if norm.startswith("/") or norm == ".." or norm.startswith("../") or "\x00" in norm:
            problems.append(f"unsafe path: {path!r}")
            continue
        if not isinstance(content, str):
            problems.append(f"non-text content for {path!r}")
            continue
        if len(content.encode()) > MAX_FILE_BYTES:
            problems.append(f"file too large: {path!r}")
            continue
        clean[norm] = content
    return clean, problems


def _log(state: AgentState, msg: str) -> list[str]:
    return [*state.get("history", []), msg]


def make_nodes(router: ModelRouter, tools: ToolClient):
    async def planner(state: AgentState) -> dict[str, Any]:
        out = await router.ask_json("planner", PLANNER_SYS, state["requirement"])
        runtime = out.get("runtime") if out.get("runtime") in ("python", "node") else "python"
        return {
            "plan": [str(s) for s in out.get("plan", [])],
            "runtime": runtime,
            "run_cmd": str(out.get("run_cmd") or ("pytest -q" if runtime == "python" else "npm test")),
            "attempt": 0,
            "max_attempts": state.get("max_attempts", 3),
            "status": "coding",
            "history": _log(state, "planner: plan ready"),
        }

    async def coder(state: AgentState) -> dict[str, Any]:
        parts = [
            f"Requirement:\n{state['requirement']}",
            "Plan:\n" + "\n".join(f"- {s}" for s in state.get("plan", [])),
            f"run_cmd: {state.get('run_cmd')}  (runtime: {state.get('runtime')})",
        ]
        review = state.get("review") or {}
        if review and not review.get("approved", True):
            parts.append("Reviewer feedback to fix:\n" + (review.get("feedback") or "") + "\n" + "\n".join(review.get("issues", [])))
        ex = state.get("exec_result") or {}
        if ex and ex.get("exit_code", 0) != 0:
            parts.append(
                f"Last run FAILED (exit {ex.get('exit_code')}, timed_out={ex.get('timed_out', False)}).\n"
                f"stdout:\n{(ex.get('stdout') or '')[-3000:]}\nstderr:\n{(ex.get('stderr') or '')[-3000:]}"
            )
        if state.get("files"):
            parts.append("Current files:\n" + "\n\n".join(f"### {p}\n{c}" for p, c in state["files"].items()))
        out = await router.ask_json("coder", CODER_SYS, "\n\n".join(parts))
        files, problems = validate_files(out.get("files"))
        attempt = state.get("attempt", 0) + 1
        update: dict[str, Any] = {
            "files": files,
            "attempt": attempt,
            "status": "reviewing",
            "history": _log(state, f"coder: attempt {attempt}, {len(files)} files"),
        }
        if problems:
            # Don't ship anything suspicious; hand the problems to the reviewer path as a rejection.
            update["review"] = {"approved": False, "issues": problems, "feedback": "Fix invalid file set."}
            update["status"] = "coding"
        return update

    async def reviewer(state: AgentState) -> dict[str, Any]:
        existing = state.get("review") or {}
        if existing.get("approved") is False and state.get("status") == "coding":
            # coder already produced an invalid file set; skip the LLM call and loop back.
            return {"history": _log(state, "reviewer: skipped (invalid files)")}
        user = (
            f"Requirement:\n{state['requirement']}\n\nPlan:\n" + "\n".join(state.get("plan", []))
            + "\n\nFiles:\n" + "\n\n".join(f"### {p}\n{c}" for p, c in state.get("files", {}).items())
        )
        out = await router.ask_json("reviewer", REVIEWER_SYS, user)
        review = {
            "approved": bool(out.get("approved")),
            "issues": [str(i) for i in out.get("issues", [])],
            "feedback": str(out.get("feedback", "")),
        }
        return {
            "review": review,
            "status": "executing" if review["approved"] else "coding",
            "history": _log(state, f"reviewer: approved={review['approved']}"),
        }

    async def executor(state: AgentState) -> dict[str, Any]:
        res = await tools.call(
            "run_in_sandbox",
            {
                "files": state["files"],
                "cmd": state["run_cmd"],
                "runtime": state.get("runtime", "python"),
            },
        )
        ex = {
            "exit_code": int(res.get("exit_code", 1)),
            "stdout": str(res.get("stdout", "")),
            "stderr": str(res.get("stderr", "")),
            "timed_out": bool(res.get("timed_out", False)),
        }
        ok = ex["exit_code"] == 0
        return {
            "exec_result": ex,
            "review": {"approved": True, "issues": [], "feedback": ""} if ok else state.get("review", {}),
            "status": "awaiting_approval" if ok else "coding",
            "history": _log(state, f"executor: exit={ex['exit_code']}"),
        }

    async def deployer(state: AgentState) -> dict[str, Any]:
        # The graph is compiled with interrupt_before=["deployer"]; a human must resume it.
        # We still refuse to deploy unless approval was explicitly recorded in state.
        if state.get("approved_by_human") is not True:
            return {"status": "rejected", "history": _log(state, "deployer: no human approval, aborted")}
        res = await tools.call(
            "deploy_faas",
            {"files": state["files"], "runtime": state.get("runtime", "python")},
        )
        return {
            "deploy_url": str(res.get("url", "")),
            "status": "deployed",
            "history": _log(state, f"deployer: {res.get('url')}"),
        }

    return {"planner": planner, "coder": coder, "reviewer": reviewer, "executor": executor, "deployer": deployer}


# ---- conditional edges (pure functions, easy to unit-test) -------------------------------------
def after_review(state: AgentState) -> str:
    if (state.get("review") or {}).get("approved"):
        return "executor"
    if state.get("attempt", 0) >= state.get("max_attempts", 3):
        return "give_up"
    return "coder"


def after_exec(state: AgentState) -> str:
    if (state.get("exec_result") or {}).get("exit_code", 1) == 0:
        return "deployer"
    if state.get("attempt", 0) >= state.get("max_attempts", 3):
        return "give_up"
    return "coder"


async def give_up(state: AgentState) -> dict[str, Any]:
    ex = state.get("exec_result") or {}
    why = (state.get("review") or {}).get("feedback") or (ex.get("stderr") or "")[-500:] or "unknown"
    return {
        "status": "failed",
        "failure_reason": f"gave up after {state.get('attempt', 0)} attempts: {why}",
        "history": _log(state, "give_up"),
    }
