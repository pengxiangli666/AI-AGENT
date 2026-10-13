"""Global state schema persisted by the checkpointer (PostgresSaver)."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class ReviewResult(TypedDict, total=False):
    approved: bool
    feedback: str
    issues: list[str]


class ExecResult(TypedDict, total=False):
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


class AgentState(TypedDict, total=False):
    # --- inputs ---
    requirement: str

    # --- planner output ---
    plan: list[str]              # ordered, human-readable steps
    run_cmd: str                 # how to run / test the project in the sandbox
    runtime: str                 # "python" | "node"

    # --- coder output ---
    files: dict[str, str]        # relative path -> file content

    # --- reviewer / executor feedback (drives the self-correction loop) ---
    review: ReviewResult
    exec_result: ExecResult

    # --- loop control ---
    attempt: int                 # number of coder passes so far
    max_attempts: int            # hard cap that prevents infinite loops
    status: str                  # planning|coding|reviewing|executing|awaiting_approval|deployed|failed|rejected
    failure_reason: str

    # --- deploy output ---
    deploy_url: str
    approved_by_human: bool

    # --- audit trail (append-only) ---
    messages: Annotated[list[Any], add_messages]
    history: list[str]
