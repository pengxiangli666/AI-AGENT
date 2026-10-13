import asyncio
import sys

import pytest

from sandbox.runner import (
    JobManager,
    SandboxConfig,
    UnsafePath,
    build_docker_cmd,
    materialize_workspace,
    run_in_sandbox,
)

LOCAL = SandboxConfig(mode="local", timeout=10)


def test_docker_cmd_has_all_hardening_flags():
    argv = build_docker_cmd("/tmp/ws", "pytest -q", SandboxConfig(), "n1")
    joined = " ".join(argv)
    for flag in ["--cap-drop ALL", "no-new-privileges", "seccomp=", "--read-only", "--network none",
                 "--pids-limit", "--memory", "--user 10001:10001"]:
        assert flag in joined, flag
    assert argv[-3:] == ["sh", "-c", "pytest -q"]
    assert "--privileged" not in argv and "docker.sock" not in joined


def test_materialize_rejects_path_escape(tmp_path):
    for bad in ["../x.py", "/etc/passwd", "a/../../x"]:
        with pytest.raises(UnsafePath):
            materialize_workspace({bad: "x"}, base=str(tmp_path))
    assert list(tmp_path.iterdir()) == []  # failed attempts leave nothing behind


@pytest.mark.asyncio
async def test_local_run_success_and_failure():
    ok = await run_in_sandbox({"a.py": "print('hi')"}, f"{sys.executable} a.py", LOCAL)
    assert ok.exit_code == 0 and ok.stdout.strip() == "hi"
    bad = await run_in_sandbox({"a.py": "import sys; sys.stderr.write('boom'); sys.exit(3)"}, f"{sys.executable} a.py", LOCAL)
    assert bad.exit_code == 3 and "boom" in bad.stderr and not bad.timed_out


@pytest.mark.asyncio
async def test_timeout_kills_process_group():
    res = await run_in_sandbox({}, "sleep 30 & sleep 30", LOCAL, timeout=1)
    assert res.timed_out and res.exit_code == 124 and res.duration_s < 8


@pytest.mark.asyncio
async def test_output_is_streamed_incrementally():
    seen = []
    code = "import time,sys\nfor i in range(3):\n print(i, flush=True); time.sleep(0.2)\n"
    res = await run_in_sandbox({"s.py": code}, f"{sys.executable} s.py", LOCAL, on_chunk=lambda s, t: seen.append((s, t)))
    assert res.exit_code == 0 and len(seen) >= 2  # arrived in more than one chunk


@pytest.mark.asyncio
async def test_job_manager_start_poll_and_kill():
    jm = JobManager(LOCAL)
    job = await jm.start({}, "echo one; sleep 0.3; echo two")
    for _ in range(50):
        if not jm.jobs[job.id].running:
            break
        await asyncio.sleep(0.1)
    first = jm.logs(job.id, 0)
    assert not first["running"] and first["exit_code"] == 0
    text = "".join(c["text"] for c in first["output"])
    assert "one" in text and "two" in text
    assert jm.logs(job.id, first["next_offset"])["output"] == []  # offset advances

    long = await jm.start({}, "sleep 30", timeout=60)
    for _ in range(50):
        if long.pid:
            break
        await asyncio.sleep(0.05)
    assert long.pid and await jm.kill(long.id)
    for _ in range(50):
        if not jm.jobs[long.id].running:
            break
        await asyncio.sleep(0.1)
    assert jm.logs(long.id)["exit_code"] != 0
