"""Ephemeral sandbox execution with async process management and log streaming.

``SANDBOX_MODE=docker`` (default): every run is a throw-away container with
    --cap-drop ALL, a custom seccomp profile, no-new-privileges, read-only rootfs,
    no network, non-root user, and pid/memory/cpu limits.
``SANDBOX_MODE=local``: dev-only escape hatch for machines without Docker. It
    gives NO isolation beyond ulimits - never use it with untrusted model output.
"""
from __future__ import annotations

import asyncio
import os
import posixpath
import shlex
import shutil
import signal
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

OUTPUT_CAP = 200_000  # bytes kept per stream
SECCOMP_DEFAULT = str(Path(__file__).with_name("seccomp.json"))


@dataclass
class SandboxConfig:
    mode: str = "docker"
    image: str = "coding-agent-sandbox:latest"
    timeout: int = 60
    memory: str = "512m"
    cpus: str = "1.0"
    pids_limit: int = 128
    network: str = "none"           # "none" | "bridge"
    seccomp_profile: str = SECCOMP_DEFAULT
    user: str = "10001:10001"

    @classmethod
    def from_env(cls) -> "SandboxConfig":
        e = os.environ
        return cls(
            mode=e.get("SANDBOX_MODE", "docker"),
            image=e.get("SANDBOX_IMAGE", cls.image),
            timeout=int(e.get("SANDBOX_TIMEOUT", cls.timeout)),
            memory=e.get("SANDBOX_MEMORY", cls.memory),
            cpus=e.get("SANDBOX_CPUS", cls.cpus),
            pids_limit=int(e.get("SANDBOX_PIDS", cls.pids_limit)),
            network=e.get("SANDBOX_NETWORK", cls.network),
            seccomp_profile=e.get("SANDBOX_SECCOMP", SECCOMP_DEFAULT),
        )


@dataclass
class RunResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    pid: int | None = None
    duration_s: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class UnsafePath(ValueError):
    pass


def materialize_workspace(files: dict[str, str], base: str | None = None) -> Path:
    """Write ``files`` into a fresh temp dir. Rejects anything that could escape it."""
    root = Path(tempfile.mkdtemp(prefix="agent-ws-", dir=base))
    os.chmod(root, 0o777)  # container user (non-root) must be able to write caches
    root_resolved = root.resolve()
    try:
        for rel, content in files.items():
            norm = posixpath.normpath(rel)
            if norm.startswith("/") or norm.startswith("../") or norm == ".." or "\x00" in norm:
                raise UnsafePath(rel)
            dest = (root / norm).resolve()
            if root_resolved not in dest.parents:
                raise UnsafePath(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(dest.parent, 0o777)
            dest.write_text(content, encoding="utf-8")
            os.chmod(dest, 0o644)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return root


def build_docker_cmd(workdir: str | Path, cmd: str, cfg: SandboxConfig, name: str) -> list[str]:
    return [
        "docker", "run", "--rm", "--name", name,
        "--network", cfg.network,
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--security-opt", f"seccomp={cfg.seccomp_profile}",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=64m,mode=1777",
        "--pids-limit", str(cfg.pids_limit),
        "--memory", cfg.memory, "--memory-swap", cfg.memory,
        "--cpus", cfg.cpus,
        "--user", cfg.user,
        "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-v", f"{workdir}:/workspace:rw",
        "-w", "/workspace",
        cfg.image, "sh", "-c", cmd,
    ]


def build_local_cmd(cmd: str, timeout: int) -> list[str]:
    limits = f"ulimit -t {timeout + 5} -f 20000 -n 256 2>/dev/null; "
    return ["bash", "-c", limits + cmd]


async def _pump(stream: asyncio.StreamReader, buf: bytearray, on_chunk: Callable[[str, str], Awaitable[None] | None] | None, which: str) -> None:
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return
        if len(buf) < OUTPUT_CAP:
            buf.extend(chunk[: OUTPUT_CAP - len(buf)])
        if on_chunk:
            r = on_chunk(which, chunk.decode("utf-8", "replace"))
            if asyncio.iscoroutine(r):
                await r


async def run_streaming(
    argv: list[str],
    *,
    timeout: int,
    cwd: str | None = None,
    on_chunk: Callable[[str, str], Awaitable[None] | None] | None = None,
    on_start: Callable[[int], None] | None = None,
    container_name: str | None = None,
) -> RunResult:
    """Run ``argv`` in its own process group, streaming output, enforcing a wall-clock timeout."""
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    if on_start:
        on_start(proc.pid)
    out, err = bytearray(), bytearray()
    pumps = asyncio.gather(_pump(proc.stdout, out, on_chunk, "stdout"), _pump(proc.stderr, err, on_chunk, "stderr"))
    timed_out = False
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        await _kill(proc, container_name)
    finally:
        try:
            await asyncio.wait_for(pumps, timeout=5)
        except asyncio.TimeoutError:
            pumps.cancel()
    code = proc.returncode if proc.returncode is not None else -9
    return RunResult(
        exit_code=124 if timed_out else code,
        stdout=out.decode("utf-8", "replace"),
        stderr=err.decode("utf-8", "replace"),
        timed_out=timed_out,
        pid=proc.pid,
        duration_s=round(time.monotonic() - t0, 3),
    )


async def _kill(proc: asyncio.subprocess.Process, container_name: str | None) -> None:
    if container_name:  # killing the docker CLI does not reliably stop the container
        k = await asyncio.create_subprocess_exec(
            "docker", "kill", container_name,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await k.wait()
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await proc.wait()


async def run_in_sandbox(
    files: dict[str, str],
    cmd: str,
    cfg: SandboxConfig | None = None,
    *,
    on_chunk: Callable[[str, str], Awaitable[None] | None] | None = None,
    on_start: Callable[[int], None] | None = None,
    timeout: int | None = None,
) -> RunResult:
    cfg = cfg or SandboxConfig.from_env()
    timeout = timeout or cfg.timeout
    ws = materialize_workspace(files)
    try:
        if cfg.mode == "docker":
            name = f"agent-sbx-{uuid.uuid4().hex[:12]}"
            argv = build_docker_cmd(ws, cmd, cfg, name)
            return await run_streaming(argv, timeout=timeout, on_chunk=on_chunk, on_start=on_start, container_name=name)
        if cfg.mode == "local":
            return await run_streaming(build_local_cmd(cmd, timeout), timeout=timeout, cwd=str(ws), on_chunk=on_chunk, on_start=on_start)
        raise ValueError(f"unknown SANDBOX_MODE {cfg.mode!r}")
    finally:
        shutil.rmtree(ws, ignore_errors=True)


# --------------------------------------------------------------------------------------------
# Long-running jobs: start now, poll logs later (async PID + incremental log streaming)
# --------------------------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    cmd: str
    pid: int | None = None
    log: list[tuple[str, str]] = field(default_factory=list)  # (stream, text)
    task: asyncio.Task | None = None
    result: RunResult | None = None

    @property
    def running(self) -> bool:
        return self.result is None


class JobManager:
    def __init__(self, cfg: SandboxConfig | None = None, max_jobs: int = 16):
        self.cfg = cfg
        self.max_jobs = max_jobs
        self.jobs: dict[str, Job] = {}

    async def start(self, files: dict[str, str], cmd: str, timeout: int | None = None) -> Job:
        if sum(1 for j in self.jobs.values() if j.running) >= self.max_jobs:
            raise RuntimeError("too many concurrent jobs")
        job = Job(id=uuid.uuid4().hex[:10], cmd=cmd)
        self.jobs[job.id] = job

        def on_chunk(stream: str, text: str) -> None:
            job.log.append((stream, text))

        def on_start(pid: int) -> None:
            job.pid = pid

        async def _go() -> None:
            try:
                job.result = await run_in_sandbox(files, cmd, self.cfg, on_chunk=on_chunk, on_start=on_start, timeout=timeout)
            except Exception as e:  # surface setup errors through the same channel
                job.result = RunResult(exit_code=-1, stdout="", stderr=f"{type(e).__name__}: {e}")

        job.task = asyncio.create_task(_go())
        return job

    def logs(self, job_id: str, offset: int = 0) -> dict:
        job = self.jobs[job_id]
        chunks = job.log[offset:]
        return {
            "job_id": job_id,
            "pid": job.pid,
            "running": job.running,
            "next_offset": offset + len(chunks),
            "output": [{"stream": s, "text": t} for s, t in chunks],
            "exit_code": None if job.result is None else job.result.exit_code,
            "timed_out": bool(job.result and job.result.timed_out),
        }

    async def kill(self, job_id: str) -> bool:
        job = self.jobs[job_id]
        if not job.running or job.pid is None:
            return False
        try:
            os.killpg(job.pid, signal.SIGKILL)
        except ProcessLookupError:
            return False
        return True


_manager: JobManager | None = None


def job_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager


def shell_quote(s: str) -> str:
    return shlex.quote(s)
