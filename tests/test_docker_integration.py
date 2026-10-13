"""Real-Docker checks of the hardening. Skipped automatically when Docker or the image is missing.

    bash scripts/build_sandbox.sh        # builds coding-agent-sandbox:latest
    pytest tests/test_docker_integration.py -v
"""
import shutil
import subprocess
import sys

import pytest

from sandbox.runner import SandboxConfig, run_in_sandbox

IMAGE = SandboxConfig().image


def _have_image() -> bool:
    if not shutil.which("docker"):
        return False
    r = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    return r.returncode == 0


pytestmark = [pytest.mark.skipif(not _have_image(), reason="docker or sandbox image not available"), pytest.mark.asyncio]
CFG = SandboxConfig(mode="docker", timeout=30)


async def sh(cmd: str, files=None):
    return await run_in_sandbox(files or {}, cmd, CFG)


async def test_runtimes_present_and_runs_as_non_root():
    r = await sh("python --version && node --version && id -u")
    assert r.exit_code == 0 and r.stdout.strip().splitlines()[-1] == "10001"


async def test_python_tests_run_offline_with_preinstalled_pytest():
    files = {"test_a.py": "def test_ok():\n    assert 1 + 1 == 2\n"}
    r = await sh("pytest -q", files)
    assert r.exit_code == 0 and "1 passed" in r.stdout


async def test_no_network():
    r = await sh("python -c \"import socket; socket.create_connection(('1.1.1.1', 80), timeout=3)\"")
    assert r.exit_code != 0


async def test_root_filesystem_is_read_only_but_workspace_is_writable():
    assert (await sh("touch /etc/pwned")).exit_code != 0
    assert (await sh("echo hi > out.txt && cat out.txt")).exit_code == 0


async def test_ptrace_blocked_by_seccomp():
    code = (
        "import ctypes, os\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "r = libc.ptrace(0, 0, 0, 0)  # PTRACE_TRACEME\n"
        "print('ret', r, 'errno', ctypes.get_errno())\n"
    )
    r = await sh("python p.py", {"p.py": code})
    assert r.exit_code == 0 and "ret -1" in r.stdout  # EPERM from seccomp, not success


async def test_namespace_and_mount_syscalls_denied():
    code = (
        "import ctypes\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "CLONE_NEWUSER = 0x10000000\n"
        "print('unshare', libc.unshare(CLONE_NEWUSER))\n"
        "print('mount', libc.mount(b'none', b'/mnt', b'tmpfs', 0, None))\n"
    )
    r = await sh("python p.py", {"p.py": code})
    assert "unshare -1" in r.stdout and "mount -1" in r.stdout


async def test_all_capabilities_dropped():
    r = await sh("grep -E 'Cap(Eff|Prm|Bnd)' /proc/self/status")
    assert r.exit_code == 0
    for line in r.stdout.strip().splitlines():
        assert line.split()[-1] == "0000000000000000", line


async def test_fork_bomb_is_contained_by_pids_limit():
    r = await run_in_sandbox({}, ":(){ :|:& };:", SandboxConfig(mode="docker", timeout=5, pids_limit=64))
    assert r.timed_out or r.exit_code != 0  # contained: host stays up, job ends
