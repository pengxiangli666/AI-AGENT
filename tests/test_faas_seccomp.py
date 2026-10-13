import json
from pathlib import Path

import pytest

from sandbox.faas import (
    DeployError,
    FaasConfig,
    build_run_args,
    deploy,
    detect_start_cmd,
    list_deployments,
    render_nginx_conf,
    teardown,
)

PROFILE = json.loads((Path(__file__).parents[1] / "sandbox" / "seccomp.json").read_text())


# ---------------------------------------------------------------- seccomp profile
def _names():
    return {n for e in PROFILE["syscalls"] for n in e["names"]}


def test_seccomp_is_default_deny():
    assert PROFILE["defaultAction"] == "SCMP_ACT_ERRNO"


def test_seccomp_removes_process_inspection_and_legacy_calls():
    for s in ["ptrace", "process_vm_readv", "process_vm_writev", "kcmp", "pidfd_getfd", "modify_ldt", "name_to_handle_at"]:
        assert s not in _names(), s


def test_seccomp_keeps_what_python_and_node_need():
    for s in ["read", "write", "openat", "mmap", "mprotect", "futex", "epoll_wait", "execve", "statx", "rseq", "getrandom", "clock_gettime"]:
        assert s in _names(), s


def test_seccomp_socket_families_limited():
    fams = {e["args"][0]["value"] for e in PROFILE["syscalls"] if e["names"] == ["socket"]}
    assert fams == {1, 2, 10, 16}


def test_seccomp_namespace_and_mount_calls_stay_cap_gated():
    gated = {n for e in PROFILE["syscalls"] if e.get("includes", {}).get("caps") for n in e["names"]}
    assert {"mount", "unshare", "setns", "bpf"} <= gated
    plain = {n for e in PROFILE["syscalls"] if not e.get("includes") and not e.get("args") for n in e["names"]}
    assert not ({"mount", "unshare", "setns", "bpf", "reboot"} & plain)


# ---------------------------------------------------------------- start command detection
@pytest.mark.parametrize(
    "files,runtime,expected",
    [
        ({"main.py": "app = FastAPI()"}, "python", "uvicorn main:app --host 0.0.0.0 --port 8000"),
        ({"app.py": "app = Flask(__name__)"}, "python", "flask --app app run --host 0.0.0.0 --port 8000"),
        ({"server.py": "print(1)"}, "python", "python server.py"),
        ({"package.json": '{"scripts": {"start": "node x.js"}}'}, "node", "npm start"),
        ({"index.js": ""}, "node", "node index.js"),
        ({"Procfile": "web: gunicorn app:app -b 0.0.0.0:8000", "app.py": ""}, "python", "gunicorn app:app -b 0.0.0.0:8000"),
    ],
)
def test_detect_start_cmd(files, runtime, expected):
    assert detect_start_cmd(files, runtime) == expected


def test_detect_start_cmd_unknown_raises():
    with pytest.raises(DeployError):
        detect_start_cmd({"README.md": "x"}, "python")


# ---------------------------------------------------------------- nginx + docker args
def test_nginx_conf_routes_by_host_and_uses_docker_dns():
    conf = render_nginx_conf("app-deadbeef", "apps.test")
    assert "server_name app-deadbeef.apps.test;" in conf
    assert "set $upstream http://app-deadbeef:8000;" in conf and "proxy_pass $upstream;" in conf


def test_run_args_are_hardened(tmp_path):
    args = build_run_args("app-deadbeef", tmp_path, "uvicorn main:app", FaasConfig())
    j = " ".join(args)
    for flag in ["--cap-drop ALL", "no-new-privileges", "seccomp=", "--read-only", "--user 10001:10001", ":/app:ro", "--network agent-faas"]:
        assert flag in j, flag
    assert "-p " not in j and "--publish" not in j      # never published straight to the host
    assert "--privileged" not in args


# ---------------------------------------------------------------- deploy orchestration (fake docker)
class FakeDocker:
    def __init__(self, ready_after=0, run_ok=True, running=True, nginx_t_ok=True):
        self.calls, self.ready_after, self.run_ok, self.running, self.nginx_t_ok = [], ready_after, run_ok, running, nginx_t_ok
        self.wget_calls = 0

    async def __call__(self, *args):
        self.calls.append(args)
        if args[0] == "run":
            return (0, "cid", "") if self.run_ok else (125, "", "no such image")
        if args[0] == "inspect":
            return 0, "true\n" if self.running else "false\n", ""
        if args[0] == "logs":
            return 0, "Traceback: boom", ""
        if args[0] == "exec" and "wget" in args:
            self.wget_calls += 1
            return (0 if self.wget_calls > self.ready_after else 4), "", ""
        if args[0] == "exec" and "-t" in args:
            return (0, "", "") if self.nginx_t_ok else (1, "", "bad config")
        return 0, "", ""


def cfg_for(tmp_path):
    return FaasConfig(domain="apps.test", nginx_conf_dir=str(tmp_path / "conf.d"), apps_dir=str(tmp_path / "apps"), ready_timeout=3)


@pytest.mark.asyncio
async def test_deploy_success_writes_conf_then_validates_then_reloads(tmp_path):
    d, cfg = FakeDocker(ready_after=2), cfg_for(tmp_path)
    dep = await deploy({"main.py": "app = FastAPI()"}, "python", cfg, docker=d)
    assert dep.url == f"http://{dep.app_id}.apps.test:8080"
    assert (tmp_path / "conf.d" / f"{dep.app_id}.conf").exists()
    assert (tmp_path / "apps" / dep.app_id / "main.py").exists()
    verbs = [c for c in d.calls if c[0] == "exec" and "nginx" in c]
    assert [("-t" in v) or ("reload" in v) for v in verbs] == [True, True] and "-t" in verbs[0] and "reload" in verbs[1]
    assert [x.app_id for x in list_deployments(cfg)] == [dep.app_id]


@pytest.mark.asyncio
async def test_deploy_rolls_back_when_app_crashes(tmp_path):
    d, cfg = FakeDocker(running=False), cfg_for(tmp_path)
    with pytest.raises(DeployError, match="exited during startup"):
        await deploy({"main.py": "app = FastAPI()"}, "python", cfg, docker=d)
    assert list((tmp_path / "conf.d").glob("*.conf")) == [] and list((tmp_path / "apps").iterdir()) == []
    assert any(c[0] == "rm" for c in d.calls)


@pytest.mark.asyncio
async def test_deploy_rolls_back_when_nginx_rejects_config(tmp_path):
    d, cfg = FakeDocker(nginx_t_ok=False), cfg_for(tmp_path)
    with pytest.raises(DeployError, match="nginx config invalid"):
        await deploy({"main.py": "app = FastAPI()"}, "python", cfg, docker=d)
    assert list((tmp_path / "conf.d").glob("*.conf")) == []


@pytest.mark.asyncio
async def test_deploy_rejects_path_traversal_and_leaves_no_files(tmp_path):
    cfg = cfg_for(tmp_path)
    with pytest.raises(DeployError, match="unsafe file path"):
        await deploy({"main.py": "app = FastAPI()", "../evil": "x"}, "python", cfg, docker=FakeDocker())
    assert list((tmp_path / "apps").iterdir()) == []


@pytest.mark.asyncio
async def test_teardown_removes_route_and_validates_id(tmp_path):
    d, cfg = FakeDocker(), cfg_for(tmp_path)
    dep = await deploy({"main.py": "app = FastAPI()"}, "python", cfg, docker=d)
    assert await teardown(dep.app_id, cfg, docker=d) is True
    assert list((tmp_path / "conf.d").glob("*.conf")) == []
    with pytest.raises(ValueError):
        await teardown("../../etc", cfg, docker=d)
