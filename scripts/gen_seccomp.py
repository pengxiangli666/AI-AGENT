#!/usr/bin/env python3
"""Derive sandbox/seccomp.json by hardening Docker's default seccomp profile.

Why derive instead of hand-writing an allowlist: Python/Node on Alpine need ~300
syscalls and a hand-made list breaks in subtle ways (rseq, statx, clone3 fallback,
futex variants ...).  Docker's profile is a maintained, default-deny allowlist
(defaultAction = SCMP_ACT_ERRNO).  We keep it and remove what an untrusted code
runner never needs:

  * ptrace / process_vm_* / kcmp / pidfd_getfd   - cross-process inspection
  * legacy or escape-adjacent calls              - modify_ldt, uselib, ustat, sysfs,
                                                   _sysctl, name_to_handle_at, move_pages,
                                                   migrate_pages
  * socket families other than UNIX/INET/INET6/NETLINK
                                                 - AF_PACKET, AF_ALG, AF_VSOCK, AF_BLUETOOTH ...
                                                   (kernel attack surface)

Cap-gated entries (mount, unshare, setns, bpf, perf_event_open, clone with CLONE_NEW*)
stay gated: the runner always passes --cap-drop ALL, so they are denied.

Usage:  python scripts/gen_seccomp.py [path/to/docker-default.json]
        (downloads moby/profiles default.json when no path is given)
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

SRC_URL = "https://raw.githubusercontent.com/moby/profiles/main/seccomp/default.json"
OUT = Path(__file__).resolve().parents[1] / "sandbox" / "seccomp.json"

DENY = {
    "ptrace", "process_vm_readv", "process_vm_writev", "kcmp", "pidfd_getfd",
    "modify_ldt", "uselib", "ustat", "sysfs", "_sysctl",
    "name_to_handle_at", "move_pages", "migrate_pages",
}
ALLOWED_SOCKET_FAMILIES = {1, 2, 10, 16}  # AF_UNIX, AF_INET, AF_INET6, AF_NETLINK


def harden(profile: dict) -> dict:
    out = []
    for entry in profile["syscalls"]:
        names = entry["names"]
        if names == ["socket"] and entry.get("args"):
            a = entry["args"][0]
            # '< 3' covers AF_UNSPEC/AF_UNIX/AF_INET; keep it but trim to the allowed set below
            if a["op"] == "SCMP_CMP_LT":
                for fam in (1, 2):
                    out.append({**entry, "args": [{"index": 0, "value": fam, "op": "SCMP_CMP_EQ"}]})
                continue
            if a["op"] == "SCMP_CMP_EQ" and a["value"] in ALLOWED_SOCKET_FAMILIES:
                out.append(entry)
            continue
        kept = [n for n in names if n not in DENY]
        if not kept:
            continue
        out.append({**entry, "names": kept})
    hardened = {k: v for k, v in profile.items() if k != "syscalls"}
    hardened["syscalls"] = out
    return hardened


def main() -> None:
    if len(sys.argv) > 1:
        profile = json.loads(Path(sys.argv[1]).read_text())
    else:
        with urllib.request.urlopen(SRC_URL, timeout=30) as r:  # noqa: S310 - fixed https URL
            profile = json.load(r)
    hardened = harden(profile)
    OUT.write_text(json.dumps(hardened, indent=2) + "\n")
    names = {n for e in hardened["syscalls"] for n in e["names"]}
    print(f"wrote {OUT} ({len(names)} distinct syscall names, defaultAction={hardened['defaultAction']})")


if __name__ == "__main__":
    main()
