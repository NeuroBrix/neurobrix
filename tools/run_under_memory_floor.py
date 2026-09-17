#!/usr/bin/env python3
"""Run a job and STOP IT at a memory floor, instead of letting the OS kill it.

This machine is Hocine's working Mac and the Parallels VM runs permanently; both
are normal conditions and no measurement may assume otherwise. On 2026-09-17 a
hat-s-x4 attempt was allowed to run until available memory reached 127 MB and
the OS killed it — three times. A footprint is measured UNDER a ceiling, never
by driving the machine into the ground.

The watchdog is armed BEFORE the child starts, samples `available_mb` every
second, and on the first sample below the floor sends SIGTERM, then SIGKILL to
the child's whole process group. It reports the lowest figure reached, so a cell
that cannot fit is recorded as NOT MEASURED with its number.

    run_under_memory_floor.py --floor-mb 4096 --label hat-s-x4 -- <cmd> [args...]

Exit codes: the child's, or 42 when the floor stopped it.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

FLOOR_STOPPED = 42


def available_mb() -> int | None:
    """Our own authority for this quantity, never `Pages free` alone."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from neurobrix.core.host_memory import memory_state
        v = memory_state().available_mb
        if v is not None:
            return int(v)
    except Exception:                                  # noqa: BLE001
        pass
    try:                                               # fallback: vm_stat
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        pages = 0
        for key in ("Pages free", "Pages speculative", "Pages inactive", "Pages purgeable"):
            for line in out.splitlines():
                if line.startswith(key):
                    pages += int(line.split(":")[1].strip().rstrip("."))
                    break
        return int(pages * 16384 / 1e6)
    except Exception:                                  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-mb", type=int, default=4096)
    ap.add_argument("--label", default="job")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        print("no command given", file=sys.stderr)
        return 2

    start = available_mb()
    if start is None:
        print("[floor] cannot read available memory on this platform — "
              "REFUSING to start, because the floor could not be enforced")
        return 2
    if start < a.floor_mb:
        print(f"[floor] {a.label}: NOT STARTED — available {start} MB is already "
              f"below the {a.floor_mb} MB floor")
        return FLOOR_STOPPED

    print(f"[floor] {a.label}: armed at {a.floor_mb} MB, {start} MB available now "
          f"(Parallels VM running: part of normal conditions)", flush=True)
    proc = subprocess.Popen(cmd, start_new_session=True)
    state = {"low": start, "stopped": False}

    def watch():
        while proc.poll() is None:
            v = available_mb()
            if v is not None:
                state["low"] = min(state["low"], v)
                if v < a.floor_mb:
                    state["stopped"] = True
                    print(f"[floor] {a.label}: STOPPING — available {v} MB "
                          f"below the {a.floor_mb} MB floor", flush=True)
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        for _ in range(10):
                            if proc.poll() is not None:
                                break
                            time.sleep(0.5)
                        if proc.poll() is None:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return
            time.sleep(a.interval)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    rc = proc.wait()
    t.join(timeout=5)
    print(f"[floor] {a.label}: lowest available {state['low']} MB "
          f"(floor {a.floor_mb} MB){' — STOPPED BY THE FLOOR' if state['stopped'] else ''}",
          flush=True)
    return FLOOR_STOPPED if state["stopped"] else rc


if __name__ == "__main__":
    sys.exit(main())
