#!/usr/bin/env python3
"""A waiter that holds its producer — register entry 55.

    tools/wait_for.py --file RUN.md --marker '== rejeu 2 termine' --producer-pid 974227
    tools/wait_for.py --file RUN.md --marker '== convert rc=0' --producer-pid 1234 --poll 60

Exit 0 the moment the marker is in the file. Exit 3 — a REFUSAL, with the
producer's name and its last sign of life — the moment the producer is gone
and the marker is not there: a waiter that only asks "is the result here
yet?" cannot tell *not yet* from *never* (four V100s idle for four hours on
2026-09-13 behind a marker whose writer had been killed after its job
succeeded). Exit 4 when a heartbeat file is given and goes stale.

The producer is the process that will WRITE the marker (the chain script),
not the job it runs. A producer that exits without the marker is a refusal
even if it exited 0: the contract is the marker, and the marker is absent.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import time
from typing import Optional


def _stat_fields(pid: int):
    """(state, starttime) from /proc/<pid>/stat, or None when the pid is gone."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            rest = f.read().split(")", 1)[1].split()
        return rest[0], rest[19]          # field 3 = state, field 22 = starttime (clock ticks since boot)
    except OSError:
        return None


_BIRTH: dict = {}                          # pid -> starttime seen at the first check


def producer_alive(pid: int) -> bool:
    """True while THE SAME process lives: the pid must exist, not be a zombie,
    and carry the start time it had when first watched — a recycled pid is a
    different process and reads as gone (a door, not a census)."""
    fields = _stat_fields(pid)
    if fields is None:
        return False
    state, birth = fields
    if state == "Z":
        return False
    first = _BIRTH.setdefault(pid, birth)
    return first == birth


def producer_name(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()[:160] or f"pid {pid}"
    except OSError:
        return f"pid {pid}"


def marker_present(path: str, pattern: "re.Pattern[str]") -> bool:
    try:
        with open(path, errors="replace") as f:
            return any(pattern.search(line) for line in f)
    except FileNotFoundError:
        return False


def last_sign_of_life(path: str) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(os.path.getmtime(path)))
    except OSError:
        return "never (the file does not exist)"


def wait_for(path: str, marker: str, producer_pid: Optional[int], poll: float = 30.0,
             heartbeat: Optional[str] = None, stale_after: Optional[float] = None,
             timeout: Optional[float] = None, out=sys.stderr) -> int:
    """0 = marker seen; 3 = producer gone without it; 4 = heartbeat stale;
    5 = timeout (only when one is given — a chain has none)."""
    pattern = re.compile(marker)
    name = producer_name(producer_pid) if producer_pid else "(no producer given)"
    t0 = time.monotonic()
    while True:
        if marker_present(path, pattern):
            return 0
        if producer_pid is not None and not producer_alive(producer_pid):
            # One more read: the producer may have written the marker as its last act.
            if marker_present(path, pattern):
                return 0
            print(f"[wait_for] REFUSED: producer {producer_pid} ({name}) is gone and "
                  f"'{marker}' is not in {path}; last write to the file {last_sign_of_life(path)}. "
                  f"Nothing will write it — the waiter stops instead of starving.", file=out, flush=True)
            return 3
        if heartbeat and stale_after is not None:
            try:
                age = time.time() - os.path.getmtime(heartbeat)
            except OSError:
                age = float("inf")
            if age > stale_after:
                print(f"[wait_for] REFUSED: heartbeat {heartbeat} stale for {age:.0f} s "
                      f"(> {stale_after:.0f}); producer {producer_pid} ({name}) may be hung.", file=out, flush=True)
                return 4
        if timeout is not None and time.monotonic() - t0 > timeout:
            print(f"[wait_for] timeout after {timeout:.0f} s waiting for '{marker}' in {path}; "
                  f"producer {producer_pid} ({name}) still alive.", file=out, flush=True)
            return 5
        time.sleep(poll)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="A waiter that holds its producer (register entry 55).")
    p.add_argument("--file", required=True, help="the record the marker is written to")
    p.add_argument("--marker", required=True, help="a regular expression matched line by line")
    p.add_argument("--producer-pid", type=int, required=True,
                   help="the process that will write the marker; its death without it is a refusal")
    p.add_argument("--poll", type=float, default=30.0)
    p.add_argument("--heartbeat", default=None, help="a file the producer touches; stale = hung")
    p.add_argument("--stale-after", type=float, default=None, help="seconds before a heartbeat is stale")
    p.add_argument("--timeout", type=float, default=None, help="a bound, for a test; a chain has none")
    a = p.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    return wait_for(a.file, a.marker, a.producer_pid, a.poll, a.heartbeat, a.stale_after, a.timeout)


if __name__ == "__main__":
    sys.exit(main())
