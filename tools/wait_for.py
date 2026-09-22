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
import subprocess
from typing import Optional


def _stat_fields(pid: int):
    """(state, starttime) for `pid`, or None when it is gone.

    `/proc/<pid>/stat` where there is a /proc; `ps` where there is not. macOS and the BSDs
    have no /proc, so the read raised, this returned None, and `producer_alive` answered
    False for EVERY pid — including a process plainly running. That made
    `certified_checkpoint.py` decide its producer was "already gone at start" every time and
    exit after one empty checkpoint, which in turn made the certifier refuse to start at all
    (`6442fe30` requires a checkpointer holding the repository). Certification on the Mac was
    impossible until this was portable (2026-09-22).

    `ps -o state=,lstart=` gives the same two facts with the same meaning: a state letter
    whose "Z" is a zombie, and a start time that is stable for one process and different for
    a recycled pid — which is what the caller compares.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            rest = f.read().split(")", 1)[1].split()
        return rest[0], rest[19]          # field 3 = state, field 22 = starttime (clock ticks since boot)
    except FileNotFoundError:
        pass                              # the pid is gone, OR this platform has no /proc
    except OSError:
        return None
    if os.path.isdir("/proc"):            # there IS a /proc and the pid was not in it
        return None
    try:
        out = subprocess.run(["ps", "-o", "state=,lstart=", "-p", str(int(pid))],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    line = (out.stdout or "").strip()
    if out.returncode != 0 or not line:
        return None
    parts = line.split(None, 1)
    if len(parts) < 2:
        return None
    return parts[0][:1], parts[1].strip()


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


def marker_present(path: str, pattern: "re.Pattern[str]", skip_lines: int = 0) -> bool:
    """True when a line at or after `skip_lines` matches. A marker that was
    already in the record before the wait began is not this wait's marker: on
    2026-09-13 a waiter for `mochi rc=` returned at once on the afternoon's
    perturbed run's line, hours before the run it watched had ended."""
    try:
        with open(path, errors="replace") as f:
            for i, line in enumerate(f):
                if i >= skip_lines and pattern.search(line):
                    return True
            return False
    except FileNotFoundError:
        return False


def line_count(path: str) -> int:
    try:
        with open(path, errors="replace") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def last_sign_of_life(path: str) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(os.path.getmtime(path)))
    except OSError:
        return "never (the file does not exist)"


def wait_for(path: str, marker: str, producer_pid: Optional[int], poll: float = 30.0,
             heartbeat: Optional[str] = None, stale_after: Optional[float] = None,
             timeout: Optional[float] = None, out=sys.stderr, from_now: bool = False) -> int:
    """0 = marker seen; 3 = producer gone without it; 4 = heartbeat stale;
    5 = timeout (only when one is given — a chain has none). `from_now`: only a
    line appended after this call counts — for a record that accumulates the
    same marker run after run (a campaign's RUN.md), where a line already there
    belongs to an earlier run. Off by default: a waiter armed AFTER its producer
    wrote the marker must still see it, or it would refuse a finished job."""
    pattern = re.compile(marker)
    skip = line_count(path) if from_now else 0
    name = producer_name(producer_pid) if producer_pid else "(no producer given)"
    t0 = time.monotonic()
    while True:
        if marker_present(path, pattern, skip):
            return 0
        if producer_pid is not None and not producer_alive(producer_pid):
            # One more read: the producer may have written the marker as its last act.
            if marker_present(path, pattern, skip):
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
    p.add_argument("--from-now", action="store_true",
                   help="count only a marker line appended after the wait begins (a record that "
                        "accumulates the same marker run after run); default: any line")
    a = p.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    return wait_for(a.file, a.marker, a.producer_pid, a.poll, a.heartbeat, a.stale_after, a.timeout,
                    from_now=a.from_now)


if __name__ == "__main__":
    sys.exit(main())
