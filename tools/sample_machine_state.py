#!/usr/bin/env python3
"""Sample the machine while a cell runs, so a kill leaves its cause on disk.

The state at the START of a cell says what it began with. The state at the
MOMENT it died says why. Six months later only the second explains a result,
and it is the one nobody has unless it was written as it happened: macOS keeps
no history of `vm_stat`, so the E cells measured before 2026-09-10 have lost
theirs and it cannot be reconstructed.

A green obtained on a half-occupied machine is not the same green as one
obtained on a free machine. Measured on this machine within a single hour,
with nobody intervening, the largest third-party resident read 8 615 MB, then
3 353 MB, then 6 123 MB — so no cached figure describes it ten minutes later.

    tools/sample_machine_state.py --while pytest --out run.mem
    tools/sample_machine_state.py --for 600 --out cell.mem

`--while` follows a process pattern and stops when nothing matches any more;
`--for` samples for a fixed number of seconds. One of the two is required:
a sampler with no end is a file that grows until someone notices.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

_PAGE_LABELS = ("Pages free", "Pages inactive", "Pages purgeable",
                "Pages speculative")


def _mb(label: str, vm_stat: str, page_bytes: int) -> int:
    m = re.search(rf"{label}:\s+(\d+)", vm_stat)
    return int(m.group(1)) * page_bytes // (1024 * 1024) if m else -1


def _residents(ps_out: str, count: int = 2) -> str:
    rows = []
    for line in ps_out.splitlines()[1:]:
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        mb = int(parts[0]) // 1024
        if mb < 50:
            break                       # the list is sorted; nothing below matters
        rows.append(f"{parts[1].strip().rsplit('/', 1)[-1][:22]}={mb}Mo")
        if len(rows) >= count:
            break
    return " ".join(rows)


def sample() -> str:
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = re.search(r"page size of (\d+) bytes", vm)
    page_bytes = int(page.group(1)) if page else 4096
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                          capture_output=True, text=True).stdout
    used = re.search(r"used = ([\d.]+)M", swap)
    ps_out = subprocess.run(["ps", "-Ao", "rss,comm", "-r"],
                            capture_output=True, text=True).stdout
    free = _mb("Pages free", vm, page_bytes)
    inactive = _mb("Pages inactive", vm, page_bytes)
    purgeable = _mb("Pages purgeable", vm, page_bytes)
    speculative = _mb("Pages speculative", vm, page_bytes)
    available = free + inactive + purgeable + speculative
    return (f"{time.strftime('%H:%M:%S')} "
            f"dispo={available}Mo (libre={free} inactif={inactive} "
            f"purgeable={purgeable} spéculatif={speculative}) "
            f"swap={used.group(1) if used else '?'}Mo "
            f"{_residents(ps_out)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--while", dest="pattern",
                    help="follow this pgrep -f pattern; stop when it is gone")
    ap.add_argument("--for", dest="seconds", type=float,
                    help="sample for this many seconds")
    ap.add_argument("--every", type=float, default=20.0)
    ap.add_argument("--out", type=argparse.FileType("w"), default=sys.stdout)
    args = ap.parse_args()
    if not args.pattern and not args.seconds:
        ap.error("give --while PATTERN or --for SECONDS: a sampler with no end "
                 "is a file that grows until someone notices")

    deadline = time.time() + args.seconds if args.seconds else None
    while True:
        if args.pattern and subprocess.run(
                ["pgrep", "-f", args.pattern], capture_output=True).returncode != 0:
            break
        if deadline and time.time() >= deadline:
            break
        print(sample(), file=args.out, flush=True)
        time.sleep(args.every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
