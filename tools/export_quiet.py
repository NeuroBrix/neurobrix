#!/usr/bin/env python3
"""Refuse to start heavy work against an export that is already busy.

WHAT THIS COST, 2026-09-12
---------------------------
Four jobs were started against one NFS export at the same time: a 21.5 GB
container upload reading from it, a 118 GB snapshot download writing to it, a GPU
run loading weights from it, and traces reading from it. The export stopped
serving bulk I/O — a 100 MB read timed out at 60 s while `ls` still answered in
milliseconds — and it stayed down for eleven minutes after every one of those
processes had been killed.

Nothing was lost, because the upload protocol had a read timeout, aborted its own
multipart and rolled back its partial key. The cost was an hour of upload and
104 GB of download that must be done again.

The project's memory already carried the rule — *"NFS build staging OFF the
export: a 45 GB build staging stalled the server twice"* — and a rule in a memory
is read by whoever is already careful. This is that rule as a refusal.

WHAT IT MEASURES, AND WHY NOT `df` OR THE LOAD AVERAGE
------------------------------------------------------
Neither says anything about the thing that breaks. `df` answers from cached
metadata while bulk I/O is dead; the load average is a decaying mean that stays
high for ten minutes after the cause is gone and is therefore a lagging indicator
of a condition you need to know NOW.

So it reads bytes. A real file on the export, a real timed read, and a refusal
under a floor — which is the only question a job about to move tens of gigabytes
is actually asking.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

#: Below this, a job that moves tens of gigabytes will take hours and will hold
#: the export while it does. Measured on a healthy day this export serves
#: 65-76 MB/s to a downloader; it was doing 20.2 MB/s while draining after the
#: 2026-09-12 stall, and 0 during it.
FLOOR_MB_S = 40.0

#: How much to read. Large enough that the client's page cache cannot answer it,
#: small enough that the probe is not itself the load it is looking for.
PROBE_MB = 200


class ExportBusy(RuntimeError):
    """The export cannot serve bulk I/O right now."""


#: How many entries the search may look at. A health check that walks a tree is
#: the load it is looking for: the first version used `rglob("*")` and took
#: minutes on the very export it was meant to declare busy.
SCAN_LIMIT = 400


def _probe_file(root: Path) -> Path | None:
    """The FIRST file of at least PROBE_MB, found under a bounded scan.

    Not the largest: finding the largest means visiting all of them, and the
    question is whether the export moves bytes, which any large file answers.
    """
    seen = 0
    stack = [root]
    while stack and seen < SCAN_LIMIT:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            seen += 1
            if seen >= SCAN_LIMIT:
                break
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    if entry.stat().st_size >= PROBE_MB * 2**20:
                        return Path(entry.path)
            except OSError:
                continue
    return None


def throughput_mb_s(path: Path, megabytes: int = PROBE_MB,
                    timeout: int = 90) -> float:
    """Bulk read rate, or 0.0 when the read does not finish in `timeout`.

    Reads with O_DIRECT where the platform allows it so the client's page cache
    cannot answer a question about the server.
    """
    t0 = time.time()
    try:
        subprocess.run(["dd", f"if={path}", "of=/dev/null", "bs=1M",
                        f"count={megabytes}", "iflag=direct"],
                       capture_output=True, timeout=timeout, check=True)
    except subprocess.TimeoutExpired:
        return 0.0
    except subprocess.CalledProcessError:
        # Some exports refuse O_DIRECT; fall back rather than call it a stall.
        try:
            subprocess.run(["dd", f"if={path}", "of=/dev/null", "bs=1M",
                            f"count={megabytes}"],
                           capture_output=True, timeout=timeout, check=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return 0.0
    elapsed = time.time() - t0
    return (megabytes / elapsed) if elapsed > 0 else 0.0


def refuse_busy_export(root, floor: float = FLOOR_MB_S,
                       allow: bool = False) -> float:
    root = Path(root)
    probe = _probe_file(root)
    if probe is None:
        # Nothing large enough to time. Say so rather than pass: a check that
        # measured nothing must not read like one that measured health.
        print(f"   [export] no file of {PROBE_MB} MB or more under {root}; "
              f"bulk throughput NOT measured")
        return -1.0
    rate = throughput_mb_s(probe)
    print(f"   [export] {rate:.1f} MB/s reading {probe.name} "
          f"(floor {floor:.0f} MB/s)")
    if rate < floor and not allow:
        raise ExportBusy(
            f"EXPORT BUSY: {root} serves {rate:.1f} MB/s, under the {floor:.0f} "
            f"MB/s floor.\n"
            f"  A job that moves tens of gigabytes will take hours here AND will "
            f"hold the export while it does.\n"
            f"  On 2026-09-12 four such jobs at once stopped it serving bulk I/O "
            f"entirely for eleven minutes.\n"
            f"  Wait, or run them one at a time. Deliberate opening: "
            f"--allow-busy-export.")
    if rate < floor:
        print(f"   [export] --allow-busy-export: proceeding at {rate:.1f} MB/s")
    return rate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--floor", type=float, default=FLOOR_MB_S)
    ap.add_argument("--allow-busy-export", action="store_true")
    args = ap.parse_args()
    try:
        refuse_busy_export(args.root, args.floor, args.allow_busy_export)
    except ExportBusy as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("   [export] clear")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
