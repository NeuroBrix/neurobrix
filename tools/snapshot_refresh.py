#!/usr/bin/env python3
"""Re-download purged model snapshots onto the export, one at a time, with the
repository's `.env` loaded the way the build toolchain loads its own.

A gated repository needs HF_TOKEN: its absence is an explicit refusal naming
the variable and the file, before any request is made — never a 401 read off
a log. Values are never printed; the progress log carries names only.

    python tools/snapshot_refresh.py --repos org/name,org/name [--dest /home/mlops/hf_snapshots] [--log DIR]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import repo_env  # noqa: E402

PY = "/home/mlops/ml/venv/bin/python"
FORGE = REPO / "forge" / "forge.py"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repos", required=True, help="comma-separated org/name")
    ap.add_argument("--dest", default="/home/mlops/hf_snapshots", help="the export, never the root fs")
    ap.add_argument("--log", default=str(REPO / "validation_outputs" / "retrace_2026_09_07" / "snap"))
    ap.add_argument("--max-workers", type=int, default=1, help="download streams; 1 on the shared export")
    ap.add_argument("--probe-seconds", type=float, default=5.0,
                    help="an `ls` of the export slower than this before a repository = the export is under pressure: stop, by name")
    args = ap.parse_args()
    repo_env.require("HF_TOKEN")                 # refuses by name before the first request
    logdir = Path(args.log); logdir.mkdir(parents=True, exist_ok=True)
    progress = logdir / "progress.log"
    def note(msg):
        with open(progress, "a") as f:
            f.write(f"{time.strftime('%H:%M')} {msg}\n")
        print(msg, flush=True)
    failed = 0
    for repo in [r for r in args.repos.split(",") if r]:
        short = repo.rsplit("/", 1)[-1]
        dest = Path(args.dest) / short
        if dest.is_dir() and any(dest.iterdir()) and not (dest / ".incomplete").exists():
            note(f"{repo}: present"); continue
        # The export's responsiveness before every repository: a burst of writes beside the
        # batteries stalled every export of the server on 2026-09-07 (the 09-02 class).
        t_probe = time.time()
        try:
            subprocess.run(["ls", args.dest], capture_output=True, timeout=args.probe_seconds)
        except subprocess.TimeoutExpired:
            note(f"{repo}: NOT STARTED — the export {args.dest} took more than {args.probe_seconds:g} s to list; stopping the chain")
            failed += 1
            break
        if time.time() - t_probe > args.probe_seconds / 2:
            note(f"{repo}: NOT STARTED — the export answered in {time.time() - t_probe:.1f} s (pressure); stopping the chain")
            failed += 1
            break
        note(f"{repo}: downloading to {args.dest} ({args.max_workers} stream(s))")
        with open(logdir / f"{short}.log", "a") as fh:
            rc = subprocess.run([PY, str(FORGE), "snap", "--name", repo, "--path", args.dest, "--max-workers", str(args.max_workers)],
                                cwd=str(REPO / "forge"), stdout=fh, stderr=subprocess.STDOUT, env={**os.environ}).returncode
        if rc == 0:
            size = subprocess.run(["du", "-sh", str(dest)], capture_output=True, text=True).stdout.split()[0] if dest.exists() else "?"
            note(f"{repo}: DONE {size}")
        else:
            failed += 1
            note(f"{repo}: FAILED (rc {rc}, see {short}.log)")
    note(f"SNAP DONE ({failed} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
