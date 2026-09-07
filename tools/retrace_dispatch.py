#!/usr/bin/env python3
"""Dispatch retraces on one card (or two) as snapshots become available.

Loops over an ordered candidate list: the first container not yet complete
whose snapshot is present runs through `retrace_zoo.py`; when none is ready
and the download supervisor is still working, waits and looks again; ends when
every candidate is complete or no snapshot can still arrive.

    python tools/retrace_dispatch.py --gpu 2,3 --candidates a,b,c --src ... [--wait-minutes 20]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = "/home/mlops/ml/venv/bin/python"
OUT = REPO / "validation_outputs" / "retrace_2026_09_07"
PROGRESS = OUT / "snap" / "progress.log"
ALIAS = {"Sana-1600M-MultiLing": "Sana_1600M_1024px_MultiLing"}


SNAP_LOGS = Path(__file__).resolve().parents[1] / "validation_outputs" / "retrace_2026_09_07" / "snap"


def has_snapshot(name: str) -> bool:
    """A snapshot is present only when it is COMPLETE: a directory with files in it is what a
    stopped download leaves behind (chatterbox 2.2 GB, openaudio 1.6 GB, granite 632 KB on
    2026-09-07). Complete = no partial file under it, and — for a repository the re-download
    tool ever touched — the toolchain's completion marker."""
    for root in (Path("/home/mlops/hf_snapshots"), Path.home() / ".cache" / "neurobrix" / "hf_snapshots"):
        for cand in (ALIAS.get(name, name), name):
            p = root / cand
            if not (p.is_dir() and any(p.iterdir())):
                continue
            if any(p.rglob("*.incomplete")):
                return False
            touched = (SNAP_LOGS / f"{cand}.log").exists()
            if touched and not (p / ".snapshot_complete").exists():
                return False
            return True
    return False


def complete(name: str) -> bool:
    sp = OUT / name / "state.json"
    if not sp.exists():
        return False
    steps = json.loads(sp.read_text()).get("steps", {})
    return bool((steps.get("upload") or {}).get("ok"))


def attempted_and_stopped(name: str) -> bool:
    """A model whose pipeline stopped (a gate to explain, a failed step): not retried by the dispatcher — a chantier."""
    sp = OUT / name / "state.json"
    if not sp.exists():
        return False
    steps = json.loads(sp.read_text()).get("steps", {})
    return any(v.get("ok") is False for v in steps.values())


def downloads_over() -> bool:
    if not PROGRESS.exists():
        return True
    text = PROGRESS.read_text()
    return "supervisor: every repository present" in text or "supervisor: the same failure four times" in text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True); ap.add_argument("--candidates", required=True); ap.add_argument("--src", required=True)
    ap.add_argument("--extra", default="--num-frames 9 --steps 4"); ap.add_argument("--wait-minutes", type=int, default=20)
    args = ap.parse_args()
    cands = [c for c in args.candidates.split(",") if c]
    def log(m): print(f"[dispatch {time.strftime('%H:%M:%S')} gpu {args.gpu}] {m}", flush=True)
    while True:
        pending = [c for c in cands if not complete(c) and not attempted_and_stopped(c)]
        if not pending:
            log("every candidate complete or stopped as a chantier"); return 0
        ready = [c for c in pending if has_snapshot(c)]
        if ready:
            m = ready[0]
            log(f"{m}: retrace ({len(pending) - 1} still pending)")
            subprocess.run([PY, str(REPO / "tools" / "retrace_zoo.py"), "--models", m, "--gpu", args.gpu, "--src", args.src,
                            "--out", str(OUT), "--extra", args.extra])
            continue
        if downloads_over():
            log(f"no snapshot can still arrive; pending without one: {', '.join(pending)}"); return 1
        log(f"waiting {args.wait_minutes} min for a snapshot ({', '.join(pending)})")
        time.sleep(args.wait_minutes * 60)


if __name__ == "__main__":
    sys.exit(main())
