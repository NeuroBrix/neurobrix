#!/usr/bin/env python3
"""Re-download purged model snapshots onto the export, one at a time, with the
repository's `.env` loaded the way the build toolchain loads its own.

A gated repository needs HF_TOKEN: its absence is an explicit refusal naming
the variable and the file, before any request is made — never a 401 read off
a log. Values are never printed; the progress log carries names only.

The export is probed (an `ls` under a time limit) before every repository AND
every few seconds while a download runs: the moment it exceeds the limit the
download is stopped by name and the chain ends — the server stalls under
write pressure beside the batteries (2026-09-02 and twice on 09-07) and
recovers by draining. A supervisor resumes the remaining repositories after a
pause; partial files resume where they stopped.

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


def _complete(dest: Path, marker: Path) -> bool:
    """A repository is present only when this tool saw its download end (the
    marker) and no partial file remains under it — a directory with files in
    it is what a stopped download leaves behind, not a snapshot."""
    if not marker.exists() or not dest.is_dir():
        return False
    return not any(dest.rglob("*.incomplete"))


def _export_answers(dest: str, limit: float) -> float | None:
    """Seconds an `ls` of the export took, or None when it exceeded `limit`.
    A directory read waits in a killable RPC wait, so the timeout's kill lands."""
    t = time.time()
    try:
        subprocess.run(["ls", dest], capture_output=True, timeout=limit)
    except subprocess.TimeoutExpired:
        return None
    return time.time() - t


def _previous_download(repo: str) -> int | None:
    """The pid of a `snap --name <repo>` still alive (a stopped chain's child can
    stay in an uninterruptible write until the export answers)."""
    me = os.getpid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"snap" in argv and b"--name" in argv and repo.encode() in argv and any(b"forge.py" in a for a in argv):
            return int(pid)
    return None


def _download_under_probe(repo: str, args, fh, note) -> int:
    """Run one download and probe the export while it runs: the moment an `ls`
    of the export exceeds the limit, the download is stopped by name — the
    export is under pressure and the writer is what it drains. Returns the
    download's return code, or -1 when the probe stopped it."""
    import signal
    proc = subprocess.Popen([PY, str(FORGE), "snap", "--name", repo, "--path", args.dest, "--max-workers", str(args.max_workers),
                             "--max-write-mbps", str(args.max_write_mbps)],
                            cwd=str(REPO / "forge"), stdout=fh, stderr=subprocess.STDOUT, env={**os.environ})
    while True:
        try:
            return proc.wait(timeout=args.probe_interval)
        except subprocess.TimeoutExpired:
            pass
        took = _export_answers(args.dest, args.probe_seconds)
        if took is not None and took <= args.probe_seconds / 2:
            continue
        reason = (f"took more than {args.probe_seconds:g} s to list" if took is None
                  else f"answered in {took:.1f} s (pressure)")
        note(f"{repo}: STOPPED by the export probe — {args.dest} {reason} during the download; "
             f"the partial files resume after the pause")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                note(f"{repo}: the download process (pid {proc.pid}) is blocked in an uninterruptible write; it ends when the export answers")
        return -1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repos", required=True, help="comma-separated org/name")
    ap.add_argument("--dest", default="/home/mlops/hf_snapshots", help="the export, never the root fs")
    ap.add_argument("--log", default=str(REPO / "validation_outputs" / "retrace_2026_09_07" / "snap"))
    ap.add_argument("--max-workers", type=int, default=1, help="download streams; 1 on the shared export")
    ap.add_argument("--probe-seconds", type=float, default=5.0,
                    help="an `ls` of the export slower than this, before a repository or during its download, = the export is under pressure: stop, by name")
    ap.add_argument("--probe-interval", type=float, default=10.0, help="seconds between two probes of the export while a download runs")
    ap.add_argument("--max-write-mbps", type=float, default=40.0,
                    help="cap on the rate the downloader writes to the export (MB/s); an unthrottled single stream stalled it beside its readers")
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
        if _complete(dest, logdir / f"{short}.done"):
            note(f"{repo}: present"); continue
        # The export's responsiveness before every repository: a burst of writes beside the
        # batteries stalled every export of the server on 2026-09-07 (the 09-02 class).
        took = _export_answers(args.dest, args.probe_seconds)
        if took is None:
            note(f"{repo}: NOT STARTED — the export {args.dest} took more than {args.probe_seconds:g} s to list; stopping the chain")
            failed += 1
            break
        if took > args.probe_seconds / 2:
            note(f"{repo}: NOT STARTED — the export answered in {took:.1f} s (pressure); stopping the chain")
            failed += 1
            break
        previous = _previous_download(repo)
        if previous:
            note(f"{repo}: NOT STARTED — a previous download of this repository (pid {previous}) is still ending on the export; stopping the chain")
            failed += 1
            break
        note(f"{repo}: downloading to {args.dest} ({args.max_workers} stream(s), writes capped at {args.max_write_mbps:g} MB/s)")
        with open(logdir / f"{short}.log", "a") as fh:
            rc = _download_under_probe(repo, args, fh, note)
        if rc == 0:
            size = subprocess.run(["du", "-sh", str(dest)], capture_output=True, text=True).stdout.split()[0] if dest.exists() else "?"
            note(f"{repo}: DONE {size}")
            (logdir / f"{short}.done").write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {repo} {size}\n")
        elif rc == -1:
            failed += 1
            break                                  # stopped by the probe: the chain resumes after the supervisor's pause
        else:
            failed += 1
            note(f"{repo}: FAILED (rc {rc}, see {short}.log)")
    note(f"SNAP DONE ({failed} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
