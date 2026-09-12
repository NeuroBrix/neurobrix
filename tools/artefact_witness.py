#!/usr/bin/env python3
"""A verification names the artefact it read, and refuses if it predates the change.

THE TWO TIMES THIS WAS PAID, BOTH ON 2026-09-12
------------------------------------------------
* A build was reported in flight from a `model.nbx.building` at 26.1 GB while the
  flight recorder already said `failed`. The size was real; it was the size of a
  corpse. Two hours.
* A verification step was placed after a re-trace without checking it could SEE
  the re-trace: the run read `~/.neurobrix/cache` (1448 ops, time=9) while the
  re-trace had written `.cache/graphs` (2271 ops, time=17). The step was not
  wrong about what it read. It read the wrong thing, and said PASS.

Neither is a reasoning error and neither would have been caught by reading the
code harder. Both are the same shape: a measurement whose SUBJECT was never
established. So the rule is mechanical, and this is its executable form —

    a verification states the artefact it read, with its identity and its time,
    and REFUSES if that artefact is older than the change it claims to verify.

WHY A REFUSAL AND NOT A PRINTED WARNING
----------------------------------------
The printed form already existed: every one of those runs logged the path it
opened. A path in a log is read by whoever is already suspicious. A refusal is
read by the person who is not — which, at hour nine, is everyone.

Usage:
    from artefact_witness import witness
    w = witness(graph_path, not_before=("git", "forge", "tracer/worker.py"))
    print(w.line)          # names it, always, pass or fail

    python tools/artefact_witness.py PATH --not-before-commit forge tracer/worker.py
    python tools/artefact_witness.py PATH --not-before-file OTHER
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class StaleArtefact(RuntimeError):
    """The artefact a verification read predates the change it verifies."""


@dataclass
class Witness:
    path: Path
    mtime: float
    size: int
    sha12: str
    reference: str
    reference_time: float

    @property
    def fresh(self) -> bool:
        return self.mtime >= self.reference_time

    @property
    def line(self) -> str:
        stamp = datetime.fromtimestamp(self.mtime, timezone.utc).isoformat(" ", "seconds")
        ref = datetime.fromtimestamp(self.reference_time, timezone.utc).isoformat(" ", "seconds")
        verdict = "AFTER" if self.fresh else "BEFORE  <- STALE"
        return (f"read {self.path} ({self.size} B, sha {self.sha12}) written {stamp}, "
                f"{verdict} {self.reference} at {ref}")


def _commit_time(repo: str, pathspec: str) -> tuple[float, str]:
    """When the change being verified last touched `pathspec` in `repo`."""
    out = subprocess.run(
        ["git", "-C", repo, "log", "-1", "--format=%ct %h", "--", pathspec],
        capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        raise StaleArtefact(
            f"cannot date the change: no commit touches {pathspec} in {repo}. "
            f"A verification with no reference point is not a verification.")
    ts, sha = out.stdout.split()[:2]
    return float(ts), f"{repo}:{pathspec}@{sha}"


def witness(path, not_before, refuse: bool = True) -> Witness:
    """Name the artefact and refuse it if it predates the change.

    `not_before` is ("git", repo, pathspec) — the commit that made the change —
    or ("file", path) — an artefact the read one must be newer than — or a raw
    POSIX timestamp with a label: (ts, label).
    """
    p = Path(path)
    if not p.exists():
        raise StaleArtefact(f"the artefact does not exist: {p}. A verification "
                            f"that reads nothing cannot pass.")
    kind = not_before[0]
    if kind == "git":
        ref_time, ref_name = _commit_time(not_before[1], not_before[2])
    elif kind == "file":
        ref = Path(not_before[1])
        if not ref.exists():
            raise StaleArtefact(f"the reference artefact is absent: {ref}")
        ref_time, ref_name = ref.stat().st_mtime, str(ref)
    else:
        ref_time, ref_name = float(not_before[0]), str(not_before[1])

    st = p.stat()
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    w = Witness(p, st.st_mtime, st.st_size, h.hexdigest()[:12], ref_name, ref_time)
    if refuse and not w.fresh:
        raise StaleArtefact(
            f"STALE ARTEFACT: {w.line}\n"
            f"  This verification would report on a state the change never reached. "
            f"Re-produce the artefact, then verify.")
    return w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--not-before-commit", nargs=2, metavar=("REPO", "PATHSPEC"))
    ap.add_argument("--not-before-file", metavar="PATH")
    args = ap.parse_args()
    if args.not_before_commit:
        nb = ("git", *args.not_before_commit)
    elif args.not_before_file:
        nb = ("file", args.not_before_file)
    else:
        print("REFUSED: give --not-before-commit or --not-before-file. A witness "
              "with no reference point is a timestamp, not a verification.",
              file=sys.stderr)
        return 2
    try:
        print(witness(args.path, nb).line)
        return 0
    except StaleArtefact as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
