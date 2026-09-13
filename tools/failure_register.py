#!/usr/bin/env python3
"""The failing tests, versioned, so "no new failures" is a diff and not a claim.

A battery report that says "150 failures, none new" is unverifiable — by the
reader and by its author. This keeps the set of failing node ids in a tracked
file, so the comparison is produced by the machine:

    # after a run, from its log
    tools/failure_register.py --check  run.log [run2.log ...]
    tools/failure_register.py --update run.log [run2.log ...]

`--check` exits 1 if any failure is NEW (not in the register) and prints both
directions: what appeared, and what is fixed and should be removed. `--update`
rewrites the register from the logs given.

The register holds node ids and nothing else. No count, no category, no date:
a count invites arithmetic instead of a diff, and a category written by hand
rots the day the reason changes. What a failure MEANS belongs in the finding
that instructs it; what this file answers is only "is this one already known".

Several logs may be given because a battery on a shared machine is run in
slices — the register is the union of what those slices report, and a slice
that was not run simply contributes nothing rather than looking fixed. That is
why `--check` names missing slices as UNRUN rather than as repairs: see
`--expect`.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REGISTER = Path(__file__).resolve().parents[1] / "tests" / "failure_register.txt"

_FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+?)(?: - .*)?$", re.MULTILINE)

_HEADER = """# Failing tests, one node id per line, sorted.
#
# Machine-written by tools/failure_register.py. Do not hand-edit: the point of
# this file is that the comparison is produced rather than asserted.
#
# A line here is a failure that is KNOWN, not one that is accepted. Why each
# one fails belongs in the finding that instructs it.
"""


def parse(paths: list[Path]) -> set[str]:
    found: set[str] = set()
    for p in paths:
        text = p.read_text(errors="replace")
        for node in _FAILED.findall(text):
            # pytest prints paths relative to its own rootdir, which differs
            # between a run of `tests/` and a run of `tests/regression/`.
            # Normalise to the repo-relative form so slices are comparable.
            if not node.startswith("tests/"):
                node = "tests/" + node.split("tests/", 1)[-1] if "tests/" in node else node
            found.add(node.strip())
    return found


def load() -> set[str]:
    if not REGISTER.exists():
        return set()
    return {line.strip() for line in REGISTER.read_text().splitlines()
            if line.strip() and not line.startswith("#")}


def save(nodes: set[str]) -> None:
    REGISTER.parent.mkdir(parents=True, exist_ok=True)
    REGISTER.write_text(_HEADER + "\n".join(sorted(nodes)) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--update", action="store_true",
                    help="rewrite the register from these logs")
    ap.add_argument("--expect", action="append", default=[],
                    help="a path prefix these logs are expected to cover; a "
                         "known failure under a prefix that was NOT covered is "
                         "reported as unrun, never as fixed")
    args = ap.parse_args()

    seen = parse(args.logs)
    if args.update:
        save(seen)
        print(f"register written: {len(seen)} failing node ids -> {REGISTER}")
        return 0

    known = load()
    if not known:
        print(f"no register at {REGISTER}; seed one with --update")
        return 1

    new = sorted(seen - known)
    gone = sorted(known - seen)

    covered = args.expect or None
    unrun = [n for n in gone
             if covered and not any(n.startswith(pre) for pre in covered)]
    fixed = [n for n in gone if n not in unrun]

    print(f"failing now: {len(seen)}    known: {len(known)}")
    if new:
        print(f"\nNEW ({len(new)}) — not in the register:")
        for n in new:
            print(f"  + {n}")
    if fixed:
        print(f"\nFIXED ({len(fixed)}) — in the register, no longer failing:")
        for n in fixed:
            print(f"  - {n}")
        print("  run --update once these are believed repaired")
    if unrun:
        print(f"\nUNRUN ({len(unrun)}) — known, but no log covered them:")
        for n in unrun:
            print(f"  ? {n}")
        print("  not counted as repairs")
    if not (new or fixed or unrun):
        print("\nidentical to the register")
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
