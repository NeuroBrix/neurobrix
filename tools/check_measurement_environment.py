#!/usr/bin/env python3
"""Refuse to measure on an environment that has silently degraded.

Written 2026-09-10 after six test failures and one collection error were
traced not to the code under test but to an external clone that the system's
periodic /private/tmp cleaner had half-deleted: 245 tracked files gone,
`triton_msl/__init__.py` among them, the directory still present so nothing
looked wrong. The measurement said "nine pre-existing failures". There were
three.

Run this before any suite or campaign whose result will be written down.
Exit 0 = the environment is what the measurement assumes. Non-zero = say so
and stop; a degraded environment does not produce a weaker number, it
produces a wrong one.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _editable_targets(module: str) -> list[Path]:
    """Directories an editable install points at, read from its finder."""
    spec = importlib.util.find_spec(f"__editable___{module}_0_2_0_finder")
    if spec is None or spec.origin is None:
        return []
    ns: dict = {}
    exec(compile(Path(spec.origin).read_text(), spec.origin, "exec"), ns)
    return [Path(p) for p in (ns.get("MAPPING") or {}).values()]


def check_importable(module: str) -> list[str]:
    try:
        __import__(module)
        return []
    except Exception as exc:
        targets = _editable_targets(module)
        detail = "".join(
            f"\n    {t} : {'present' if t.exists() else 'GONE'}"
            f"{'' if (t / '__init__.py').exists() else '  (no __init__.py)'}"
            for t in targets)
        return [f"{module} does not import: {exc}{detail}"]


def check_worktree_intact(path: Path) -> list[str]:
    """A checkout with files deleted underneath it is not a checkout."""
    if not (path / ".git").exists():
        return [f"{path} is not a git checkout"]
    out = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                         capture_output=True, text=True).stdout
    deleted = [l[3:] for l in out.splitlines() if l.startswith(" D")]
    if not deleted:
        return []
    return [f"{path}: {len(deleted)} tracked files deleted from the working "
            f"tree (e.g. {', '.join(deleted[:3])}). "
            f"Restore with: git -C {path} checkout -- ."]


def check_object_store(path: Path) -> list[str]:
    """A checkout whose history is unreadable is eroding, not merely dirty.

    The working-tree check above missed this: `git checkout -- .` restored
    every file, and `git log -S` still answered `fatal: unable to read tree`,
    because the cleaner had eaten loose objects too. A measurement can be
    sound on such a repo (the tree at HEAD is complete) while every question
    about *when* something changed is unanswerable — and the next thing the
    cleaner eats may be the tree itself.
    """
    if not (path / ".git").exists():
        return []
    out = subprocess.run(["git", "-C", str(path), "fsck", "--no-progress",
                          "--connectivity-only"],
                         capture_output=True, text=True, timeout=300)
    bad = [l for l in (out.stdout + out.stderr).splitlines()
           if l.startswith(("missing ", "broken link")) or "unable to read" in l]
    if not bad:
        return []
    return [f"{path}: object store is incomplete ({len(bad)} problems, "
            f"e.g. {bad[0]}). History is partly unreadable; the tree at HEAD "
            f"may still be sound, but this repo is being eroded."]


def check_branch_is_recoverable(path: Path) -> list[str]:
    """Work that exists only here is one cleaner pass from gone."""
    if not (path / ".git").exists():
        return []
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", str(path), "rev-parse",
                             "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    remotes = subprocess.run(["git", "-C", str(path), "remote"],
                             capture_output=True, text=True).stdout.split()
    for r in remotes:
        ls = subprocess.run(["git", "-C", str(path), "ls-remote", "--heads",
                             r], capture_output=True, text=True, timeout=120)
        if head in ls.stdout:
            return []
    return [f"{path}: HEAD {head[:12]} ({branch}) is on no remote. If this "
            f"checkout lives under a directory the system cleans, the work "
            f"exists in exactly one place."]


def main() -> int:
    # Two classes, and the line between them is the same one this repo draws
    # everywhere else: REFUSE where the measurement would be wrong, SAY IT
    # LOUDLY where it would not.
    #
    #   * a deleted working-tree file changes what is compiled -> refuse;
    #   * an incomplete object store leaves the tree at HEAD sound, so the
    #     measurement stands — but the repo is eroding and every question
    #     about when something changed is already unanswerable -> warn.
    problems: list[str] = []
    warnings: list[str] = []
    problems += check_importable("triton_msl")
    for target in _editable_targets("triton_msl"):
        # the clone root is the parent of the package directory
        clone = target.parent
        problems += check_worktree_intact(clone)
        warnings += check_object_store(clone)
        warnings += check_branch_is_recoverable(clone)
        break

    for w in warnings:
        print(f"  ! {w}")
    if problems:
        print("REFUSING TO MEASURE — the environment has degraded:")
        for p in problems:
            print(f"  * {p}")
        return 1
    print("environment sound for measuring: working tree complete, "
          "triton_msl imports"
          + (f" ({len(warnings)} warning(s) above)" if warnings else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
