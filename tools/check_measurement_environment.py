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


def main() -> int:
    problems: list[str] = []
    problems += check_importable("triton_msl")
    for target in _editable_targets("triton_msl"):
        # the clone root is the parent of the package directory
        problems += check_worktree_intact(target.parent)
        break

    if problems:
        print("REFUSING TO MEASURE — the environment has degraded:")
        for p in problems:
            print(f"  * {p}")
        return 1
    print("environment intact: external clone complete, triton_msl imports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
