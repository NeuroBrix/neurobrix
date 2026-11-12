#!/usr/bin/env python3
"""package_files_check.py <wheel> <sdist> [--tree DIR]: a release gate (owner's chain, 2026-11-12, after 0.6.10 shipped
without its census tables): every file the repository tracks under src/neurobrix is in the built wheel AND the built
sdist; exits 1 naming each one missing, 0 when none is. Bytecode and caches are never tracked, so nothing is excused.
An empty or unreadable input is refused by name (rc 2): a gate that read nothing proves nothing."""
import argparse
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def tracked(tree: Path):
    out = subprocess.run(["git", "ls-files", "src/neurobrix"], cwd=tree, capture_output=True, text=True, check=True)
    files = [p for p in out.stdout.splitlines() if p]
    if not files:
        raise SystemExit(f"package_files_check: no tracked file under {tree}/src/neurobrix — wrong --tree")
    return files


def wheel_members(path: Path):
    with zipfile.ZipFile(path) as z:
        return {"src/" + n for n in z.namelist() if n.startswith("neurobrix/")}


def sdist_members(path: Path):
    with tarfile.open(path) as t:
        return {n.split("/", 1)[1] for n in t.getnames() if "/" in n}


def missing(files, members):
    return sorted(f for f in files if f not in members)


def main(argv=None) -> int:
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("wheel", type=Path)
    a.add_argument("sdist", type=Path)
    a.add_argument("--tree", type=Path, default=Path(__file__).resolve().parents[1])
    args = a.parse_args(argv)
    for p in (args.wheel, args.sdist):
        if not p.is_file() or p.stat().st_size == 0:
            print(f"package_files_check: {p} is missing or empty — nothing to check", file=sys.stderr)
            return 2
    files = tracked(args.tree)
    gone = {"wheel": missing(files, wheel_members(args.wheel)), "sdist": missing(files, sdist_members(args.sdist))}
    for kind, names in gone.items():
        for n in names:
            print(f"MISSING from the {kind}: {n}")
    print(f"package_files_check: {len(files)} tracked file(s); missing from the wheel {len(gone['wheel'])}, "
          f"from the sdist {len(gone['sdist'])}")
    return 1 if any(gone.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
