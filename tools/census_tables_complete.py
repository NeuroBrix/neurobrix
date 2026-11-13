#!/usr/bin/env python3
"""census_tables_complete.py: every hardware class's census carries every table another class carries.

A class whose census lacks a table the others have serves that part of the engine by sweeping at run time: a served
continuation on a class without its prefill-chunk record was planned without one and swept on the user's machine. This
door is run before a release, on the tree that ships.

CONTRACT
  * the classes are the census tables `<vendor>/<profile>/<N>g.jsonl` under the census root (the engine's
    `kernels/census_table.ROOT`, or --root); a class's companions are the files `kernels/census_table` names beside
    its table (`inputs_path`, `chunks_path`), never a list kept here;
  * GREEN (exit 0) when every class has every companion that any class has, and no companion stands without its table;
  * RED (exit 1) naming each class and the file it lacks, and each companion without its table;
  * refused by name (exit 2) when the root holds no class table at all (an empty or mistyped root is never green);
  * the verdict is printed, and written to --out when given (every exit leaves a readable file).

    census_tables_complete.py [--root DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neurobrix.kernels import census_table  # noqa: E402

TABLE = re.compile(r"^\d+g\.jsonl$")
COMPANIONS = {"inputs": census_table.inputs_path, "chunks": census_table.chunks_path}


def verdict(root: Path):
    """(exit code, lines) for the census under `root`."""
    tables = sorted(p for p in root.glob("*/*/*.jsonl") if TABLE.match(p.name))
    if not tables:
        return 2, [f"REFUSED: no census class table (<vendor>/<profile>/<N>g.jsonl) under {root}"]
    have = {t: {kind for kind, path_of in COMPANIONS.items() if path_of(t).exists()} for t in tables}
    wanted = set().union(*have.values())
    lines = []
    for t in tables:
        for kind in sorted(wanted - have[t]):
            others = sorted(str(o.relative_to(root)) for o in tables if kind in have[o])
            lines.append(f"RED {t.relative_to(root)}: lacks {COMPANIONS[kind](t).name} ({kind}), which "
                         f"{', '.join(others)} carr{'ies' if len(others) == 1 else 'y'}")
    expected = {COMPANIONS[kind](t) for t in tables for kind in COMPANIONS}
    for p in sorted(root.glob("*/*/*.jsonl")):
        if not TABLE.match(p.name) and p not in expected and re.match(r"^\d+g\.", p.name):
            lines.append(f"RED {p.relative_to(root)}: a companion without its class table")
    if lines:
        return 1, lines
    return 0, [f"GREEN {len(tables)} classes, each with {', '.join(sorted(wanted)) or 'no companion'}: "
               + ", ".join(str(t.relative_to(root)) for t in tables)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="every census class carries every table another class carries")
    ap.add_argument("--root", type=Path, default=census_table.ROOT)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    rc, lines = verdict(a.root)
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if a.out:
        a.out.write_text(text)
    return rc


if __name__ == "__main__":
    sys.exit(main())
