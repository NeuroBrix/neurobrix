#!/usr/bin/env python3
"""The judgments record: quality judged ONCE per model and branch, committed (the validation regime, rule 2).

A gate judges rc and misses; the quality of what a cell produced is judged from OUTSIDE (R29: an instrument
looked at the artefact) once per model and branch, and that judgment is a record, not a message. The harness
writes each judgment where the cell's row sits (`regression_matrix.py judge` → `<matrix dir>/judgments.jsonl`);
this tool gathers them into THE committed record, `docs/reference/judgments.jsonl`, one line per (model, mode):
the latest judgment, with its class, its instrument and what it saw, its date and the matrix directory it came
from. The gate's tally reads this record and names every wanted (model, mode) it does not hold.

    judgments.py collect --record docs/reference/judgments.jsonl <matrix dir>...
        gathers every judgments.jsonl under the directories given (a directory named matrix_<cls> gives its
        class), merges them with the record (the latest date per (model, mode) wins; a re-judgment supersedes),
        and rewrites the record sorted. An input without a single judgment is refused by name; a second run
        over the same inputs changes nothing.
    judgments.py check --record docs/reference/judgments.jsonl --gate-list <gate_<cls>.list> [--modes native,triton]
        prints the wanted (model, mode) pairs the record holds and names the ones it does not; exit 1 when any
        is missing — the regime's "judged once per model and branch", read, never assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

FIELDS = ("model", "mode", "class", "verdict", "judged", "date", "source")


class Refused(RuntimeError):
    pass


def read_jsonl(path: Path) -> List[dict]:
    out = []
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def class_of(matrix_dir: Path) -> str:
    name = matrix_dir.name
    return name.split("matrix_", 1)[1] if name.startswith("matrix_") else ""


def gather(dirs: Iterable[Path]) -> List[dict]:
    """Every judgment under `dirs` (each dir's judgments.jsonl, or the judgments.jsonl of its matrix_* children),
    normalised to FIELDS. Refused by name when a directory holds none."""
    out: List[dict] = []
    for d in dirs:
        d = Path(d)
        files = [d / "judgments.jsonl"] if (d / "judgments.jsonl").is_file() else sorted(d.glob("matrix_*/judgments.jsonl"))
        found = 0
        for f in files:
            for j in read_jsonl(f):
                if not all(k in j for k in ("model", "mode", "verdict", "judged", "date")):
                    continue
                out.append({"model": j["model"], "mode": j["mode"], "class": class_of(f.parent),
                            "verdict": j["verdict"], "judged": j["judged"], "date": j["date"], "source": str(f.parent)})
                found += 1
        if not found:
            raise Refused(f"{d}: no judgment under it (no judgments.jsonl with model, mode, verdict, judged, date) — "
                          f"an empty input is refused by name, never recorded as 'nothing to judge'")
    return out


def merge(record: List[dict], new: Iterable[dict]) -> List[dict]:
    """One line per (model, mode): the latest date wins (a re-judgment supersedes the earlier one)."""
    latest: Dict[Tuple[str, str], dict] = {}
    for j in list(record) + list(new):
        k = (j["model"], j["mode"])
        if k not in latest or j.get("date", "") >= latest[k].get("date", ""):
            latest[k] = {f: j.get(f, "") for f in FIELDS}
    return [latest[k] for k in sorted(latest)]


def write_record(path: Path, rows: List[dict]) -> bool:
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    if path.is_file() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return True


def wanted(gate_list: Path, modes: List[str]) -> List[Tuple[str, str]]:
    out = []
    for line in gate_list.read_text().splitlines():
        p = line.split()
        if not p or p[0].startswith("#"):
            continue
        listed = p[1].split(",") if len(p) > 1 else modes
        out.extend((p[0], m) for m in listed if m in modes)
    return out


def check(record: List[dict], cells: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str, dict]], List[Tuple[str, str]]]:
    held = {(j["model"], j["mode"]): j for j in record}
    judged = [(m, x, held[(m, x)]) for m, x in cells if (m, x) in held]
    missing = [(m, x) for m, x in cells if (m, x) not in held]
    return judged, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--record", required=True)
    c.add_argument("dirs", nargs="+")
    k = sub.add_parser("check")
    k.add_argument("--record", required=True)
    k.add_argument("--gate-list", required=True)
    k.add_argument("--modes", default="native,triton")
    a = ap.parse_args(argv)
    rec = Path(a.record)
    if a.cmd == "collect":
        try:
            new = gather(Path(d) for d in a.dirs)
        except Refused as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        before = read_jsonl(rec)
        rows = merge(before, new)
        changed = write_record(rec, rows)
        print(f"{rec}: {len(rows)} (model, mode) judgments, {len(new)} gathered from {len(a.dirs)} input(s), "
              + ("rewritten" if changed else "unchanged"))
        return 0
    record = read_jsonl(rec)
    if not rec.is_file():
        print(f"REFUSED: {rec} does not exist — the record is committed, never assumed empty", file=sys.stderr)
        return 2
    cells = wanted(Path(a.gate_list), a.modes.split(","))
    judged, missing = check(record, cells)
    print(f"{len(judged)} of {len(cells)} wanted (model, mode) judged; {len(missing)} not judged")
    for m, x, j in judged:
        print(f"  {m}:{x} {j['verdict']} ({j['date']}) {j['judged'][:80]}")
    for m, x in missing:
        print(f"  NOT JUDGED {m}:{x}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
