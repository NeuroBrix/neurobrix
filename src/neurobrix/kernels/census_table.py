"""THE census table — one per vendor profile and memory class, committed, the single source of truth.

The owner's method (2026-09-28 20:06): the census extracts from every container every kernel key its
runs will form, the certifier certifies exactly that list, and a confirmation run is served from the
certified directory alone. This module is the one reader and writer of the list:

    config/census/<vendor>/<profile>/<memory class>g.jsonl

one JSON object per line, one line per (model, mode, op, kernel, key), sorted, with these columns:

    model      the container's name
    container  the sha the census computed over the container's graph.json files — a retrace changes
               it, and the model's rows are replaced, never added to
    mode       the served mode that formed the key (triton, triton-sequential)
    rungs_mb   the memory budgets the shadow planned under when it formed the key, ascending (a key
               formed only at small rungs is a tiled plan's shape class; its reason to exist); None
               for the profile's own budget
    op         the graph op that formed it (op uid), None for rows consolidated from censuses taken
               before the shadow recorded it
    kernel     the kernel's qualified name
    key        the key tuple as the certifier reads it (`key_repr`) — the bucketed shape class
    dtype      the dtypes the key names, in order
    tool       the census tool's tree revision

The census tool writes it, the certifier reads it and nothing else, and a confirmation run's refusal
(`autotune_certified.census_row`) names the row that should have held a missing key. No torch, no
device: a table is data.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

COLUMNS = ("model", "container", "mode", "rungs_mb", "op", "kernel", "key", "dtype", "tool")

#: The table's root inside the engine package, beside the certified directory. Tests redirect it.
ROOT = Path(__file__).resolve().parents[1] / "config" / "census"


def table_path(vendor: str, profile: str, memory_class: int, root: Optional[Path] = None) -> Path:
    return (root or ROOT) / vendor / profile / f"{int(memory_class)}g.jsonl"


def key_line(kernel_qual: str, key_repr_text: str) -> str:
    """The key as one line — the form the certifier reads and the census records."""
    return f"{kernel_qual}::{key_repr_text}"


def dtypes_of(key_repr_text: str) -> str:
    """The dtype names a key carries, in order ('fp16,fp16,fp32'), read from its repr."""
    import ast
    try:
        tup = ast.literal_eval(key_repr_text)
    except (ValueError, SyntaxError):
        return ""
    return ",".join(x for x in tup if isinstance(x, str))


def read(path: Path) -> List[Dict]:
    """Every row of a table file; a missing file is an empty table, an unreadable line an error."""
    if not path.exists():
        return []
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: not a table row ({exc})") from exc
        missing = [c for c in COLUMNS if c not in row]
        if missing:
            raise ValueError(f"{path}:{n}: a row without {missing}")
        rows.append(row)
    return rows


def _sort_key(row: Dict) -> Tuple:
    return (row["model"], row["mode"], row["kernel"], row["key"], row["op"] or "")


def write(path: Path, rows: Iterable[Dict]) -> int:
    """Write the table, sorted, one row per (model, mode, op, kernel, key) — the rungs of duplicate
    rows merged into one ascending list — atomically, readable by all. Returns the row count."""
    merged: Dict[Tuple, Dict] = {}
    for r in rows:
        row = {c: r.get(c) for c in COLUMNS}
        ident = (row["model"], row["mode"], row["op"], row["kernel"], row["key"])
        prev = merged.get(ident)
        if prev is None:
            merged[ident] = row
        elif prev["rungs_mb"] is not None and row["rungs_mb"] is not None:
            prev["rungs_mb"] = sorted(set(prev["rungs_mb"]) | set(row["rungs_mb"]))
        else:
            prev["rungs_mb"] = None     # formed under the profile's own budget too: no rung is its reason
    out = sorted(merged.values(), key=_sort_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)
    return len(out)


def replace_model(path: Path, model: str, rows: Iterable[Dict]) -> Tuple[int, int]:
    """The model's rows replaced by `rows` (a new census of that container), every other model's kept:
    a retrace replaces, never adds. Returns (rows removed, rows written for the model)."""
    rows = list(rows)
    if any(r.get("model") != model for r in rows):
        raise ValueError(f"replace_model({model!r}) was handed rows of another model")
    old = read(path)
    kept = [r for r in old if r["model"] != model]
    write(path, kept + rows)
    return len(old) - len(kept), len(rows)


_INDEX: Dict[Path, Tuple[float, Dict[str, List[Dict]]]] = {}


def rows_for(kernel_qual: str, key_repr_text: str, vendor: str, profile: str, memory_class: int,
             root: Optional[Path] = None) -> Tuple[Path, List[Dict]]:
    """(the table's path, its rows holding this key) — read once per file version."""
    path = table_path(vendor, profile, memory_class, root)
    mtime = path.stat().st_mtime if path.exists() else -1.0
    cached = _INDEX.get(path)
    if cached is None or cached[0] != mtime:
        index: Dict[str, List[Dict]] = {}
        for row in read(path):
            index.setdefault(key_line(row["kernel"], row["key"]), []).append(row)
        _INDEX[path] = (mtime, index)
    return path, _INDEX[path][1].get(key_line(kernel_qual, key_repr_text), [])
