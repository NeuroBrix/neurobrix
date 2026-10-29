"""THE census table — one per vendor profile and memory class, committed, the single source of truth.

The owner's method (2026-09-28 20:06): the census extracts from every container every kernel key its
runs will form, the certifier certifies exactly that list, and a confirmation run is served from the
certified directory alone. This module is the one reader and writer of the list:

    config/census/<vendor>/<profile>/<memory class>g.jsonl

one JSON object per line, one line per (model, mode, kernel, key), sorted, with these columns:

    model      the container's name
    container  the sha the census computed over the container's graph.json files — a retrace changes
               it, and the model's rows are replaced, never added to
    mode       the served mode that formed the key (triton, triton-sequential)
    rungs_mb   the memory budgets the shadow planned under when it formed the key, ascending (a key
               formed only at small rungs is a tiled plan's shape class; its reason to exist); None
               for the profile's own budget
    ops        the graph ops that formed it (op uids), sorted; a None among them for a key formed
               outside any graph op (a flow's own call) or consolidated from a census taken before the
               shadow recorded ops. One row per key with the LIST, never one row per (op, key): a
               transformer forms one key in every layer, and a row per op made a class table ~15 MB
    kernel     the kernel's qualified name
    key        the key tuple as the certifier reads it (`key_repr`) — the bucketed shape class
    dtype      the dtypes the key names, in order
    tool       the census tool's tree revision

The census tool writes it, the certifier reads it and nothing else, and a confirmation run's refusal
(`autotune_certified.census_row`) names the row that should have held a missing key. No torch, no
device: a table is data.
"""
from __future__ import annotations

import sys

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

COLUMNS = ("model", "container", "mode", "rungs_mb", "ops", "kernel", "key", "dtype", "tool")

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
        if "op" in row and "ops" not in row:
            raise ValueError(f"{path}:{n}: a row of the one-row-per-op schema — convert the table once with "
                             f"`python tools/census_table.py migrate {path}`")
        missing = [c for c in COLUMNS if c not in row]
        if missing:
            raise ValueError(f"{path}:{n}: a row without {missing}")
        rows.append(row)
    return rows


def _sort_key(row: Dict) -> Tuple:
    return (row["model"], row["mode"], row["kernel"], row["key"])


def _ops(ops: Iterable[Optional[str]]) -> List[Optional[str]]:
    """The ops column's canonical form: distinct, None first, then by uid."""
    return sorted(set(ops), key=lambda o: (o is not None, o or ""))


def write(path: Path, rows: Iterable[Dict]) -> int:
    """Write the table, sorted, one row per (model, mode, kernel, key) — the ops and the rungs of
    duplicate rows merged into one list each — atomically, readable by all. Returns the row count."""
    merged: Dict[Tuple, Dict] = {}
    for r in rows:
        row = {c: r.get(c) for c in COLUMNS}
        if row["ops"] is None or isinstance(row["ops"], str):
            raise ValueError(f"a row's ops must be a list of op uids (None for no op): {row['ops']!r}")
        row["ops"] = _ops(row["ops"])
        ident = (row["model"], row["mode"], row["kernel"], row["key"])
        prev = merged.get(ident)
        if prev is None:
            merged[ident] = row
            continue
        prev["ops"] = _ops(prev["ops"] + row["ops"])
        if prev["rungs_mb"] is not None and row["rungs_mb"] is not None:
            prev["rungs_mb"] = sorted(set(prev["rungs_mb"]) | set(row["rungs_mb"]))
        else:
            prev["rungs_mb"] = None     # formed under the profile's own budget too: no rung is its reason
    out = sorted(merged.values(), key=_sort_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in out:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        # the table stays what it was; the half-written temporary never outlives the failure
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return len(out)


@contextlib.contextmanager
def locked(path: Path):
    """The table's exclusive lock (a sidecar `<table>.lock`): every read-modify-write of a table
    holds it — `replace_model` and every whole-table rewrite (consolidate, migrate). Readers need
    none: `write` replaces the file atomically."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def replace_model(path: Path, model: str, rows: Iterable[Dict], keyless_ops: int = 0) -> Tuple[int, int]:
    """The model's rows replaced by `rows` (a new census of that container), every other model's kept:
    a retrace replaces, never adds. Returns (rows removed, rows written for the model).

    `keyless_ops` is the derivation's PROOF that a model forms no key: the count of kernel-bearing ops it
    placed whose launcher key function returned no key (a profile's matrix unit carries its tile in the
    profile, never in an autotune key: volta-tensor-cores, 2026-10-08). Only with that proof may the
    rows be replaced by none.

    Under an exclusive lock beside the table: many census processes (one per model, the supervisor's
    2026-09-28 21:56) write one class table, and an unlocked read-modify-write lets the last writer
    drop the rows the others wrote in between."""
    rows = list(rows)
    if any(r.get("model") != model for r in rows):
        raise ValueError(f"replace_model({model!r}) was handed rows of another model")
    if not rows and keyless_ops <= 0:
        # A door, not a census: a census that formed no key (no mode asked, an empty key record, a
        # shadow that ran nothing) is not the knowledge that the model forms none — replacing its
        # rows by nothing unserved it silently (the tools audit, 2026-09-29).
        raise EmptyCensus(f"replace_model({model!r}): no rows — a census that formed no key is not the "
                          f"knowledge that the model forms none (no keyless op placed); its rows in {path.name} "
                          f"are kept")
    with locked(path):
        old = read(path)
        kept = [r for r in old if r["model"] != model]
        write(path, kept + rows)
    return len(old) - len(kept), len(rows)


def inputs_path(path: Path) -> Path:
    """The table's input record beside it (`<cls>g.inputs.jsonl`): one line per model, what its rows were
    derived FROM — the plan of every derivation, the code the derivation executed and the files it read, each
    with its hash (tools/derived_census.py `table --only-moved` re-derives a model only where one moved)."""
    return path.with_name(path.name[:-len(".jsonl")] + ".inputs.jsonl")


def read_inputs(path: Path) -> Dict[str, Dict]:
    """{model: its input record} of the table at `path`; a missing record file is none."""
    ip = inputs_path(path)
    if not ip.exists():
        return {}
    out = {}
    for n, line in enumerate(ip.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{ip}:{n}: not an input record ({exc})") from exc
            out[rec["model"]] = rec
    return out


def replace_inputs(path: Path, model: str, record: Dict) -> None:
    """The model's input record replaced, every other model's kept, under the TABLE's lock (the rows and their
    record are written by the same writers), atomically."""
    if record.get("model") != model:
        raise ValueError(f"replace_inputs({model!r}) was handed the record of {record.get('model')!r}")
    ip = inputs_path(path)
    with locked(path):
        recs = read_inputs(path)
        recs[model] = record
        fd, tmp = tempfile.mkstemp(dir=str(ip.parent), prefix=ip.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for m in sorted(recs):
                    fh.write(json.dumps(recs[m], sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, 0o644)
            os.replace(tmp, ip)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


class EmptyCensus(ValueError):
    """A model's rows would be replaced by none."""


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


def census_row(kernel_qual: str, key: tuple, memory_gb: Optional[int]) -> str:
    """Where the census table stands on a key a confirmation run missed, in words (installed as the library's miss
    explainer, `autotune_certified.set_miss_explainer`, by `neurobrix._nbxtensor_host.install`): the table of the profile in force and
    this card's memory class holds it (a certification gap) or not (a census defect)."""
    from nbxtensor.ops.autotune_certified import active_profile, key_repr
    census_table = sys.modules[__name__]
    prof = active_profile()
    if prof is None:
        return "no vendor profile is bound, so no census table can be named"
    if memory_gb is None:
        return f"the card's memory class is unknown, so no {prof[0]}/{prof[1]} census table can be named"
    line = census_table.key_line(kernel_qual, key_repr(tuple(key)))
    try:
        path, rows = census_table.rows_for(kernel_qual, key_repr(tuple(key)), prof[0], prof[1], memory_gb)
    except Exception as exc:  # noqa: BLE001 — said: an unreadable table is not an absent row
        return f"the census table could not be read ({type(exc).__name__}: {exc})"
    if not path.exists():
        return f"there is no census table {path} — take the census for this profile and memory class first"
    if not rows:
        return (f"ABSENT from the census table {path} — a CENSUS defect: fix the census tool so the shadow "
                f"forms {line}, certify it, re-run the cell")
    models = ", ".join(sorted({r["model"] for r in rows}))
    return (f"present in the census table {path} (models: {models}) but not certified for this card — "
            f"a CERTIFICATION gap: certify the table, re-run the cell")
