#!/usr/bin/env python3
"""Consolidate census campaigns into THE census table (`neurobrix.kernels.census_table`).

    python tools/census_table.py consolidate --vendor nvidia --profile volta --class 32 \\
        --source <campaign census dir>@<tool tree revision> [--source ...]

Each source is a directory the census tool wrote: its census JSON (the models' `graph_sha`) and its
`<model>.<mode>[.probe][.r<rung>][.walk].keys` files (under `census_runs/`, or in the directory itself), which keep what the merged JSON
lost — the MODE and the RUNG each key was formed at. A model's rows come from the LAST source that
holds it (a later census of a model replaces an earlier one, never adds to it), and only when the
graph sha that source recorded is the container's in the cache today: a model retraced since its
census is left out and named, never written with keys of a graph that no longer exists. The census
tree's revision is the `tool` column; `op` is None for these rows (the shadow did not record it).

Nothing here runs a model or touches a device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CACHE = Path.home() / ".neurobrix" / ("ca" "che")
MODES = ("triton-sequential", "triton")   # longest first: "triton" is a suffix of the other


def graph_sha(model: str) -> str:
    """The census tool's own container hash (tools/certified_census.py `_graph_sha`)."""
    h = hashlib.sha256()
    for gp in sorted((CACHE / model / "components").glob("*/graph.json")):
        h.update(gp.name.encode())
        h.update(gp.read_bytes())
    return h.hexdigest()[:16]


def parse_keys_name(name: str):
    """'<model>.<mode>[.probe][.r<rung>][.walk].keys' -> (model, mode, rung_mb or None), peeled from the
    right because model names carry dots ('granite-3.1-1b-a400m-instruct')."""
    if not name.endswith(".keys"):
        return None
    stem = name[: -len(".keys")]
    rung = None
    for suffix in (".walk",):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    m = re.search(r"\.r(\d+)$", stem)
    if m:
        rung = int(m.group(1))
        stem = stem[: m.start()]
    if stem.endswith(".probe"):
        stem = stem[: -len(".probe")]
    for mode in MODES:
        if stem.endswith("." + mode):
            return stem[: -len(mode) - 1], mode, rung
    return None


def source_models(src: Path) -> dict:
    """model -> graph_sha, from every census JSON in the source directory."""
    out = {}
    for f in sorted(src.glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(doc, dict):             # a directory may hold other JSON (lists of keys, perf
            continue                              # tables): only a census document names models
        for model, row in (doc.get("models") or {}).items():
            if isinstance(row, dict) and row.get("graph_sha"):
                out[model] = row["graph_sha"]
    return out


def rows_of_source(src: Path, tool: str):
    """{model: [rows]} for one source directory, and the models it recorded no graph sha for."""
    from neurobrix.kernels import census_table as T
    shas = source_models(src)
    by_model, unhashed = {}, set()
    # the census tool writes its key files under census_runs/ beside the census JSON, or straight into
    # the --logs directory it was given (the Mac's campaigns: one logs_<tag>/ per invocation)
    runs = src / "census_runs" if (src / "census_runs").is_dir() else src
    for kf in sorted(runs.glob("*.keys")):
        parsed = parse_keys_name(kf.name)
        if parsed is None:
            raise SystemExit(f"{kf}: a key file name the census tool does not write")
        model, mode, rung = parsed
        if model not in shas:
            unhashed.add(model)
            continue
        for line in kf.read_text().splitlines():
            if "::" not in line:
                continue
            kernel, key = line.split("::", 1)
            by_model.setdefault(model, []).append(
                {"model": model, "container": shas[model], "mode": mode, "rungs_mb": [rung] if rung is not None else None, "op": None,
                 "kernel": kernel, "key": key, "dtype": T.dtypes_of(key), "tool": tool})
    return by_model, unhashed


def consolidate(a) -> int:
    from neurobrix.kernels import census_table as T
    chosen, origin = {}, {}
    for spec in a.source:
        path, _, tool = spec.partition("@")
        if not tool:
            raise SystemExit(f"--source {spec}: name the census tree's revision as <dir>@<rev>")
        by_model, unhashed = rows_of_source(Path(path), tool)
        for model in sorted(unhashed):
            print(f"[census-table] {model}: key files in {path} but no graph sha recorded — left out", flush=True)
        for model, rows in by_model.items():
            chosen[model], origin[model] = rows, path       # a later source replaces, never adds
    stale = []
    for model in sorted(chosen):
        if not (CACHE / model).is_dir():
            stale.append((model, "no such container in the cache"))
        elif graph_sha(model) != chosen[model][0]["container"]:
            stale.append((model, f"retraced since its census (census {chosen[model][0]['container']}, "
                                 f"cache {graph_sha(model)})"))
    for model, why in stale:
        print(f"[census-table] {model}: LEFT OUT — {why} (from {origin[model]})", flush=True)
        del chosen[model]
    out = T.table_path(a.vendor, a.profile, a.cls, Path(a.root) if a.root else None)
    n = T.write(out, [r for rows in chosen.values() for r in rows])
    print(f"[census-table] {out}: {n} rows, {len(chosen)} models; {len(stale)} left out", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("consolidate")
    c.add_argument("--vendor", required=True)
    c.add_argument("--profile", required=True)
    c.add_argument("--class", dest="cls", type=int, required=True, help="the memory class in GB (16, 32)")
    c.add_argument("--source", action="append", required=True, help="<census dir>@<tool tree revision>, oldest first")
    c.add_argument("--root", default=None, help="the table root (default: the engine package's config/census)")
    a = ap.parse_args(argv)
    return consolidate(a)


if __name__ == "__main__":
    sys.exit(main())
