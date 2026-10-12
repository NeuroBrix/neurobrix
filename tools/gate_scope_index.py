#!/usr/bin/env python3
"""The cell side of the computed gate scope: what each container's graphs present to the engine.

One record per container, built from the graph AS THE EXECUTOR RUNS IT (`derived_census.runtime_graph`:
the SDPA normalisation, the K/V layout mark and the MoE fusion the GraphExecutor applies at load — the
fused `custom::moe_fused` op exists only there), read through the engine's own cache door, READ-ONLY, and
cached under the scope's scratch keyed by every graph's size and mtime and this module's RECORD_VERSION:

    ops             the op types of every component's runtime graph ("aten::div", "custom::moe_fused")
    sigs            per op type, the distinct ARGUMENT SIGNATURES its instances present to a kernel wrapper:
                    each positional argument as {"t": <tensor dtype>} | {"s": <scalar>} | {"r": <record>},
                    each keyword the same — what a guard on a wrapper's parameters is evaluated over
    arg_records     every distinct op-argument record ({"type": "dtype", "value": "torch.bool"}, ...): the
                    values an attribute resolver is handed
    explicit_dtypes the dtype values the graph passes as an ARGUMENT (records of type "dtype")
    tensor_dtypes   the dtype of every tensor
    torch_dtypes    every component's traced `torch_dtype`
    flow            the topology's flow type

Dtype names are written bare (`bool`, `float64`), the form `canonical_dtype` gives any spelling.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RECORD_VERSION = 3
TOOLS = Path(__file__).resolve().parent


def canonical_dtype(x) -> str:
    """One spelling of a dtype: `torch.bool`, `bool_`, NBXDtype.bool_ and `bool` are all `bool`."""
    name = getattr(x, "name", None) if not isinstance(x, str) else x
    name = str(name if name is not None else x)
    for pre in ("torch.", "NBXDtype."):
        if name.startswith(pre):
            name = name[len(pre):]
    return name.rstrip("_")


def _arg_sig(a, tdtype: dict):
    if isinstance(a, dict):
        if a.get("type") in ("tensor", "tensor_ref") and "tensor_id" in a:
            return {"t": canonical_dtype(tdtype.get(a["tensor_id"], "?"))}
        if a.get("type") == "scalar":
            return {"s": a.get("value")}
        return {"r": a}
    if isinstance(a, list):
        return {"l": [_arg_sig(x, tdtype) for x in a]}
    return {"s": a}


def _records(node, out: set, strings: set) -> None:
    if isinstance(node, dict):
        if "type" in node:
            out.add(json.dumps(node, sort_keys=True))
            if isinstance(node.get("value"), str):
                strings.add(node["value"])
        for v in node.values():
            _records(v, out, strings)
    elif isinstance(node, list):
        for v in node:
            _records(v, out, strings)


def graph_features(g: dict) -> dict:
    tdtype = {tid: t.get("dtype") for tid, t in (g.get("tensors") or {}).items()}
    ops, sigs, recs, strings, explicit = set(), {}, set(), set(), set()
    for op in (g.get("ops") or {}).values():
        t = op.get("op_type", "")
        ops.add(t)
        at = op.get("attributes") or {}
        sig = {"args": [_arg_sig(a, tdtype) for a in at.get("args") or []],
               "kwargs": {k: _arg_sig(v, tdtype) for k, v in (at.get("kwargs") or {}).items()}}
        sigs.setdefault(t, set()).add(json.dumps(sig, sort_keys=True))
        _records(at, recs, strings)
    for r in recs:
        d = json.loads(r)
        if d.get("type") == "dtype" and isinstance(d.get("value"), str):
            explicit.add(canonical_dtype(d["value"]))
    tdt = {canonical_dtype(v) for v in tdtype.values() if v}
    return {"ops": ops, "sigs": sigs, "arg_records": recs, "strings": strings, "explicit_dtypes": explicit,
            "tensor_dtypes": tdt, "torch_dtype": canonical_dtype(g.get("torch_dtype") or "")}


def _stamp(model_dir: Path) -> list:
    return sorted([str(p.relative_to(model_dir)), p.stat().st_size, int(p.stat().st_mtime)]
                  for p in model_dir.glob("components/*/graph.json"))


def build_record(cache: Path, model: str) -> dict:
    """The record of one container — the runtime graph through `derived_census.runtime_graph` (the engine's
    load passes), every component."""
    sys.path.insert(0, str(TOOLS))
    import derived_census as DC                         # noqa: E402 — the one runtime-graph derivation
    d = cache / model
    stamp = _stamp(d)
    comps, agg = {}, {"ops": set(), "sigs": {}, "arg_records": set(), "strings": set(), "explicit_dtypes": set(),
                      "tensor_dtypes": set()}
    for rel, _, _ in stamp:
        comp = Path(rel).parts[1]
        f = graph_features(DC.runtime_graph(model, comp))
        comps[comp] = {"ops": sorted(f["ops"]), "torch_dtype": f["torch_dtype"]}
        for k in ("ops", "arg_records", "strings", "explicit_dtypes", "tensor_dtypes"):
            agg[k] |= f[k]
        for t, s in f["sigs"].items():
            agg["sigs"].setdefault(t, set()).update(s)
        DC._RUNTIME_GRAPHS.clear()
    topo = d / "topology.json"
    flow = (json.loads(topo.read_text()).get("flow") or {}).get("type") if topo.is_file() else None
    return {"model": model, "version": RECORD_VERSION, "stamp": stamp, "flow": flow, "components": comps,
            "ops": sorted(agg["ops"]), "sigs": {t: sorted(s) for t, s in sorted(agg["sigs"].items())},
            "arg_records": sorted(agg["arg_records"]), "strings": sorted(agg["strings"]),
            "explicit_dtypes": sorted(agg["explicit_dtypes"]), "tensor_dtypes": sorted(agg["tensor_dtypes"]),
            "torch_dtypes": sorted({c["torch_dtype"] for c in comps.values() if c["torch_dtype"]})}


def cached(cache: Path, model: str, store: Path):
    """The stored record when its graphs' stamps and version are unchanged, else None."""
    d = cache / model
    f = store / f"{model}.json"
    if not d.is_dir():
        # a container retraced away after the gate ran it: its record, indexed while it was there, is the
        # container the cell ran (named by the caller); without one the cell cannot be scoped
        if f.is_file():
            rec = json.loads(f.read_text())
            if rec.get("version") == RECORD_VERSION:
                return dict(rec, gone=True)
        raise SystemExit(f"REFUSED: the container {model} is not in {cache} and no record of it was indexed "
                         f"— a cell's scope is computed from the container it runs")
    stamp = _stamp(d)
    if not stamp:
        raise SystemExit(f"REFUSED: {d} holds no components/*/graph.json")
    if f.is_file():
        rec = json.loads(f.read_text())
        if rec.get("stamp") == stamp and rec.get("version") == RECORD_VERSION:
            return rec
    return None


def records(cache: Path, models, store: Path, python: str, src: Path, jobs: int = 6) -> dict:
    """{model: record} — the stale ones built in parallel subprocesses (one per container, the engine's
    `src` first on the path, no card visible)."""
    store.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for m in models:
        r = cached(cache, m, store)
        if r is None:
            todo.append(m)
        else:
            out[m] = r

    def one(m):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"{src}:{TOOLS}", PYTHONNOUSERSITE="1")
        p = subprocess.run([python, str(Path(__file__).resolve()), "--cache", str(cache), "--store", str(store), m],
                           capture_output=True, text=True, env=env)
        if p.returncode != 0:
            raise SystemExit(f"REFUSED: indexing {m} failed: {(p.stderr or p.stdout).strip()[-400:]}")
        return m

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for m in ex.map(one, todo):
            out[m] = cached(cache, m, store)
    return out


if __name__ == "__main__":                                  # index the named containers (one process each)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        sys.exit("REFUSED: the index reads graphs with no card visible (CUDA_VISIBLE_DEVICES='')")
    store = Path(a.store)
    store.mkdir(parents=True, exist_ok=True)
    for m in a.models:
        rec = build_record(Path(a.cache), m)
        f = store / f"{m}.json"
        tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, f)
        print(m, rec["flow"], len(rec["ops"]), "ops", sum(len(v) for v in rec["sigs"].values()), "sigs", flush=True)
    sys.exit(0)
