#!/usr/bin/env python3
"""Which spatial shape arguments does a container carry SYMBOLICALLY?

Two questions, one probe, because answering only the first is what made
the 2026-08-28 swinir hypothesis look plausible and be wrong:

  1. what does the RUNTIME rewrite — runs the engine's own
     `_spatial_promotion_pass` over a copy of the container's ops
     metadata and diffs which shape args it turns symbolic;
  2. what did the TRACER already emit — censuses the shape args that
     arrive symbolic (`symbol` / `floordiv` / arith nodes) against the
     ones baked as literals.

The distinction is the whole diagnosis. A container whose window counts
arrive as `floordiv(s_h, window)` resolves them at any input size; one
whose counts arrive as the trace-time integers reassembles rows at the
trace width and renders horizontal banding off-trace, and NO runtime
pass corrects it — measured, the promotion pass rewrites nothing on
either the working or the broken Swin container.

POSITIVE CONTROL IS BUILT IN and runs by default (--no-control to skip).
A probe that flags nothing proves nothing until it flags a known case:
Sana 4Kpx is the model this promotion pass was written for, and it must
report a non-zero rewrite count or the probe is broken rather than the
container clean. The first version of this probe read `node["args"]`
instead of `node["attributes"]["args"]` and reported a serene 0
everywhere, control included.

Usage:
    python3 tools/probe_spatial_promotion.py <container> [<container> ...]
    python3 tools/probe_spatial_promotion.py --all-upscalers
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
CACHE = pathlib.Path.home() / ".neurobrix" / "cache"

CONTROL = ("Sana_1600M_4Kpx_BF16", "vae")
SHAPE_OPS = ("aten::view", "aten::_unsafe_view", "aten::reshape",
             "aten::expand")
SYM_TYPES = ("symbol", "arith", "expression", "expr", "binary_op",
             "floordiv", "mul", "add", "sub", "mod", "product", "neg")


def graphs(model: str):
    comp = CACHE / model / "components"
    if not comp.exists():
        return []
    return sorted(comp.glob("*/graph.json"))


def _flat(node):
    """(path, value) for every leaf inside an op's attribute args."""
    def walk(v, path):
        if isinstance(v, dict):
            t = v.get("type")
            if t in SYM_TYPES:
                yield path, ("SYM", t, json.dumps(v, sort_keys=True))
                return
            for k, sub in v.items():
                yield from walk(sub, f"{path}.{k}")
        elif isinstance(v, list):
            for i, sub in enumerate(v):
                yield from walk(sub, f"{path}[{i}]")
        elif isinstance(v, int) and not isinstance(v, bool):
            yield path, ("INT", v)
    attrs = node.get("attributes") or {}
    for key in ("args", "kwargs", "shape"):
        if key in attrs:
            yield from walk(attrs[key], key)


def _snapshot(ops):
    return {uid: dict(_flat(n)) for uid, n in ops.items()}


def runtime_rewrites(dag) -> int:
    """How many shape args the engine's promotion pass turns symbolic."""
    from neurobrix.triton.promotion import _spatial_promotion_pass
    ops = copy.deepcopy(dag["ops"])
    symbols = (dag.get("symbolic_context") or {}).get("symbols") or {}
    before = _snapshot(ops)
    _spatial_promotion_pass(dag, dag["tensors"], ops, symbols, set(), set())
    after = _snapshot(ops)
    return sum(1 for uid in ops
               for p in set(before[uid]) | set(after[uid])
               if before[uid].get(p) != after[uid].get(p))


def traced_symbolic(dag):
    """Symbolic vs baked census over shape-producing ops."""
    sym = collections.Counter()
    baked_windows = 0
    for _uid, n in dag["ops"].items():
        if n.get("op_type") not in SHAPE_OPS:
            continue
        blob = json.dumps((n.get("attributes") or {}).get("args"))
        for t in ("floordiv", "symbol", "mul", "add"):
            sym[t] += blob.count(f'"{t}"')
        # A 6-D output is the Swin window-partition layout
        # (B, H/ws, ws, W/ws, ws, C). Carrying no symbolic node there
        # means both counts were frozen at the trace size.
        out = (n.get("output_shapes") or [[]])[0]
        if len(out) == 6 and '"floordiv"' not in blob:
            baked_windows += 1
    return sym, baked_windows


def report(model: str) -> None:
    found = graphs(model)
    if not found:
        print(f"{model}: not in the local cache ({CACHE})")
        return
    for g in found:
        dag = json.loads(g.read_text())
        syms = (dag.get("symbolic_context") or {}).get("symbols") or {}
        named = {i.get("name"): i.get("trace_value") for i in syms.values()}
        rewrites = runtime_rewrites(dag)
        sym, baked = traced_symbolic(dag)
        label = f"{model}/{g.parent.name}"
        print(f"{label:46s} ops={len(dag['ops']):6d} symbols={named}")
        print(f"{'':46s} runtime promotion rewrites : {rewrites}")
        print(f"{'':46s} traced symbolic shape nodes: "
              f"floordiv={sym['floordiv']} symbol={sym['symbol']}")
        verdict = ("BAKED WINDOW COUNTS" if baked else "clean")
        print(f"{'':46s} 6-D window views with NO symbolic count: "
              f"{baked}  -> {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*")
    ap.add_argument("--all-upscalers", action="store_true")
    ap.add_argument("--no-control", action="store_true")
    args = ap.parse_args()

    models = list(args.models)
    if args.all_upscalers:
        models += [p.name for p in sorted(CACHE.iterdir())
                   if any(k in p.name.lower()
                          for k in ("swin", "hat-", "esrgan"))]
    if not models:
        ap.error("name at least one container, or pass --all-upscalers")

    if not args.no_control:
        cg = CACHE / CONTROL[0] / "components" / CONTROL[1] / "graph.json"
        if cg.exists():
            n = runtime_rewrites(json.loads(cg.read_text()))
            state = "OK" if n else "PROBE IS BROKEN"
            print(f"[positive control] {CONTROL[0]}/{CONTROL[1]}: "
                  f"{n} rewrites -> {state}")
            if not n:
                print("  The promotion pass was written for this container. "
                      "A zero here means the probe stopped seeing the args, "
                      "not that the container is clean. Fix the probe.")
                return 2
        else:
            print(f"[positive control] {CONTROL[0]} not cached — "
                  f"a zero below is UNVERIFIED")
        print()

    for m in models:
        report(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
