#!/usr/bin/env python3
"""A WEIGHT's shape is architectural. It can never depend on a request dimension.

Principle 1 says a dim left concrete where a symbol belongs is a tracer bug. The inverse is
equally a tracer bug and nothing looked for it: a SYMBOL where a constant belongs. An
activation's dims follow the request; a parameter's dims follow the model, and a tracer that
binds one to the other has mistaken a coincidence for a relationship.

WHAT IT COST, measured 2026-09-23
---------------------------------
mochi-1-preview was retraced at `--trace-spatial 30,54`. At those values

    time(10) + width(54) == 64 == the attention head dim

so the tracer bound `param::pos_frequencies` axis 2 — the head dim — to `s1 + s3`. At the
container's own default request that resolves to 14 + 106 = 120 against an activation of 64,
and every run dies in denoise step 1:

    Cannot broadcast (2, 22260, 24, 64) and (11872, 24, 120)

The same container retraced at `26,30` (10 + 30 = 40, no collision) carries the literal 64 and
runs. Seven runs were spent finding that, because the two builds sat in different caches and
nothing anywhere said one of them was malformed.

Vacuous-gates register entry 91 named the collision guard's blind spot — it checks products and
affine forms and never sums. This is the detector that spot needed, and it is deliberately
WIDER than sums: any non-literal dim on a parameter is the same mistake whatever its shape.

Usage:
    python tools/weights_are_not_symbolic.py [--cache DIR] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def offending_parameters(graph: dict):
    """(tensor_id, axis, expression) for every parameter dim that is not an int literal."""
    out = []
    for tid, t in (graph.get("tensors") or {}).items():
        if not isinstance(t, dict) or not t.get("is_parameter"):
            continue
        dims = (t.get("symbolic_shape") or {}).get("dims") or []
        for axis, dim in enumerate(dims):
            if isinstance(dim, dict):          # a literal is a plain int; anything else is an expression
                out.append((tid, axis, dim))
    return out


#: (activation arg index, its contracted axis, parameter arg index, the parameter's contracted axis)
_CONTRACTIONS = {"aten::mm": (0, -1, 1, 0), "aten::addmm": (1, -1, 2, 0), "aten::linear": (0, -1, 1, 1),
                 "aten::matmul": (0, -1, 1, -2), "aten::bmm": (0, -1, 1, -2)}
#: ops a parameter passes through unchanged in what it IS (its axes move, its extents do not)
_LAYOUT_OPS = {"aten::t", "aten::transpose", "aten::permute", "aten::view", "aten::reshape",
               "aten::_unsafe_view", "aten::expand", "aten::unsqueeze", "aten::squeeze"}


def _is_weight(graph: dict, tid: str, depth: int = 4) -> bool:
    """A parameter, or a parameter carried through layout ops only."""
    t = (graph.get("tensors") or {}).get(tid) or {}
    if t.get("is_parameter") or str(tid).startswith("param::"):
        return True
    op = (graph.get("ops") or {}).get(t.get("producer_op_uid") or "") or {}
    if depth and op.get("op_type") in _LAYOUT_OPS and op.get("input_tensor_ids"):
        return _is_weight(graph, op["input_tensor_ids"][0], depth - 1)
    return False


def offending_contractions(graph: dict):
    """(op_uid, activation tensor id, axis, expression) for every activation dim that a
    contraction multiplies against a WEIGHT's dim and that is not an int literal.

    The weight's extent is architectural, so the activation dim it is contracted with is the same
    number by construction: a symbol there is the mirror of a symbol on the parameter itself —
    a coincidence mistaken for a relationship. Measured 2026-09-27 on PixArt-XL-2-1024-MS: the
    timestep sinusoid's half-width 128 (Timesteps(num_channels=256), typed inside diffusers) was
    bound to the latent height (128 at the 1024 trace); off the trace aten.addmm::0 received
    2x160 against the 256x1152 time projection."""
    out = []
    tensors = graph.get("tensors") or {}
    for uid, op in (graph.get("ops") or {}).items():
        spec = _CONTRACTIONS.get(op.get("op_type"))
        ins = op.get("input_tensor_ids") or []
        if spec is None or len(ins) <= max(spec[0], spec[2]):
            continue
        act, a_ax, w, _ = spec
        if not _is_weight(graph, ins[w]) or _is_weight(graph, ins[act]):
            continue
        dims = ((tensors.get(ins[act]) or {}).get("symbolic_shape") or {}).get("dims") or []
        if dims and isinstance(dims[a_ax], dict):
            out.append((uid, ins[act], len(dims) + a_ax if a_ax < 0 else a_ax, dims[a_ax]))
    return out


def scan(cache: Path):
    """{model: {component: [(tensor_id, axis, expr), ...]}} for every container that offends."""
    found: dict = {}
    for g in sorted(cache.glob("*/components/*/graph.json")):
        try:
            d = json.loads(g.read_text())
        except Exception:                      # noqa: BLE001 — an unreadable graph is not this tool's finding
            continue
        bad = offending_parameters(d)
        if bad:
            found.setdefault(g.parts[-4], {})[g.parts[-2]] = bad
    return found


def scan_contractions(cache: Path):
    """{model: {component: [(op_uid, tensor_id, axis, expr), ...]}} — the contraction half."""
    found: dict = {}
    for g in sorted(cache.glob("*/components/*/graph.json")):
        try:
            d = json.loads(g.read_text())
        except Exception:                      # noqa: BLE001 — an unreadable graph is not this tool's finding
            continue
        bad = offending_contractions(d)
        if bad:
            found.setdefault(g.parts[-4], {})[g.parts[-2]] = bad
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/.neurobrix/ca" + "che"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    found = scan(Path(a.cache))
    contractions = scan_contractions(Path(a.cache))
    if a.json:
        print(json.dumps({"parameters": {m: {c: [(t, ax, e.get("type"), e.get("trace")) for t, ax, e in v]
                                             for c, v in comps.items()} for m, comps in found.items()},
                          "contractions": {m: {c: [(u, t, ax, e.get("type"), e.get("trace")) for u, t, ax, e in v]
                                               for c, v in comps.items()} for m, comps in contractions.items()}},
                         indent=2))
        return 1 if (found or contractions) else 0
    total = sum(len(v) for comps in found.values() for v in comps.values())
    print(f"{len(found)} container(s), {total} parameter dim(s) that are not literals\n")
    for model, comps in sorted(found.items(), key=lambda kv: -sum(len(v) for v in kv[1].values())):
        n = sum(len(v) for v in comps.values())
        print(f"  {model}  ({n} dims)")
        for comp, bad in comps.items():
            tid, axis, expr = bad[0]
            print(f"    {comp}: {tid} axis {axis} is a {expr.get('type')!r} "
                  f"(trace {expr.get('trace')}), and {len(bad) - 1} more")
    total = sum(len(v) for comps in contractions.values() for v in comps.values())
    print(f"\n{len(contractions)} container(s), {total} activation dim(s) contracted against a weight that are not literals\n")
    for model, comps in sorted(contractions.items(), key=lambda kv: -sum(len(v) for v in kv[1].values())):
        n = sum(len(v) for v in comps.values())
        print(f"  {model}  ({n} contractions)")
        for comp, bad in comps.items():
            uid, tid, axis, expr = bad[0]
            print(f"    {comp}: {uid} contracts {tid} axis {axis}, a {expr.get('type')!r} "
                  f"(trace {expr.get('trace')}), against a weight; and {len(bad) - 1} more")
    return 1 if (found or contractions) else 0


if __name__ == "__main__":
    raise SystemExit(main())
