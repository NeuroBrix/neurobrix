#!/usr/bin/env python3
"""Which views break at a size other than the trace? — a census of the element-count law.

A view preserves the element count. Its recorded output dims are symbolic expressions,
so the law can be checked at ANY point of the symbols' domain, not only at the trace
point where every recorded expression reproduces the witnessed shape by construction.
This tool evaluates every view / reshape / _unsafe_view whose target carries an inferred
`-1` slot at a SECOND point — every symbol at 2·trace + 1, an odd value off the trace
that collides with nothing small — and names the sites where the input's element count
is not what the output's dims imply. Such a site fails the first time the container
runs at another length, another batch or another resolution.

The case that engraved it, 2026-09-26: granite-speech-3.3-8b re-traced at seq 31 with
batch 1 had its q-projection view [1, 31, 4096] -> [1, 31, 32, 128] recorded with the
head count as `s0 + s1` (1 + 31 = 32). The trace-point check passed in all forty layers;
at 209 tokens the sequential arm asked for [1, 209, 210, 128] and died. The same census
over the shared cache named 22 components, three classes: an inferred slot written as an
expression of the wrong symbols (granite), a batch frozen to the literal 1 in the target
(the LLM `view(1, S, -1)` class), and an inferred slot frozen to its trace literal.

Sites whose recorded dims do not even reproduce the trace are the CORRUPTED class
(symbolic dim != shape); they are counted apart, never evaluated, because a second
point means nothing for a first point that is already wrong.

Usage:
    python tools/view_numel_census.py                 # the cache the engine resolves
    python tools/view_numel_census.py --root DIR      # somewhere else
    python tools/view_numel_census.py --json          # machine-readable
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

VIEW_OPS = {"aten::view", "aten::reshape", "aten::_unsafe_view"}


def evaluate(dim, env: dict) -> int:
    """A recorded dim at the assignment `env` (symbol id -> value)."""
    if isinstance(dim, bool):
        raise TypeError("a bool is not a dim")
    if isinstance(dim, int):
        return dim
    kind = dim.get("type")
    if kind == "symbol":
        return int(env[dim["id"]])
    if kind == "literal":
        return int(dim["value"])
    left = evaluate(dim["left"], env)
    right = evaluate(dim["right"], env)
    if kind == "add":
        return left + right
    if kind == "sub":
        return left - right
    if kind == "mul":
        return left * right
    if kind == "floordiv":
        return left // right
    if kind == "ceildiv":
        return -(-left // right)
    if kind == "mod":
        return left % right
    if kind == "pow":
        return left ** right
    if kind == "max":
        return max(left, right)
    if kind == "min":
        return min(left, right)
    raise KeyError(f"unknown dim node type {kind!r}")


def second_point(symbols: dict) -> dict:
    """Every symbol at 2·trace + 1: off the trace, odd, never 0, 1 or 2."""
    return {sid: 2 * int((meta or {}).get("trace_value") or 1) + 1 for sid, meta in symbols.items()}


def census_one(graph: dict) -> dict:
    """{'evaluated': n, 'misbound': [sites], 'corrupted': n} for one graph."""
    tensors = graph.get("tensors") or {}
    ops = graph.get("ops") or {}
    ops = ops if isinstance(ops, list) else list(ops.values())
    symbols = (graph.get("symbolic_context") or {}).get("symbols") or {}
    at_trace = {sid: int((meta or {}).get("trace_value") or 1) for sid, meta in symbols.items()}
    off_trace = second_point(symbols)
    out = {"evaluated": 0, "misbound": [], "corrupted": 0, "unevaluable": 0}
    for op in ops:
        if op.get("op_type") not in VIEW_OPS:
            continue
        args = ((op.get("attributes") or {}).get("args") or [])
        if len(args) < 2:
            continue
        target = args[1]
        items = target.get("value") if isinstance(target, dict) else target
        if not isinstance(items, list) or -1 not in items:
            continue
        ins, outs = op.get("input_tensor_ids") or [], op.get("output_tensor_ids") or []
        t_in = tensors.get(ins[0], {}) if ins else {}
        t_out = tensors.get(outs[0], {}) if outs else {}
        in_dims = (t_in.get("symbolic_shape") or {}).get("dims")
        out_dims = (t_out.get("symbolic_shape") or {}).get("dims")
        if not in_dims or not out_dims or len(out_dims) != len(items):
            continue
        slot = items.index(-1)
        try:
            if ([evaluate(d, at_trace) for d in in_dims] != list(t_in.get("shape") or [])
                    or [evaluate(d, at_trace) for d in out_dims] != list(t_out.get("shape") or [])):
                out["corrupted"] += 1
                continue
            numel = 1
            for d in in_dims:
                numel *= evaluate(d, off_trace)
            known = 1
            for j, d in enumerate(out_dims):
                if j != slot:
                    known *= evaluate(d, off_trace)
            implied = evaluate(out_dims[slot], off_trace)
        except (KeyError, TypeError, ZeroDivisionError):
            out["unevaluable"] += 1
            continue
        out["evaluated"] += 1
        if known == 0 or numel % known or numel // known != implied:
            out["misbound"].append({
                "op": op.get("op_uid"), "slot": slot,
                "trace_in": list(t_in.get("shape") or []), "trace_out": list(t_out.get("shape") or []),
                "off_trace_in": [evaluate(d, off_trace) for d in in_dims],
                "off_trace_out": [evaluate(d, off_trace) for d in out_dims],
                "element_count_implies": (numel // known) if known else None,
                "recorded": out_dims[slot],
            })
    return out


def census(root: Path) -> dict:
    report = {"root": str(root), "components": {}, "totals": collections.Counter()}
    graphs = sorted(root.glob("*/components/*/graph.json"))
    if not graphs:
        raise SystemExit(f"no graph.json under {root}: a census over zero graphs proves nothing")
    for path in graphs:
        try:
            graph = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            report["components"][f"{path.parts[-4]}/{path.parts[-2]}"] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        one = census_one(graph)
        key = f"{path.parts[-4]}/{path.parts[-2]}"
        report["components"][key] = one
        report["totals"]["graphs"] += 1
        report["totals"]["evaluated"] += one["evaluated"]
        report["totals"]["corrupted"] += one["corrupted"]
        report["totals"]["unevaluable"] += one["unevaluable"]
        if one["misbound"]:
            report["totals"]["components_misbound"] += 1
            report["totals"]["sites_misbound"] += len(one["misbound"])
    report["totals"] = dict(report["totals"])
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="a cache directory (default: the one the engine resolves)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()
    if args.root:
        root = Path(args.root)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from neurobrix.core import paths
        root = paths.cache_dir()
    report = census(root)
    if args.json:
        print(json.dumps(report, indent=1))
        return 0
    t = report["totals"]
    print(f"view element-count census over {root}: {t.get('graphs', 0)} graphs, "
          f"{t.get('evaluated', 0)} inferred-slot views evaluated off-trace, "
          f"{t.get('corrupted', 0)} skipped as corrupted at the trace point, "
          f"{t.get('unevaluable', 0)} unevaluable")
    for key, one in sorted(report["components"].items(), key=lambda kv: -len(kv[1].get("misbound") or [])):
        sites = one.get("misbound") or []
        if not sites:
            continue
        s = sites[0]
        print(f"  {len(sites):5d}  {key}  e.g. {s['op']}: trace {s['trace_in']} -> {s['trace_out']}; "
              f"off-trace {s['off_trace_in']} -> {s['off_trace_out']}, the element count implies "
              f"{s['element_count_implies']} at slot {s['slot']}")
    print(f"components with a misbound inferred slot: {t.get('components_misbound', 0)} "
          f"({t.get('sites_misbound', 0)} sites)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
