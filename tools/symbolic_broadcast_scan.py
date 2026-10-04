#!/usr/bin/env python3
"""Static scan: an elementwise op whose inputs broadcast at the trace values and NOT at others.

A container is symbolic when every variable dim is an expression of the request's symbols. A
dim the tracer froze, or bound to the wrong symbol, still agrees with its neighbours at the
trace values — that is how it passed the trace — and disagrees at any other request. The
regression matrix met one on 2026-09-26: the Wan VAE encoder's mid-block attention reshapes
(b, c, t, h, w) -> (b*t, c, h, w); traced with one latent frame and batch 1, the view bound its
dim 0 to the batch symbol and folded the time expression into the channel dim (T'*384), while the
norm's expand kept a literal 384 — `aten.div::62` broadcasts 384 with 384 at the trace and 7296
with 384 at 352x832.

For every op with two or more tensor inputs whose op type broadcasts, each input's
`symbolic_shape.dims` is evaluated at the trace values and at two assignments away from them
(each symbol at 3x and 5x its trace value, its minimum respected); right-aligned dims must be
equal or 1. A pair that broadcasts at the trace and not away from it is reported with its op,
module and both expressions' values. Read-only over the graphs; no model is run.

    python tools/symbolic_broadcast_scan.py [--cache DIR] [--models A,B] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# The scan's core is the engine's own door (`core/flow/input_extent.broadcast_breaks`): the flows
# refuse a container whose graph does not follow the extent they feed with the SAME function this
# tool sweeps the catalogue with — one implementation, evaluated by the runtime's resolver.
from neurobrix.core.flow.input_extent import BROADCASTING, broadcast_breaks  # noqa: E402,F401


def scan_graph(graph: dict):
    symbols = (graph.get("symbolic_context") or {}).get("symbols") or {}
    if not symbols:
        return [], 0
    trace = {k: int(v["trace_value"]) for k, v in symbols.items()}
    # One request dimension moved at a time (3x its trace value), so each break names what it
    # follows: a dim that breaks when the BATCH moves is not the defect of one that breaks when the
    # length does. Symbols sharing a NAME move together — two `seq_len` symbols are one request
    # length measured twice, and moving one alone asks for a request no one can make.
    groups: dict = {}
    for k, v in symbols.items():
        groups.setdefault(v.get("name") or k, []).append(k)
    by_op: dict = {}
    unknown = 0
    for name, ks in sorted(groups.items()):
        env = {**trace, **{k: max(trace[k] * 3, int((symbols[k].get("constraints") or {}).get("min", 1)))
                           for k in ks}}
        breaks, u = broadcast_breaks(graph, env, trace)
        unknown = max(unknown, u)
        for b in breaks:
            rec = by_op.setdefault(b["op"], {"op": b["op"], "op_type": b["op_type"],
                                             "module": b["module"], "inputs": b["inputs"],
                                             "at_trace": b["at_trace"], "breaks_when": [], "away": {}})
            rec["breaks_when"].append(name)
            rec["away"][name] = b["at_binding"]
    order = {uid: i for i, uid in enumerate(graph.get("ops") or {})}
    return sorted(by_op.values(), key=lambda r: order[r["op"]]), unknown


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=os.path.expanduser("~/.neurobrix/ca" + "che"))
    ap.add_argument("--models", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    want = set(a.models.split(",")) if a.models else None
    report = {}
    for g in sorted(Path(a.cache).glob("*/components/*/graph.json")):
        model, comp = g.parts[-4], g.parts[-2]
        if want and model not in want:
            continue
        found, unknown = scan_graph(json.loads(g.read_text()))
        if found or unknown:
            report.setdefault(model, {})[comp] = {"breaks": found, "unevaluable_ops": unknown}
        by = {}
        for f in found:
            by.setdefault(tuple(f["breaks_when"]), []).append(f)
        for moved, fs in sorted(by.items()):
            f = fs[0]
            print(f"{model} | {comp} | breaks when {','.join(moved)} moves | {len(fs)} op(s) | first {f['op']} "
                  f"({f['module']}): trace {f['at_trace']} -> {f['away'][moved[0]]}")
    n = sum(len(c["breaks"]) for m in report.values() for c in m.values())
    print(f"\n{n} op(s) broadcast at the trace and not away from it, in "
          f"{sum(1 for m in report.values() for c in m.values() if c['breaks'])} component(s) of "
          f"{sum(1 for m in report.values() if any(c['breaks'] for c in m.values()))} container(s); "
          f"ops with an expression this scan cannot evaluate: "
          f"{sum(c['unevaluable_ops'] for m in report.values() for c in m.values())}")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
