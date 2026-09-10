#!/usr/bin/env python3
"""MoE routing probe — does the executed graph carry trace-time expert bounds?

The alert: a MoE trace burns the per-expert token counts of the trace batch into
`aten::slice` bounds (topk -> sort -> floor_divide -> index -> per-expert slice).
Those bounds depend on values the router produced at trace time, not on seq_len,
so no symbolic shape can cover them. If such a slice still sits in
`execution_order` when the model runs, every generation routes with the trace's
expert histogram.

`core/runtime/graph/moe_fusion.py` claims to remove exactly those ops. This probe
runs that pass the way the runtime runs it and counts what survives, per
component, WITHOUT loading weights or touching a GPU.

Usage:
    python3 tools/moe_routing_probe.py <container-graph-dir> --src <src> [--norm-topk-prob false]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional


def _attr_ints(op: Dict[str, Any]) -> Dict[str, Any]:
    """Slice bounds live either in `attributes.args[]` (positional) or by name."""
    attrs = op.get("attributes", {}) or {}
    out = dict(attrs)
    args = attrs.get("args")
    if isinstance(args, list):
        for i, a in enumerate(args):
            v = a.get("value") if isinstance(a, dict) else a
            out[f"arg{i}"] = v
    return out


def _first_axis(shapes: Any) -> Optional[int]:
    if isinstance(shapes, list) and shapes and isinstance(shapes[0], list) and shapes[0]:
        v = shapes[0][0]
        return v if isinstance(v, int) else None
    return None


def frozen_expert_slices(dag: Dict[str, Any], trace_seq_len: int) -> List[Dict[str, Any]]:
    """Axis-0 slices whose literal bounds can only come from the trace's routing.

    A per-expert slice cuts the gathered buffer [seq_len * top_k, dim] at the
    cumulative token counts of the trace's router. The signature is: dim 0, both
    bounds finite integers, and an input first axis that is a multiple of the
    trace's seq_len (the gather's [seq_len * top_k, dim]) — never seq_len itself,
    which the promotion pass covers.
    """
    ops = dag.get("ops", {})
    found = []
    for uid in dag.get("execution_order", []):
        op = ops.get(uid)
        if op is None or op.get("op_type") != "aten::slice":
            continue
        a = _attr_ints(op)
        dim = a.get("dim", a.get("arg1"))
        start = a.get("start", a.get("arg2"))
        end = a.get("end", a.get("arg3"))
        if dim != 0 or not isinstance(end, int) or not isinstance(start, int):
            continue
        if end > 2 ** 40:  # sys.maxsize sentinel = "to the end", not frozen
            continue
        buf = _first_axis(op.get("input_shapes"))
        if buf is None or buf == trace_seq_len:
            continue
        if buf % trace_seq_len != 0:
            continue
        found.append({"uid": uid, "start": start, "end": end, "buffer": buf,
                      "parent": op.get("parent_module", "")})
    return found


def sum_consumption(dag: Dict[str, Any]) -> Dict[str, int]:
    """`aten::sum` collapses the per-expert contributions. Ignored = routing lost."""
    ops = dag.get("ops", {})
    order = dag.get("execution_order", [])
    consumers: Dict[str, List[str]] = {}
    for uid in order:
        op = ops.get(uid) or {}
        for t in op.get("input_tensor_ids", []) or []:
            consumers.setdefault(t, []).append(uid)
        for a in (op.get("attributes", {}) or {}).get("args", []) or []:
            if isinstance(a, dict) and isinstance(a.get("tensor_id"), str):
                consumers.setdefault(a["tensor_id"], []).append(uid)
    total = dead = 0
    for uid in order:
        op = ops.get(uid) or {}
        if op.get("op_type") != "aten::sum":
            continue
        total += 1
        if not any(consumers.get(t) for t in op.get("output_tensor_ids", []) or []):
            dead += 1
    return {"sum_ops": total, "sum_outputs_unconsumed": dead}


def probe(graph_path: str, family: str, norm_topk_prob: bool) -> Dict[str, Any]:
    with open(graph_path) as f:
        dag = json.load(f)

    sym = dag.get("symbolic_context", {}) or {}
    # `symbolic_context` is {"symbols": {sN: {name, trace_value, ...}}, ...}.
    # A component can carry several seq_len symbols (one per input); the expert
    # gather is indexed by the LM's own sequence, so take the widest — a wrong
    # divisor below would silently reclassify every slice.
    trace_seq_lens = sorted({
        m.get("trace_value") for m in (sym.get("symbols") or {}).values()
        if isinstance(m, dict) and m.get("name") == "seq_len"
        and isinstance(m.get("trace_value"), int)})
    trace_seq_len = trace_seq_lens[-1] if trace_seq_lens else None
    if trace_seq_len is None:
        return {"graph": graph_path, "trace_seq_len": None, "symbolic_context": sym,
                "topk_ops_by_k": {}, "moe_layer_candidates": 0,
                "frozen_expert_slices_before": 0, "frozen_expert_slices_after": 0,
                "fused_ops": 0, "ops_before": len(dag.get("ops", {})),
                "ops_after": len(dag.get("ops", {})), "order_before": 0,
                "order_after": 0, "aten_sum": {"sum_ops": 0, "sum_outputs_unconsumed": 0},
                "surviving_examples": [], "removed_examples": [],
                "note": "no seq_len symbol — not a sequence component"}

    ops = dag.get("ops", {})
    topk_k = Counter()
    for uid in dag.get("execution_order", []):
        op = ops.get(uid) or {}
        if op.get("op_type") != "aten::topk":
            continue
        a = _attr_ints(op)
        k = a.get("k", a.get("arg1"))
        if isinstance(k, int):
            topk_k[k] += 1

    before = frozen_expert_slices(dag, trace_seq_len)
    sums = sum_consumption(dag)
    order_before = len(dag.get("execution_order", []))

    from neurobrix.core.runtime.graph.moe_fusion import detect_and_fuse_moe
    dag = detect_and_fuse_moe(dag, family, norm_topk_prob=norm_topk_prob, declared=True)

    after = frozen_expert_slices(dag, trace_seq_len)
    fused = [u for u, o in dag.get("ops", {}).items()
             if (o or {}).get("op_type") == "custom::moe_fused"]
    fused_in_order = [u for u in dag.get("execution_order", []) if u in set(fused)]

    return {
        "graph": graph_path,
        "trace_seq_len": trace_seq_len,
        "symbolic_context": sym,
        "topk_ops_by_k": dict(topk_k),
        "moe_layer_candidates": sum(n for k, n in topk_k.items() if k > 1),
        "frozen_expert_slices_before": len(before),
        "frozen_expert_slices_after": len(after),
        "fused_ops": len(fused_in_order),
        "ops_before": len(ops),
        "ops_after": len(dag.get("ops", {})),
        "order_before": order_before,
        "order_after": len(dag.get("execution_order", [])),
        "aten_sum": sums,
        "surviving_examples": after[:10],
        "removed_examples": before[:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_dir", help="container graph dir (holds <component>/graph.json)")
    ap.add_argument("--src", default=None, help="engine src/ to import from")
    ap.add_argument("--family", default="llm")
    ap.add_argument("--norm-topk-prob", default="false")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.src:
        # APPEND, never insert: an inserted path shadows the engine under test
        # with this repo's own src and both arms then measure the same tree.
        sys.path.append(os.path.abspath(args.src))

    norm = args.norm_topk_prob.lower() in ("1", "true", "yes")
    results = []
    for comp in sorted(os.listdir(args.graph_dir)):
        gp = os.path.join(args.graph_dir, comp, "graph.json")
        if not os.path.isfile(gp):
            continue
        r = probe(gp, args.family, norm)
        r["component"] = comp
        results.append(r)
        print(f"=== {comp} ===")
        print(f"  trace_seq_len          : {r['trace_seq_len']}  {r.get('note','')}")
        print(f"  symbolic_context       : {r['symbolic_context']}")
        print(f"  topk ops by k          : {r['topk_ops_by_k']}")
        print(f"  MoE layer candidates   : {r['moe_layer_candidates']}")
        print(f"  fused ops in order     : {r['fused_ops']}")
        print(f"  frozen slices BEFORE   : {r['frozen_expert_slices_before']}")
        print(f"  frozen slices AFTER    : {r['frozen_expert_slices_after']}")
        print(f"  ops dict {r['ops_before']} -> {r['ops_after']}  |  execution_order {r['order_before']} -> {r['order_after']}")
        print(f"  aten::sum              : {r['aten_sum']}")
        if r["surviving_examples"]:
            print("  SURVIVING (executed with trace bounds):")
            for e in r["surviving_examples"]:
                print(f"    {e['uid']} [{e['start']}:{e['end']}] of buffer {e['buffer']} @ {e['parent']}")
        print()

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"written: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
