#!/usr/bin/env python
"""Follow every declared input symbol through a graph and say WHERE its chain breaks.

The defect class (2026-09-16, the third of its family this week): a symbolic
expression meets an operator the tracer does not propagate, the operator
evaluates it at the trace value, and the RESULT is recorded instead of the
expression. The declaration stays symbolic; the body becomes literal. The
container then announces one thing and demands another, and every gate we own
reads the declaration.

So the question is not which containers lie — it is BY WHICH OPERATORS the
symbolic is lost. This walks each graph and, per declared input symbol:

  * taints the input tensor the symbol binds to, and propagates the taint
    through the producer/consumer links;
  * calls an op a CARRIER when its recorded arguments reference the symbol
    (`{"type": "symbol", "id": "sN"}`) or an expression naming it;
  * calls an op a BREAK when it consumes a tainted tensor, records NO symbol,
    and records a LITERAL that (a) equals the symbol's trace value or a value
    derived from it by a relation a shape op performs (v, v//2, v//4, v//8,
    v*2, v*4, v-1, v+1, v+2, v*v), (b) is at least 4, and (c) APPEARS IN THE
    OP'S OUTPUT SHAPE. The third condition is what makes this a measurement
    rather than a list: `transpose(1, 2)`, `permute`, `unsqueeze` and `select`
    record AXIS INDICES as literals, and without (c) every one of them is
    counted the moment some symbol's trace value happens to be 2 or 4. The
    first run of this census, without (c), reported 57 transposes and 31
    permutes; they are axis arguments, not lost expressions. What (c) keeps is
    a literal the op WROTE INTO A SHAPE — the expression the tracer threw away;
  * reports the last carrier before it and the first break, with the op type.

Measured, never inferred: everything printed is read from graph.json. An op
type appearing here is a place the tracer must propagate through; the count per
op type is the order of work.

    python tools/where_the_symbol_chain_breaks.py [--cache DIR] [--models a,b] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CACHE = Path(os.environ.get("NEUROBRIX_CACHE_DIR") or (Path.home() / ".neurobrix" / "cache"))


def derived(v: int) -> dict:
    """Literal -> the relation to v that would explain it (a shape op's arithmetic)."""
    out = {}
    for name, val in (("v", v), ("v//2", v // 2), ("v//4", v // 4), ("v//8", v // 8),
                      ("v*2", v * 2), ("v*4", v * 4), ("v-1", v - 1), ("v+1", v + 1),
                      ("v+2", v + 2), ("v*v", v * v)):
        if val > 1 and val not in out:
            out[val] = name
    return out


def arg_facts(attrs) -> tuple:
    """(symbol ids referenced, literal ints recorded) of one op's arguments."""
    syms, lits = set(), []

    def walk(node):
        if isinstance(node, dict):
            t = node.get("type")
            if t == "symbol" and isinstance(node.get("id"), str):
                syms.add(node["id"])
                return
            if t == "expression":
                for f in (node.get("factors") or []) + (node.get("terms") or []):
                    if isinstance(f, str) and f.startswith("s") and f[1:].isdigit():
                        syms.add(f)
                return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            if node.startswith("s") and node[1:].isdigit():
                syms.add(node)
        elif isinstance(node, bool):
            pass
        elif isinstance(node, int):
            lits.append(node)

    walk(attrs)
    return syms, lits


def ops_in_order(g: dict) -> list:
    ops = g.get("ops") or {}
    order = g.get("execution_order") or list(ops)
    return [(uid, ops[uid]) for uid in order if uid in ops]


def analyse(graph_path: Path) -> list:
    """One row per (symbol, break) found in this graph — and one per symbol that
    is declared and never carried at all (the extreme case: it breaks at birth)."""
    try:
        g = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [{"graph": str(graph_path), "error": f"{type(exc).__name__}: {exc}"}]
    syms = (g.get("symbolic_context") or {}).get("symbols") or {}
    if not syms:
        return []
    tensors = g.get("tensors") or {}
    # Every extent a WEIGHT of this graph carries. A literal that is one of them
    # is most often an architecture constant (a head dim, a patch size, a channel
    # count) that merely coincides with an arithmetic of the symbol — Flex's
    # `view [1,512,4096] -> [1,512,64,64]` records 64 because 4096 = 64x64 heads,
    # not because 512//8 = 64. The flag does not remove the row: real-esrgan's
    # height 64 IS the channel count too, and that collision is the defect
    # itself. It says which rows are defensible without reading them one by one.
    param_extents = set()
    for t in tensors.values():
        if t.get("is_parameter"):
            for d in (t.get("shape") or []):
                if isinstance(d, int):
                    param_extents.add(d)
    ordered = ops_in_order(g)
    facts = {uid: arg_facts(op.get("attributes")) for uid, op in ordered}
    # A DIMENSION, not a symbol id, is what can be lost. The tracer declares one
    # symbol per INPUT that carries the dimension, so a decoder that takes both
    # `input_ids` and `position_ids` declares `seq_len` twice — and binds its
    # expressions to one of the two. Following the other finds no operation that
    # names it and reports every consumer as a break: TinyLlama's flatten records
    # `s0*s1` in full, and the first run of this census called it a lost symbol
    # twenty times over because it was following `s3`. Aliases are grouped by
    # name AND trace value, and a carrier of any member carries the dimension.
    aliases = {}
    for sid, meta in syms.items():
        aliases.setdefault((meta.get("name"), meta.get("trace_value")), []).append(sid)
    alias_of = {sid: set(group) for group in aliases.values() for sid in group}
    rows = []
    seen_dims = set()
    for sid, meta in syms.items():
        src = str(meta.get("source") or "")
        v = meta.get("trace_value")
        if not isinstance(v, int) or v <= 1:
            continue                      # a trace value of 0 or 1 explains nothing
        dim_key = (meta.get("name"), v)
        if dim_key in seen_dims:
            continue                      # one row per DIMENSION, not per declaration
        seen_dims.add(dim_key)
        group = alias_of.get(sid, {sid})
        tainted = set()
        for member in group:
            msrc = str((syms.get(member) or {}).get("source") or "")
            if "::" in msrc:
                tid = msrc.split("::dim_")[0]
                if tid in tensors:
                    tainted.add(tid)
        if not tainted:                   # no input tensor to start from
            tainted = {t for t, d in tensors.items() if t.startswith("input::")}
        rel = derived(int(v))
        # "Never carried" is judged over the WHOLE graph, not along the taint.
        # An operation can carry a dimension without consuming any tensor that
        # holds it: `arange(s3)`, `full`, `new_zeros`, `expand` of a scalar all
        # take the extent as an ARGUMENT. Following only the tensor flow reported
        # GLM-4.1V's four `grid_thw` dimensions as never carried while 87 of that
        # graph's operations reference them by id (2026-09-16). The taint still
        # locates the BREAK — for that, the question is which consumer dropped
        # the expression — but it may not decide whether the dimension lives.
        references_anywhere = sum(1 for uid, _ in ordered if facts.get(uid, (set(), []))[0] & group)
        carriers = 0
        last_carrier = None
        first_break = None
        for uid, op in ordered:
            ins = set(op.get("input_tensor_ids") or [])
            outs = list(op.get("output_tensor_ids") or [])
            if not (ins & tainted):
                continue
            s_here, lits = facts.get(uid, (set(), []))
            if s_here & group:
                carriers += 1
                last_carrier = uid
            elif first_break is None:
                out_dims = {d for shp in (op.get("output_shapes") or [])
                            if isinstance(shp, list) for d in shp if isinstance(d, int)}
                # A literal that was ALREADY an extent of the input is not an
                # expression this operator computed — it preserved a dimension
                # that was there. `view [140,33,32] -> [2,70,33,32]` records 33
                # because 33 is the input's own second axis, and reading it as
                # "height+1" because height traced at 32 is an arithmetic
                # coincidence; the same shape removes mochi's `time+2` (12 is in
                # the input) and orpheus's identity view. Conservative in one
                # direction, and it is named: a genuine break whose literal
                # happens to equal an input extent is hidden by this condition.
                in_dims = {d for shp in (op.get("input_shapes") or [])
                           if isinstance(shp, list) for d in shp if isinstance(d, int)}
                hit = [(l, rel[l]) for l in lits
                       if l in rel and l >= 4 and l in out_dims and l not in in_dims]
                if hit:
                    first_break = {"op_uid": uid, "op_type": op.get("op_type"),
                                   "literal": hit[0][0], "relation": hit[0][1],
                                   "literal_is_a_parameter_extent": hit[0][0] in param_extents,
                                   "input_shapes": op.get("input_shapes"),
                                   "output_shapes": op.get("output_shapes")}
            tainted.update(outs)
        rows.append({"graph": str(graph_path), "symbol": sid, "name": meta.get("name"),
                     "trace_value": v, "source": src, "aliases": sorted(group), "carriers": carriers,
                     "last_carrier": last_carrier, "first_break": first_break,
                     "references_anywhere": references_anywhere,
                     "never_carried": references_anywhere == 0})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--models", default=None, help="comma-separated; default every container")
    ap.add_argument("--json", dest="out", default=None)
    a = ap.parse_args()
    cache = Path(a.cache)
    names = a.models.split(",") if a.models else sorted(
        p.name for p in cache.iterdir() if (p / "manifest.json").exists())
    rows, graphs = [], 0
    for name in names:
        for gp in sorted((cache / name / "components").glob("*/graph.json")):
            graphs += 1
            for r in analyse(gp):
                r["model"] = name
                r["component"] = gp.parent.name
                rows.append(r)
    by_op, comps_by_op, never = {}, {}, []
    for r in rows:
        if r.get("error"):
            continue
        if r["never_carried"]:
            never.append(r)
        b = r.get("first_break")
        if b:
            t = b["op_type"]
            by_op[t] = by_op.get(t, 0) + 1
            comps_by_op.setdefault(t, set()).add(f"{r['model']}/{r['component']}")
    print(f"graphs read                     : {graphs}")
    print(f"declared input symbols followed : {sum(1 for r in rows if not r.get('error'))}")
    print(f"symbols whose chain BREAKS      : {sum(by_op.values())}")
    print("  (a break = an op that consumed the symbol's tensor, referenced no symbol,")
    print("   and wrote a literal derived from its trace value INTO ITS OUTPUT SHAPE)")
    print(f"symbols NEVER carried by any op : {len(never)}")
    print()
    clean = [r for r in rows if (r.get("first_break") or {}).get("relation") == "v"
             and not r["first_break"].get("literal_is_a_parameter_extent")]
    amb = [r for r in rows if r.get("first_break") and
           (r["first_break"]["relation"] != "v" or r["first_break"].get("literal_is_a_parameter_extent"))]
    print(f"  of those, the literal EQUALS the trace value and is not a weight extent: {len(clean)}")
    print(f"  the rest ({len(amb)}) carry a derived relation or a literal that is also a weight")
    print("  extent — an architecture constant can coincide with an arithmetic of the symbol,")
    print("  so those rows are read one by one, never counted as a defect.")
    print()
    print("by the operator that broke the chain (count, components touched):")
    for t, n in sorted(by_op.items(), key=lambda kv: -kv[1]):
        c = sum(1 for r in clean if r["first_break"]["op_type"] == t)
        print(f"  {t:36s} {n:4d}   {len(comps_by_op[t])} component(s)   of which defensible: {c}")
    if never:
        print()
        print("declared and never carried (the extreme case):")
        for r in never[:20]:
            print(f"  {r['model']}/{r['component']}  {r['symbol']} ({r['name']}={r['trace_value']}) from {r['source']}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
        print(f"\nrows written: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
