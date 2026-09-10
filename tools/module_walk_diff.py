#!/usr/bin/env python3
"""Our transformer against the vendor's, module by module, and the FIRST one that departs.

CHANTIER 2, step two. The step ladder framed the search: our post-denoise latent
differs from the vendor's by 34 % after a SINGLE forward pass, a decaying gain
over a stable ~24 % structural residual present from step one. So the answer is
structural and inside one pass, and this walk is what names where.

HOW THE TWO SIDES ALIGN
-----------------------
No hand-written correspondence table. Our container records `parent_module` per
op — the vendor module path the op was traced from — and `NBX_DUMP_TIDS` writes
`(component, op_uid, op_type, shape, head10, l2_norm)` per op. The vendor side
(`tools/vendor_module_walk.py`) writes one record per module invocation. The two
join on the module name the tracer already wrote down.

Our stand-in for "the module's output" is the LAST recorded op whose
`parent_module` is that module, in execution order. That is a stated heuristic,
not a certainty: a module whose final op is fused away, or whose last op is a
view, will be represented by whatever recorded last. The walk therefore reports
the shape on both sides, and a shape mismatch is called out rather than folded
into the deviation — comparing two different tensors and printing a percentage
is exactly the kind of confident nonsense this repository has been paying for.

THE VERDICT IT GIVES
--------------------
The first module, in execution order, whose relative L2 deviation exceeds the
bound — with every module behind it listed, because the ops after the first are
cascade until proven otherwise (the three-class discipline: root, cascade,
common baseline).
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import zipfile
from pathlib import Path


def read_ours(dump: str, component: str) -> list:
    out = []
    for line in Path(dump).read_text(errors="replace").splitlines():
        try:
            o = json.loads(line)
        except Exception:
            continue
        r = o.get("record", o)
        if isinstance(r, str):
            try:
                r = ast.literal_eval(r)
            except Exception:
                continue
        if isinstance(r, dict) and r.get("component") == component:
            out.append(r)
    return out


def read_theirs(path: str, call: int) -> dict:
    per = {}
    order = []
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("call") != call:
            continue
        m = r.get("module")
        if m is None:
            continue
        if m not in per:
            order.append(m)
        per[m] = r          # last write per module within the call
    return {"per": per, "order": order}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container", required=True)
    ap.add_argument("--component", default="transformer")
    ap.add_argument("--ours", required=True, help="NBX_DUMP_TIDS jsonl")
    ap.add_argument("--theirs", required=True, help="vendor_module_walk jsonl")
    ap.add_argument("--call", type=int, default=0,
                    help="which invocation of the vendor sub-model to compare "
                         "(CFG runs it more than once per step)")
    ap.add_argument("--bound", type=float, default=0.05,
                    help="relative L2 deviation that counts as a departure")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with zipfile.ZipFile(a.container) as z:
        name = f"components/{a.component}/graph.json"
        if name not in z.namelist():
            cand = [n for n in z.namelist()
                    if n.endswith("graph.json") and a.component in n]
            if not cand:
                print(f"no graph for component {a.component!r}", file=sys.stderr)
                return 2
            name = cand[0]
        g = json.loads(z.read(name))
    ops, order = g.get("ops", {}), g.get("execution_order", [])

    ours_recs = {r["op_uid"]: r for r in read_ours(a.ours, a.component)}
    theirs = read_theirs(a.theirs, a.call)

    # our stand-in per module: the LAST recorded op of that module, in order
    ours_by_module, module_order = {}, []
    for uid in order:
        mod = (ops.get(uid) or {}).get("parent_module")
        if not mod or uid not in ours_recs:
            continue
        if mod not in ours_by_module:
            module_order.append(mod)
        ours_by_module[mod] = ours_recs[uid]

    common = [m for m in module_order if m in theirs["per"]]
    print(f"  ours: {len(ours_by_module)} module(s) with a record; "
          f"vendor: {len(theirs['per'])} on call {a.call}; "
          f"{len(common)} join by name\n")
    if not common:
        print("  NOTHING JOINS — the two sides do not share module names. That is "
              "the finding: the walk cannot be aligned, and no deviation printed "
              "here would mean anything.")
        return 2

    rows, first = [], None
    for m in common:
        o, t = ours_by_module[m], theirs["per"][m]
        so, st = list(o.get("shape") or []), list(t.get("shape") or [])
        lo, lt = float(o.get("l2_norm") or 0.0), float(t.get("l2_norm") or 0.0)
        same_shape = so == st
        dev = abs(lo - lt) / lt if (lt and same_shape) else None
        r = {"module": m, "op_uid": o.get("op_uid"), "op_type": o.get("op_type"),
             "ours_shape": so, "theirs_shape": st, "same_shape": same_shape,
             "ours_l2": lo, "theirs_l2": lt, "rel_dev": dev}
        rows.append(r)
        if first is None and same_shape and dev is not None and dev > a.bound:
            first = r

    print(f"  {'module':52s} {'ours l2':>12s} {'vendor l2':>12s} {'dev':>8s}")
    for r in rows[:a.top]:
        d = "shape!=" if not r["same_shape"] else (
            f"{r['rel_dev']*100:7.2f}%" if r["rel_dev"] is not None else "  n/a")
        mark = "<<<" if first is not None and r is first else ""
        print(f"  {r['module'][:52]:52s} {r['ours_l2']:12.3f} "
              f"{r['theirs_l2']:12.3f} {d} {mark}")
    if len(rows) > a.top:
        print(f"  ... {len(rows)-a.top} more")

    mism = [r for r in rows if not r["same_shape"]]
    if mism:
        print(f"\n  {len(mism)} module(s) compared different shapes and were NOT "
              f"scored — e.g. {mism[0]['module']}: ours {mism[0]['ours_shape']} "
              f"vs vendor {mism[0]['theirs_shape']}")
    if first:
        print(f"\n  FIRST DEPARTURE beyond {a.bound*100:.0f}%: {first['module']}")
        print(f"     our op {first['op_uid']} ({first['op_type']}), shape "
              f"{first['ours_shape']}, l2 {first['ours_l2']:.3f} vs "
              f"{first['theirs_l2']:.3f} — {first['rel_dev']*100:.2f}%")
        print("     everything after it is cascade until proven otherwise")
    else:
        print(f"\n  no module departs beyond {a.bound*100:.0f}% on the compared set")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"component": a.component, "call": a.call, "bound": a.bound,
             "first_departure": first, "modules": rows}, indent=1))
        print(f"  written: {a.out}")
    return 0 if first is None else 1


if __name__ == "__main__":
    sys.exit(main())
