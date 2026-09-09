#!/usr/bin/env python3
"""Every op that executed must be observable. A replaced call site must stay observable.

THE RULE
--------
**A brick that replaces a call site preserves the observability of that site.**

An interceptor, a fusion, a proxy, a tiling wrapper — anything that stands in
for an op — inherits the watchers attached to what it replaced. If it cannot
produce what the original produced, it produces the record itself. Silence is
not permitted, because silence is indistinguishable from correctness.

WHY, AND WHY IT IS URGENT
-------------------------
This is the THIRD time in this repository, in three different dresses:

1. the decode replay watched a call site the engine had replaced, and lost
   x9.7 of throughput while staying byte-correct (`feedback_a_replaced_call_site_blinds_every_watcher`);
2. a harness metric read a key its brick never emitted, so an image gate could
   never pass (`tools/harness_metric_key_audit.py`);
3. and here: Prism's op-level tiling fuses an upsample with its conv, the
   upsample interceptor returns a `FusionUpsampleProxy` that computes nothing,
   and the per-op recorder drops it on `if not isinstance(tensor, torch.Tensor):
   return` — the SAME guard that legitimately skips a tuple-returning op. Two of
   the three spatial upsamples of the Wan VAE, the two of highest resolution,
   produced no record at all. We believed we were looking and we were not.

The family is one family: an observer attached to a site that something else now
occupies. It makes the instrumentation lie BY CONSTRUCTION, which is worse than
an instrument that is merely wrong — a wrong number invites a second look, a
missing one does not.

HOW THE CHECK DECIDES, WITHOUT CRYING WOLF
------------------------------------------
Read from the container, never from a hardcoded op list:

    an op with EXACTLY ONE output tensor  -> a record is REQUIRED
    an op with several (or none)          -> the recorder's single-tensor guard
                                             legitimately skips it

That distinction is in `graph.json` itself: `aten.split` declares 3 output
tensors, `aten._scaled_dot_product_efficient_attention` declares 4, and
`aten._upsample_nearest_exact2d` declares 1. So the four "missing" ops that are
fine and the two that are blind separate themselves, from data.

A gap on a single-output op is `BLIND` and fails the gate.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import zipfile
from pathlib import Path


def load_records(dump: str) -> dict:
    """op_uids that produced a record, per component."""
    per = {}
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
        if not isinstance(r, dict):
            continue
        comp, uid = r.get("component"), r.get("op_uid")
        if comp is None or uid is None:
            continue
        per.setdefault(comp, set()).add(uid)
    return per


def component_graphs(nbx: str) -> dict:
    """component -> graph.json, from the container."""
    out = {}
    with zipfile.ZipFile(nbx) as z:
        for n in z.namelist():
            if n.endswith("graph.json") and "/components/" in "/" + n:
                comp = n.split("/")[-2]
                try:
                    out[comp] = json.loads(z.read(n))
                except Exception:
                    pass
    return out


def audit(nbx: str, dump: str) -> list:
    graphs = component_graphs(nbx)
    recorded = load_records(dump)
    rows = []
    for comp, g in sorted(graphs.items()):
        ops = g.get("ops", {})
        order = g.get("execution_order", [])
        seen = recorded.get(comp, set())
        if not seen:
            # the component never ran in this request (a decoder on a text path,
            # a talker that was not asked for) — not a blind spot
            rows.append({"component": comp, "verdict": "NOT-EXERCISED",
                         "ops_in_order": len(order), "records": 0, "blind": []})
            continue
        # Which op TYPES record at all in this component. A type that never
        # records anywhere is handled outside the recorder by construction (a
        # weight transpose the engine folds); a type whose SIBLINGS record but
        # this one does not is a site something replaced. That is the whole
        # discriminator, and it is read from the run, not from a list someone
        # has to maintain.
        types_seen = {ops.get(u, {}).get("op_type") for u in seen}
        blind, skipped, systematic = [], 0, {}
        for uid in order:
            o = ops.get(uid, {})
            n_out = len(o.get("output_tensor_ids", []) or o.get("output_shapes", []))
            if n_out != 1:
                skipped += 1
                continue
            if uid in seen:
                continue
            t = o.get("op_type")
            if t not in types_seen:
                systematic[t] = systematic.get(t, 0) + 1
                continue
            blind.append({"op_uid": uid, "op_type": t,
                          "parent_module": o.get("parent_module"),
                          "output_shape": (o.get("output_shapes") or [None])[0],
                          "siblings_recorded": sum(
                              1 for u in seen if ops.get(u, {}).get("op_type") == t)})
        rows.append({"component": comp,
                     "verdict": "BLIND" if blind else "OBSERVED",
                     "ops_in_order": len(order), "records": len(seen),
                     "expected_skips": skipped,
                     "systematic_classes": systematic, "blind": blind})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container", required=True, help="path to model.nbx")
    ap.add_argument("--dump", required=True, help="NBX_DUMP_TIDS jsonl of a run")
    ap.add_argument("--label", default="", help="what this run was")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = audit(a.container, a.dump)
    blind_total = 0
    for r in rows:
        mark = {"OBSERVED": "✓", "NOT-EXERCISED": "–"}.get(r["verdict"], "✗")
        print(f"  {mark} {r['verdict']:14s} {r['component']:22s} "
              f"{r['records']:5d} record(s) / {r['ops_in_order']:5d} op(s)"
              + (f", {r.get('expected_skips',0)} legitimately skipped"
                 if r["verdict"] != "NOT-EXERCISED" else ""))
        for t, n in sorted((r.get("systematic_classes") or {}).items()):
            print(f"      class  {t}: {n} op(s), and this type records NOWHERE in "
                  f"this component — handled outside the recorder, not a blinded site")
        for b in r["blind"]:
            blind_total += 1
            print(f"      BLIND {b['op_uid']}  {b['op_type']}")
            print(f"            shape {b['output_shape']}  module {b['parent_module']}")
            print(f"            {b['siblings_recorded']} sibling(s) of this type DID "
                  f"record here — this site was replaced, not excluded")
    print(f"\n{blind_total} site(s) BLIND: a single-tensor op with no record while its "
          f"own siblings recorded — a place we believe we are looking and are not")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"label": a.label, "container": a.container, "dump": a.dump,
             "blind_total": blind_total, "components": rows}, indent=1))
        print(f"written: {a.out}")
    return 1 if blind_total else 0


if __name__ == "__main__":
    sys.exit(main())
