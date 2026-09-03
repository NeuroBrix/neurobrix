#!/usr/bin/env python3
"""Cross-engine per-op differential on the NBX_DUMP_TIDS JSONL dumps, on the
LAST-POSITION window (`last_pos10`) — the field the head10 window cannot
see when a divergence is position-local (D-TSEQ-ORPHEUS-STEP110: the
collapse lives in the last logits row at context 128).

  python3 tools/dump_diff_lastpos.py A.jsonl B.jsonl [--rel 0.02] [--top 12]

Walks the two dumps in A's op order (matched by tid), prints the first op
whose last-position window deviates beyond `--rel` (relative to the
window's own scale) and the `--top` largest deviations.
"""
import argparse
import json
import math


def load(path):
    recs = {}
    order = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            r = d.get("record", d)
            tid = r.get("tid")
            if tid is None:
                continue
            # Keys carry the component: op uids restart per component
            # (lm_head's mm::0 must not shadow model's mm::0).
            key = (r.get("component", "?"), tid)
            if key in recs:
                continue
            recs[key] = r
            order.append(key)
    return recs, order


def dev(a, b):
    if not a or not b or len(a) != len(b):
        return None
    scale = max(max(abs(x) for x in a), max(abs(x) for x in b), 1e-12)
    return max(abs(x - y) for x, y in zip(a, b)) / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rel", type=float, default=0.02)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--field", default="last_pos10")
    args = ap.parse_args()
    A, order = load(args.a)
    B, _ = load(args.b)
    rows = []
    missing = 0
    for tid in order:
        ra, rb = A[tid], B.get(tid)
        if rb is None:
            missing += 1
            continue
        d = dev(ra.get(args.field), rb.get(args.field))
        if d is None:
            continue
        rows.append((f"{tid[0]}/{tid[1]}", ra.get("op_type"), ra.get("shape"), d))
    print(f"ops in A {len(order)}, matched in B {len(rows)} (B missing {missing}); field={args.field} rel={args.rel}")
    first = next((r for r in rows if r[3] > args.rel), None)
    if first:
        name, ot, shp, d = first
        comp, tid = name.split("/", 1)
        print(f"FIRST op over the bound: {name} ({ot}) shape={shp} rel_dev={d:.4f}")
        print(f"   A {args.field}: {[round(v, 5) for v in A[(comp, tid)].get(args.field)]}")
        print(f"   B {args.field}: {[round(v, 5) for v in B[(comp, tid)].get(args.field)]}")
    else:
        print("no op over the bound")
    print(f"top {args.top} deviations:")
    for tid, ot, shp, d in sorted(rows, key=lambda r: -r[3])[:args.top]:
        print(f"   {d:8.4f}  {tid}  ({ot}) {shp}")


if __name__ == "__main__":
    main()
