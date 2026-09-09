#!/usr/bin/env python3
"""Fold the MoE length sweep and the vendor run into one verdict table.

Ours comes from `moe_real_path_check.py --skip-vendor` (the CLI frames the text
as "Generated N tokens" followed by the generation); theirs comes from the same
prompts sent to the ORIGINAL model on ollama, greedy, same seed.

The comparison is deliberately NOT a byte gate. The vendor here is Q4_0 and we
run fp16, so a difference in wording is the quantization talking, not a defect.
What a frozen-routing defect would produce is categorical: coherence at the
trace length alone. The table therefore reports, per length, the agreement with
the vendor AND whether the length is singled out.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re


def engine_text(raw: str) -> str:
    """The CLI prints 'Generated <n> tokens' then a blank line then the text."""
    m = re.search(r"Generated\s+\d+\s+tokens\s*\n", raw)
    if m:
        return raw[m.end():].strip()
    for marker in ("Generated text:", "=== OUTPUT ===", "Output:"):
        if marker in raw:
            return raw.split(marker)[-1].strip()
    return raw.strip()


def words(t: str):
    return re.findall(r"[A-Za-z0-9_]+", (t or "").lower())


def common_prefix(a: str, b: str) -> int:
    wa, wb = words(a), words(b)
    n = 0
    for x, y in zip(wa, wb):
        if x != y:
            break
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="moe_routing_<date> directory")
    ap.add_argument("--trace-len", type=int, default=23)
    args = ap.parse_args()

    vendor = {r["rendered_len"]: r for r in
              json.load(open(os.path.join(args.dir, "vendor_ollama.json")))}
    rows = []
    for f in sorted(glob.glob(os.path.join(args.dir, "text", "row_len*_triton.json"))):
        r = json.load(open(f))
        ours = engine_text(r["engine_text"])
        theirs = (vendor.get(r["rendered_len"], {}) or {}).get("vendor_text", "")
        rows.append({
            "len": r["rendered_len"], "rc": r["engine_rc"],
            "seconds": r["engine_seconds"],
            "ours": ours, "theirs": (theirs or "").strip(),
            "identical": ours == (theirs or "").strip(),
            "common_prefix_words": common_prefix(ours, theirs or ""),
            "our_words": len(words(ours)),
        })
    rows.sort(key=lambda r: r["len"])

    print(f"{'len':>4} {'rc':>3} {'ident':>6} {'prefix':>7} {'words':>6}  text (ours)")
    print("-" * 108)
    for r in rows:
        mark = " <= TRACE LENGTH" if r["len"] == args.trace_len else ""
        print(f"{r['len']:>4} {r['rc']:>3} {str(r['identical']):>6} "
              f"{r['common_prefix_words']:>7} {r['our_words']:>6}  "
              f"{r['ours'][:64]!r}{mark}")

    ok = [r for r in rows if r["rc"] == 0]
    ident = [r for r in ok if r["identical"]]
    at_trace = [r for r in ok if r["len"] == args.trace_len]
    off_trace = [r for r in ok if r["len"] != args.trace_len]
    print()
    print(f"rows run            : {len(ok)}/{len(rows)}")
    print(f"identical to vendor : {len(ident)}/{len(ok)}")
    if off_trace:
        mean_off = sum(r['common_prefix_words'] for r in off_trace) / len(off_trace)
        print(f"mean vendor prefix  : off-trace {mean_off:.1f} words"
              + (f" · at trace {at_trace[0]['common_prefix_words']} words"
                 if at_trace else ""))
    # The discriminator: a frozen-routing defect is coherent at the trace length
    # ALONE. If the trace length is not distinguished, the defect is not there.
    if at_trace and off_trace:
        singled_out = all(r["common_prefix_words"] < at_trace[0]["common_prefix_words"] / 2
                          for r in off_trace)
        print(f"trace length singled out: {singled_out} "
              f"(True would mean the routing is frozen)")

    out = os.path.join(args.dir, "verdict_table.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
