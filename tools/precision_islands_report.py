#!/usr/bin/env python3
"""Group the engine's own fp32-island decision (NBX_PRECISION_ISLANDS dump).

Run any request with ``NBX_PRECISION_ISLANDS=<tsv>`` set: at load, the
precision contract appends one line per op it pins fp32 (component, op_uid,
op_type, parent_module, calibrated magnitude). This tool groups that dump by
component, op type and module suffix — the activation proof of a calibration
record, as the runtime decided it, not as a re-derivation would.

    NBX_PRECISION_ISLANDS=/path/islands.tsv neurobrix run --model M ...
    python tools/precision_islands_report.py /path/islands.tsv [--top 12]
"""
import argparse
import collections
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsv")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    rows = collections.defaultdict(list)
    with open(args.tsv) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            comp, uid, op_type, module, mag = parts
            rows[comp].append((uid, op_type, module, float(mag)))
    for comp, lst in rows.items():
        print(f"\n## {comp}: {len(lst)} op(s) islanded fp32")
        groups = collections.defaultdict(list)
        for uid, op_type, module, mag in lst:
            parts = module.split(".")
            suffix = ".".join(parts[2:]) if len(parts) > 2 and parts[1].isdigit() else module
            groups[(op_type, suffix)].append((uid, mag, module))
        ordered = sorted(groups.items(), key=lambda kv: -max((v for _, v, _ in kv[1] if v == v), default=0.0))
        for (op_type, suffix), g in ordered[:args.top]:
            blocks = sorted({m.split(".")[1] for _, _, m in g if m.count(".") >= 1 and m.split(".")[1].isdigit()}, key=int)
            vmax = max((v for _, v, _ in g if v == v), default=float("nan"))
            print(f"   {op_type:<32} {suffix or '?':<34} n={len(g):>3} max={vmax:.4g} "
                  f"blocks={blocks[:8]}{'…' if len(blocks) > 8 else ''}")
        if len(ordered) > args.top:
            print(f"   … {len(ordered) - args.top} more group(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
