"""Compare two NBX_DUMP_TIDS JSON dumps and identify the first
divergent op in execution order.

Usage:
    python scripts/diff_tiled_ops.py REF_JSON CMP_JSON [GRAPH_JSON]

Prints a sorted table of divergent ops and identifies the FIRST
divergent op by graph execution_order (when GRAPH_JSON is provided).

Divergence flags:
  - max abs head delta > 0.5
  - cosine similarity < 0.99 (computed from head10)
  - L2 norm relative > 0.10
  - NaN in cmp
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _load(path):
    with open(path) as f:
        d = json.load(f)
    out = {}
    for r in d.get("records", []):
        key = (r["op_uid"], tuple(r["shape"]))
        out[key] = r
    return out


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _has_nan(values):
    return any(v != v for v in values)


def diff(ref_path, cmp_path, graph_path=None):
    ref = _load(ref_path)
    cmp = _load(cmp_path)

    common = sorted(set(ref) & set(cmp))
    print(f"ref records: {len(ref)}")
    print(f"cmp records: {len(cmp)}")
    print(f"common keys: {len(common)}")

    exec_order_map = {}
    if graph_path and Path(graph_path).exists():
        g = json.load(open(graph_path))
        for i, uid in enumerate(g.get("execution_order", [])):
            exec_order_map[uid] = i

    divergent = []
    for key in common:
        r, c = ref[key], cmp[key]
        rl, cl = r["l2_norm"], c["l2_norm"]
        rh, ch = r["head10"], c["head10"]
        # NaN check first
        cmp_nan = _has_nan(ch) or (cl != cl)
        ref_nan = _has_nan(rh) or (rl != rl)
        if cmp_nan and not ref_nan:
            divergent.append((key, "NAN_IN_CMP", rl, cl, 1.0, 0.0))
            continue
        # cosine on head10
        cos = _cosine(rh, ch)
        l2_rel = abs(rl - cl) / max(abs(rl), 1e-9)
        head_max_delta = max(abs(rv - cv) for rv, cv in zip(rh, ch))
        if cos < 0.99 or l2_rel > 0.1 or head_max_delta > 0.5:
            flag = f"cos={cos:.4f} l2rel={l2_rel:.4f} head_dmax={head_max_delta:.3f}"
            divergent.append((key, flag, rl, cl, l2_rel, cos))

    # Sort by exec_order if available
    def _exec_idx(key):
        return exec_order_map.get(key[0], 10**9)

    divergent.sort(key=lambda x: _exec_idx(x[0]))

    print(f"\nDIVERGENT ops: {len(divergent)} / {len(common)}")
    print()
    if not divergent:
        print("No divergence detected.")
        return

    print(f"FIRST divergent op (by exec_order): {divergent[0][0][0]}")
    print(f"  shape={divergent[0][0][1]}")
    print(f"  flag: {divergent[0][1]}")
    print(f"  ref l2={divergent[0][2]:.4f} cmp l2={divergent[0][3]:.4f}")
    print(f"  ref head: {ref[divergent[0][0]]['head10'][:5]}")
    print(f"  cmp head: {cmp[divergent[0][0]]['head10'][:5]}")
    print()
    print("Top 20 divergent (exec order):")
    print(f"{'idx':>5}  {'op_uid':<30} {'flag':<40}")
    for entry in divergent[:20]:
        key, flag, rl, cl, l2r, cos = entry
        idx = _exec_idx(key)
        print(f"{idx:>5}  {key[0]:<30} {flag:<40}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    diff(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else None)
