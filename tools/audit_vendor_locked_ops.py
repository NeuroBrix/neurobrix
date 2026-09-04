#!/usr/bin/env python3
"""Which graphs in the catalogue cannot leave NVIDIA — an executable census.

R19 says `graph.json` carries EXCLUSIVELY essential ATen ops, and R23 says every
solution is designed from the start for V100/A100/H100/AMD CDNA. An operator
whose NAME binds it to one vendor's library breaks both at once: it is not a
neutral ATen identity, and it cannot be lowered anywhere else. Such a graph
runs on NVIDIA and refuses everywhere — CPU, AMD, Apple — with
`NotImplementedError: Could not run 'aten::X' with arguments from the 'CPU'
backend`.

Found 2026-09-04 on Kokoro-82M: `aten::cudnn_batch_norm` in the decoder graph.
It surfaced only because a single-card placement put that component on the
host; on three cards the component stays on CUDA and the graph looks fine. The
battery, pinned to three cards, cannot see it. So a name census is the right
instrument: it does not depend on which placement a run happens to take.

The fix for anything found here is BUILD-side, in the tracer, and the artifact
is re-traced. Runtime compensation of a build-side limit is rejected —
mapping `cudnn_batch_norm` to `native_batch_norm` in the engine would hide a
graph that is not portable behind an engine that pretends it is.

    python3 tools/audit_vendor_locked_ops.py [--cache DIR] [--json OUT]

Exit code 1 if any vendor-locked operator is present, so this is usable as a
gate on the catalogue.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

# Operator name fragments that name a vendor's library rather than a
# mathematical operation. Each maps to the vendor it locks the graph to.
VENDOR_TOKENS = {
    "cudnn": "NVIDIA (cuDNN)",
    "cublas": "NVIDIA (cuBLAS)",
    "cufft": "NVIDIA (cuFFT)",
    "cusparse": "NVIDIA (cuSPARSE)",
    "nvrtc": "NVIDIA",
    "miopen": "AMD (MIOpen)",
    "mkldnn": "Intel (oneDNN)",
    "onednn": "Intel (oneDNN)",
    "nnpack": "Intel/ARM (NNPACK)",
    "xnnpack": "XNNPACK",
    "mps_": "Apple (MPS)",
}

# Ops that carry a vendor name but are dispatched generically by PyTorch and
# lower on every backend. Kept explicit so the census cannot quietly widen.
BENIGN: set[str] = set()


def _ops(graph: dict) -> list[str]:
    """Every op_type in a graph.

    `ops` is a DICT keyed by op_uid ("aten.unsqueeze::0"), not a list. An
    earlier version of this reader assumed a list, found nothing, and reported
    a clean catalogue — a census that scans zero operators will always say
    everything is fine. `_ops` is therefore checked by
    `test_the_reader_actually_finds_operators`, and this tool refuses to print
    a verdict when it has read no operators at all.
    """
    ops = graph.get("ops") if isinstance(graph, dict) else graph
    if isinstance(ops, dict):
        ops = list(ops.values())
    if not isinstance(ops, list):
        return []
    out = []
    for o in ops:
        if isinstance(o, dict):
            name = o.get("op_type") or o.get("type") or o.get("op") or ""
            if name:
                out.append(str(name))
        elif isinstance(o, str):
            out.append(o)
    return out


def _lock(op: str) -> str | None:
    low = op.lower()
    if op in BENIGN:
        return None
    for token, vendor in VENDOR_TOKENS.items():
        if token in low:
            return vendor
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(Path.home() / ".neurobrix" / "cache"))
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    cache = Path(args.cache)
    if not cache.is_dir():
        print(f"no cache at {cache}", file=sys.stderr)
        return 2

    findings: dict[str, dict[str, collections.Counter]] = {}
    scanned_models = scanned_graphs = total_ops = 0

    for model_dir in sorted(p for p in cache.iterdir() if p.is_dir()):
        graphs = sorted(model_dir.rglob("graph.json"))
        if not graphs:
            continue
        scanned_models += 1
        for g in graphs:
            try:
                ops = _ops(json.loads(g.read_text()))
            except Exception:                                  # noqa: BLE001
                continue
            scanned_graphs += 1
            total_ops += len(ops)
            for op in ops:
                vendor = _lock(op)
                if vendor:
                    comp = g.parent.name
                    findings.setdefault(model_dir.name, {}) \
                            .setdefault(comp, collections.Counter())[f"{op}  [{vendor}]"] += 1

    print("=" * 78)
    print("VENDOR-LOCKED OPERATOR CENSUS")
    print("=" * 78)
    print(f"models scanned : {scanned_models}")
    print(f"graphs scanned : {scanned_graphs}")
    print(f"operators seen : {total_ops:,}")
    print()

    if total_ops == 0:
        print("REFUSING A VERDICT: zero operators were read.")
        print("A census that scans nothing reports everything as clean. The")
        print("graph schema has changed or the cache path is wrong.")
        return 2

    if not findings:
        print("No vendor-locked operator in any catalogue graph.")
        print("Every graph is a neutral ATen identity and can lower on any backend.")
        return 0

    print(f"{len(findings)} model(s) carry an operator that cannot leave its vendor:\n")
    for model in sorted(findings):
        print(f"  {model}")
        for comp in sorted(findings[model]):
            for op, n in findings[model][comp].most_common():
                print(f"      {comp}/graph.json   {op}  x{n}")
    print()
    print("Each of these is a BUILD-side defect (R19 pure ATen identity, R23")
    print("hardware universality). Fix in the tracer and re-trace the artifact;")
    print("do NOT map it to a generic op in the engine — that hides a graph that")
    print("is not portable behind an engine pretending it is.")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            {m: {c: dict(v) for c, v in comps.items()} for m, comps in findings.items()},
            indent=2))
        print(f"\nwritten: {args.json_out}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
