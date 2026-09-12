#!/usr/bin/env python3
"""Which components hold a TEMPORAL LOOP THE TRACER UNROLLED?

WHY THIS IS ASKED
-----------------
A causal video VAE processes frames in chunks. If the tracer unrolls that loop,
the graph holds one copy of the chunk body per chunk and the chunk count is fixed
at the trace stimulus. Such a component is NOT symbolic in time however many
symbols its table declares: a graph unrolled for five chunks cannot process
eighty-one frames, and the symbolic machinery then folds the temporal factor into
a neighbouring slot rather than failing honestly.

Measured on `Wan2.1-VACE-1.3B-diffusers/vae_encoder` (2026-09-12): 1448 ops at
T=9, 2271 at T=17, 3305 at T=25, and every module group exactly linear in the
chunk count k = (T-1)//4+1 — 3 chunks to 5 moves the four groups 8 -> 14,
21 -> 35, 6 -> 10, 3 -> 5, fitting 3k-1, 7k, 2k, k.

TWO KINDS OF LINE, AND THEY DO NOT READ THE SAME
-------------------------------------------------
  MEASURED   two trace points exist for this component, so the dependence of the
             op count on the stimulus is fitted, not guessed.
  INFERRED   one trace point. The signature is the module-repetition histogram
             agreeing with this graph's own k. It is evidence, not a fit, and a
             single graph cannot distinguish "r modules because k chunks" from
             "r modules because the architecture has r of them".

A census that prints both in one column is a census that has already lost the
distinction. This one refuses to.

LIVE OR THEORETICAL
-------------------
An unrolled component is only a LIVE limit if the request asks for a chunk count
other than the traced one. The container states its own default frame count, so
the comparison is available without running anything: traced k against runtime k.

Usage:
    python tools/temporal_unroll_census.py
    python tools/temporal_unroll_census.py --json
    python tools/temporal_unroll_census.py --second-point DIR   # measured lines
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

#: Symbol names that denote a temporal axis. Read from the graph's DECLARED
#: table, never from position: the positional base calls s1 a latent height on
#: every graph, and a census built on it was retracted on 2026-09-12.
TEMPORAL_NAMES = {"time", "num_frames", "frames", "temporal", "frame"}

#: Fallback temporal compression when the container states none. 4 is the causal
#: video VAE convention (4k+1 frames -> k+1 latent frames); it is READ from
#: `runtime/defaults.json` whenever the container carries it.
DEFAULT_COMPRESSION = 4


def chunks(extent: int, compression: int) -> int:
    """Latent/chunk count for a temporal extent, causal convention."""
    if extent is None or compression < 1:
        return 0
    return (extent - 1) // compression + 1


def module_histogram(graph: dict) -> collections.Counter:
    """{repetition count: how many distinct parent modules repeat that often}."""
    per = collections.Counter()
    for op in (graph.get("ops") or {}).values():
        parent = op.get("parent_module")
        if parent:
            per[parent] += 1
    return collections.Counter(per.values())


def temporal_symbols(graph: dict):
    syms = (graph.get("symbolic_context") or {}).get("symbols") or {}
    return [(sid, m.get("trace_value")) for sid, m in syms.items()
            if isinstance(m, dict) and (m.get("name") or "").lower() in TEMPORAL_NAMES]


def signature(graph: dict, k: int) -> dict:
    """How much of the graph sits in module groups explained by k chunks.

    A group of `n` modules each repeated `r` times is "explained" when r is a
    small multiple of k, or one short of one (the `3k-1` form the measured case
    shows, which is k chunks minus a boundary that has no predecessor -- exactly
    what a CAUSAL chunk loop produces at its first chunk).
    """
    hist = module_histogram(graph)
    if k < 2 or not hist:
        return {"share": 0.0, "groups": {}, "explained": {}}
    total = sum(n * r for r, n in hist.items())
    explained, share = {}, 0
    for r, n in hist.items():
        m = r % k
        if m == 0 or m == k - 1:            # r = jk  or  r = jk - 1
            explained[r] = n
            share += n * r
    return {"share": share / total if total else 0.0,
            "groups": dict(sorted(hist.items(), reverse=True)[:6]),
            "explained": explained}


def defaults_for(model_dir: Path) -> dict:
    f = model_dir / "runtime" / "defaults.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return {}


def census(root: Path, second_point: Path | None):
    rows = []
    for path in sorted(root.glob("*/components/*/graph.json")):
        model_dir = path.parents[2]
        model, comp = model_dir.name, path.parent.name
        try:
            graph = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            rows.append({"component": f"{model}/{comp}", "error": str(exc)})
            continue
        temporal = temporal_symbols(graph)
        if not temporal:
            continue
        defaults = defaults_for(model_dir)
        compression = int(defaults.get("temporal_compression_ratio")
                          or DEFAULT_COMPRESSION)
        traced = temporal[0][1]
        k_trace = chunks(traced, compression)
        run_frames = defaults.get("num_frames")
        k_run = chunks(run_frames, compression) if isinstance(run_frames, int) else None
        sig = signature(graph, k_trace)

        # A SECOND TRACE POINT turns the line from inferred to measured.
        row_kind, fit = "INFERRED", None
        if second_point:
            other = second_point / model / comp / "graph.json"
            if other.exists():
                try:
                    g2 = json.loads(other.read_text())
                except (OSError, ValueError):
                    g2 = None
                if g2:
                    t2 = temporal_symbols(g2)
                    if t2 and t2[0][1] != traced:
                        k2 = chunks(t2[0][1], compression)
                        n1, n2 = len(graph.get("ops") or {}), len(g2.get("ops") or {})
                        if k2 != k_trace:
                            slope = (n2 - n1) / (k2 - k_trace)
                            row_kind = "MEASURED"
                            fit = {"points": [(traced, n1), (t2[0][1], n2)],
                                   "ops_per_chunk": slope}
        rows.append({
            "component": f"{model}/{comp}", "kind": row_kind,
            "traced_extent": traced, "compression": compression,
            "k_trace": k_trace, "k_runtime": k_run, "runtime_frames": run_frames,
            "ops": len(graph.get("ops") or {}),
            "share_explained": sig["share"], "groups": sig["groups"],
            "explained": sig["explained"], "fit": fit,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.environ.get(
        "NEUROBRIX_CACHE", str(Path.home() / ".neurobrix" / "cache")))
    ap.add_argument("--second-point", default=None,
                    help="a second trace tree (e.g. .cache/graphs) for MEASURED lines")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="share of ops in k-explained module groups to call it unrolled")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    second = Path(args.second_point) if args.second_point else None
    rows = census(root, second)
    if not rows:
        print("REFUSED: no component declares a temporal symbol under "
              f"{root}. A census over zero components is not a clean census.",
              file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rows, indent=1))
        return 0

    suspect = [r for r in rows if r.get("share_explained", 0) >= args.threshold
               and r.get("k_trace", 0) >= 2]
    print(f"Temporal-unroll census — {len(rows)} component(s) declaring a temporal "
          f"axis, under {root}\n")
    print(f"{'component':52s} {'kind':9s} {'T':>4s} {'k_tr':>5s} {'k_run':>6s} "
          f"{'ops':>6s} {'share':>6s}  verdict")
    print("-" * 118)
    for r in sorted(rows, key=lambda r: (-r.get("share_explained", 0), r["component"])):
        if "error" in r:
            print(f"{r['component']:52s} UNREADABLE — {r['error'][:40]}")
            continue
        unrolled = r["share_explained"] >= args.threshold and r["k_trace"] >= 2
        live = (unrolled and r["k_runtime"] is not None
                and r["k_runtime"] != r["k_trace"])
        verdict = ("unrolled, LIVE" if live else
                   "unrolled, theoretical" if unrolled else "—")
        print(f"{r['component']:52s} {r['kind']:9s} {r['traced_extent']:>4} "
              f"{r['k_trace']:>5} {str(r['k_runtime']):>6} {r['ops']:>6} "
              f"{r['share_explained']:>6.2f}  {verdict}")

    measured = [r for r in rows if r.get("kind") == "MEASURED"]
    print(f"\n  MEASURED lines (two trace points, slope fitted): {len(measured)}")
    for r in measured:
        f = r["fit"]
        print(f"    {r['component']}: {f['points']} -> {f['ops_per_chunk']:.0f} "
              f"ops per chunk")
    print(f"  INFERRED lines (one trace point, histogram signature): "
          f"{len(rows) - len(measured)}")
    live = [r for r in suspect if r["k_runtime"] is not None
            and r["k_runtime"] != r["k_trace"]]
    print(f"\n  carrying the signature : {len(suspect)} of {len(rows)}")
    print(f"  containers affected    : "
          f"{len({r['component'].split('/')[0] for r in suspect})}")
    print(f"  LIVE (runtime chunk count differs from the traced one): {len(live)}")
    for r in live:
        print(f"    {r['component']}: traced {r['k_trace']} chunks, runtime "
              f"{r['k_runtime']} ({r['runtime_frames']} frames)")
    print("\n  An INFERRED line is evidence, not a fit: one graph cannot separate "
          "'r copies\n  because k chunks' from 'r copies because the architecture "
          "has r of them'.\n  Converting one costs a second trace, nothing more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
