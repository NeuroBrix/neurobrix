#!/usr/bin/env python3
"""Render the temporal-unroll census from its measurements, never from memory.

Two kinds of line, and they do not read the same:

  MEASURED   two trace points from the SAME tracer, slope fitted. A fit across
             two tracer versions is not a fit, which is why the cached graphs
             are not used as a first point.
  INFERRED   one trace point. The claim rests on the chunk-loop module groups
             matching a measured component's, group for group, at the same
             chunk count -- not on the graphs "looking alike".

Reads the campaign's own JSON so a number cannot be retyped wrong.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

MEASURED = Path("validation_outputs/unroll_census_20260912/measured.json")
CACHE = Path.home() / ".neurobrix" / "cache"


def chunk_groups(graph_path: Path, k: int) -> dict:
    """{repetitions: modules} for the groups a chunk loop would own at k.

    Restricted to repetition counts a chunk loop produces -- jk and jk-1 for
    small j -- because the groups ABOVE them are the model's own size and differ
    between VAE variants that share the same loop.
    """
    g = json.loads(graph_path.read_text())
    per = collections.Counter()
    for op in (g.get("ops") or {}).values():
        m = op.get("parent_module")
        if m:
            per[m] += 1
    hist = collections.Counter(per.values())
    return {r: n for r, n in hist.items()
            if k >= 2 and r <= 8 * k and (r % k == 0 or r % k == k - 1)}


def main() -> int:
    if not MEASURED.exists():
        print(f"REFUSED: {MEASURED} absent. The report is rendered from the "
              f"measurement, never from memory.", file=sys.stderr)
        return 1
    rows = json.loads(MEASURED.read_text())

    print("| component | line | ops @k=3 | ops @k=7 | ops/chunk | verdict |")
    print("|---|---|---:|---:|---:|---|")
    for r in sorted(rows, key=lambda r: r["component"]):
        if "points" not in r:
            print(f"| `{r['component']}` | — | — | — | — | {r['verdict']} |")
            continue
        p = list(r["points"].values())
        print(f"| `{r['component']}` | MEASURED | {p[0]} | {p[1]} | "
              f"{r['ops_per_chunk']:.2f} | **{r['verdict']}** |")

    # The inferred lines, and exactly what they rest on.
    anchor = Path("/home/mlops/nbx/stage/unroll_measure/T9/"
                  "Wan2.1-VACE-1.3B-diffusers/vae_encoder/graph.json")
    if not anchor.exists():
        print("\n(no measured anchor graph at k=3 — inferred lines not rendered)")
        return 0
    ref = chunk_groups(anchor, 3)
    print(f"\nThe measured anchor's chunk-loop groups at k=3 "
          f"(Wan2.1-VACE/vae_encoder): {dict(sorted(ref.items()))}\n")
    print("| component | line | chunk-loop groups at k=3 | identical to the anchor |")
    print("|---|---|---|---|")
    for model in ("Wan2.1-I2V-14B-480P-Diffusers", "Wan2.2-I2V-A14B-Diffusers"):
        g = CACHE / model / "components" / "vae_encoder" / "graph.json"
        if not g.exists():
            continue
        got = chunk_groups(g, 3)
        print(f"| `{model}/vae_encoder` | INFERRED | {dict(sorted(got.items()))} | "
              f"{'yes' if got == ref else 'NO'} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
