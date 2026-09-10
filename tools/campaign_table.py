#!/usr/bin/env python3
"""The certified-directory campaign's table, ordered by the law that governs it.

WHAT THE TABLE IS ORDERED BY, AND WHY IT MATTERS

The gain from the certified directory is a function of **how many shape keys the
model demands**, not of how big the model is. Measured on the five cells closed
by 2026-09-10:

    DeepSeek-Coder-V2-Lite  137 keys  x16.64
    Ming-Lite-Omni-1.5      203 keys  x11.69
    deepseek-moe-16b-chat     9 keys  x2.39
    Qwen3-30B-A3B-Thinking    8 keys  x1.57
    Qwen3-Coder-30B           8 keys  x1.54

A 54 GB multimodal and a 16 GB MoE sit at opposite ends of that list, and the
thing that separates them is the key count. A table sorted by name lets a reader
invent a law about model size; a table sorted by keys shows the one that is
there. So this tool sorts by keys and refuses `--sort name`.

THE MEDIAN, AND THE COMPARISON THAT IS NOT ALLOWED

A median is published **with the population it covers, named model by model**,
or it is not published. The reason is concrete: an earlier campaign reported
x13.9 over a population that included `chatterbox` (674 keys) and `openaudio`
(693) — two of the highest key counts in the catalogue — which this campaign
does not carry. Writing "the median fell" against that number would be exactly
as false as the x13.9 was.

So `--compare <other campaign>` REFUSES unless the two populations are
identical, and names the difference. That is a door, not a warning: a
cross-population comparison cannot be made honest by a footnote.

Usage:
    python tools/campaign_table.py /home/mlops/nbx/campaigns/<dated dir>
    python tools/campaign_table.py <dir> --compare <other dir>
    python tools/campaign_table.py <dir> --markdown
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

#: Everything in this table describes ONE profile. The doctrine sentence is
#: printed with every table rather than left to the reader's memory.
SCOPE = (
    "Everything below concerns `nvidia/volta` and nothing else. On every other "
    "target the runtime sweeps and the guarantee is the consensus screen — where "
    '"screened" means the candidates agreed with each other, never that anything '
    "was proved. See docs/reference/what-certified-means.md."
)


def _cells(campaign: Path) -> list[dict]:
    out = []
    for result in sorted((campaign / "proof").glob("*/result.json")):
        try:
            out.append(json.loads(result.read_text()))
        except (OSError, ValueError) as exc:
            print(f"  ! {result.parent.name}: unreadable ({exc.__class__.__name__})",
                  file=sys.stderr)
    return out


def _median_exec(arm: dict) -> float | None:
    reps = [r.get("exec_s") for r in (arm.get("reps") or []) if r.get("rc") == 0]
    reps = [r for r in reps if isinstance(r, (int, float))]
    if reps:
        return statistics.median(reps)
    return arm.get("exec_s")


def _row(cell: dict) -> dict:
    a, b = cell.get("A") or {}, cell.get("B") or {}
    am, bm = _median_exec(a), _median_exec(b)
    keys = b.get("swept")
    served = a.get("certified_served")

    # The guard from 2026-09-10: an arm that served nothing AND swept nothing
    # carries NO ratio rather than a ratio of one. A cell whose lever did not
    # move is a cell that measured nothing, and 1.0 reads as "no gain".
    ratio = None
    if am and bm and (keys or served):
        ratio = bm / am

    return {
        "model": cell.get("model", "?"),
        "family": cell.get("family", "?"),
        "gb": cell.get("weight_gb"),
        "keys": keys,
        "served": served,
        "a_med": am,
        "b_med": bm,
        "ratio": ratio,
        "identical": (cell.get("gate") or {}).get("identical"),
        "excluded": b.get("screen_excluded"),
        "contradictions": b.get("contradictions"),
        "rc": (a.get("rc"), b.get("rc")),
    }


def _fmt(rows: list[dict], markdown: bool) -> str:
    head = ["model", "family", "GB", "keys", "A med s", "B med s", "gain",
            "bytes", "screen-out", "contra"]
    lines = []
    if markdown:
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "|".join("---" for _ in head) + "|")
    else:
        lines.append(f"{'model':<34} {'family':<12} {'GB':>6} {'keys':>5} "
                     f"{'A med':>9} {'B med':>9} {'gain':>7} {'bytes':>7} "
                     f"{'scr-out':>8} {'contra':>7}")
    for r in rows:
        cells = [
            r["model"],
            r["family"],
            f"{r['gb']:.1f}" if r["gb"] else "?",
            str(r["keys"]) if r["keys"] is not None else "?",
            f"{r['a_med']:.2f}" if r["a_med"] else "?",
            f"{r['b_med']:.2f}" if r["b_med"] else "?",
            f"x{r['ratio']:.2f}" if r["ratio"] else "no ratio",
            {True: "same", False: "DIFFER", None: "?"}[r["identical"]],
            str(r["excluded"]) if r["excluded"] is not None else "?",
            str(r["contradictions"]) if r["contradictions"] is not None else "?",
        ]
        if markdown:
            lines.append("| " + " | ".join(cells) + " |")
        else:
            lines.append(f"{cells[0]:<34} {cells[1]:<12} {cells[2]:>6} {cells[3]:>5} "
                         f"{cells[4]:>9} {cells[5]:>9} {cells[6]:>7} {cells[7]:>7} "
                         f"{cells[8]:>8} {cells[9]:>7}")
    return "\n".join(lines)


def _population(rows: list[dict]) -> list[str]:
    return sorted(r["model"] for r in rows if r["ratio"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--compare", type=Path, default=None,
                    help="another campaign directory; REFUSED unless its "
                         "population is identical")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--sort", default="keys", choices=["keys"],
                    help="keys, and only keys — the key count is the law this "
                         "table exists to show, and a name sort invites the "
                         "reader to invent one about model size")
    args = ap.parse_args()

    cells = _cells(args.campaign)
    if not cells:
        print(f"No cell under {args.campaign}/proof.", file=sys.stderr)
        return 1
    rows = sorted((_row(c) for c in cells),
                  key=lambda r: (r["keys"] is None, -(r["keys"] or 0)))

    print(SCOPE)
    print()
    print(_fmt(rows, args.markdown))
    print()

    ratios = [r["ratio"] for r in rows if r["ratio"]]
    population = _population(rows)
    no_ratio = [r["model"] for r in rows if not r["ratio"]]

    if ratios:
        print(f"Median gain: x{statistics.median(ratios):.2f} "
              f"over {len(ratios)} cells — and that number means nothing "
              f"detached from this population:")
        for name in population:
            keys = next(r["keys"] for r in rows if r["model"] == name)
            print(f"    {name} ({keys} keys)")
    if no_ratio:
        print(f"\nNo ratio, deliberately, for {len(no_ratio)}: "
              + ", ".join(no_ratio))
        print("    An arm that served nothing and swept nothing did not measure "
              "the lever. 1.0 would read as 'no gain'; the truth is 'no "
              "measurement'.")

    if args.compare:
        other = _cells(args.compare)
        other_rows = [_row(c) for c in other]
        other_pop = _population(other_rows)
        if other_pop != population:
            only_here = sorted(set(population) - set(other_pop))
            only_there = sorted(set(other_pop) - set(population))
            print("\nCOMPARISON REFUSED — the two populations differ.", file=sys.stderr)
            if only_there:
                print(f"  only in {args.compare.name}: {', '.join(only_there)}",
                      file=sys.stderr)
            if only_here:
                print(f"  only in {args.campaign.name}: {', '.join(only_here)}",
                      file=sys.stderr)
            print("  A median moves when the population moves. Comparing these "
                  "would say something about which models were run, in the "
                  "grammar of a statement about the engine. Re-run the missing "
                  "cells, or compare the models present in both — explicitly, "
                  "and say so.", file=sys.stderr)
            return 2
        o = statistics.median([r["ratio"] for r in other_rows if r["ratio"]])
        print(f"\nSame population. {args.compare.name}: x{o:.2f} → "
              f"{args.campaign.name}: x{statistics.median(ratios):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
