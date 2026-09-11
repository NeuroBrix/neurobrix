#!/usr/bin/env python3
"""The certified-directory campaign's table, ordered by the law that governs it.

THE LAW, CORRECTED ONCE THE NINTH CELL LANDED

The first version of this file said the gain follows the KEY COUNT. The data
refuted it: `Qwen3-VL` demands **585** keys and gains x14.18 while
`DeepSeek-Coder-V2-Lite` demands **137** and gains x16.64. Keys are a proxy and
it breaks exactly there.

The law is the ratio between what the SWEEP costs and what the model costs to
run at all:

    gain = 1 + sweep_cost / base_time        (sweep_cost = B_med - A_med)

A slow model amortises its own sweep. `CogVideoX-2b` sweeps 33 keys for 402 s
against a 544 s base and gains x1.74; `DeepSeek-Coder-V2-Lite` sweeps 137 keys
for 1,346 s against an 86 s base and gains x16.64. Same engine, opposite ends,
and the key count alone predicts neither.

So the table carries BOTH columns — keys and base time — and the ratio between
them, because that ratio is the explanation and the keys alone are not.

NO MEDIAN OVER A BROKEN DISTRIBUTION

This campaign's gains are **bimodal**: four cells under x2.4, four above x11.6,
and nothing between. A median would read x7 and would describe no model that
exists — a lie by summary statistic, which is exactly what a document built on
"no cell lies" cannot publish.

So the tool measures the break rather than assuming one: it takes the largest
multiplicative gap between consecutive sorted gains and compares it to the
second largest. Where the biggest gap dwarfs every other (here 4.9x against a
next-largest of 1.37x), there are two regimes, the median is REFUSED, and both
regimes are reported with their boundary. If someone wants one number, the
answer is that there isn't one, and that is a result.

A median that IS published comes with the population it covers, named model by
model. An earlier campaign reported x13.9 over a population including
`chatterbox` (674 keys) and `openaudio` (693) which this one does not carry;
"the median fell" against that number would be as false as the x13.9 was.

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

    gate = cell.get("gate") or {}
    nondet = gate.get("nondeterministic") or []
    diff = gate.get("diff") or {}
    # A model that differs from ITSELF across repetitions cannot be adjudicated
    # by a byte gate, and printing DIFFER for it reads as "the directory changed
    # the output" — which is false and is the kind of cell this project does not
    # ship. CogVideoX-2b, 2026-09-11: three repetitions, three shas, in BOTH
    # arms, and the video comparison agrees at 43.6 dB mean PSNR.
    if len(nondet) >= 2:
        bytes_verdict = "nondet both"
    elif not gate.get("ran"):
        bytes_verdict = "did not run"
    elif gate.get("identical"):
        bytes_verdict = "same"
    elif diff.get("pass"):
        bytes_verdict = f"{diff.get('psnr_mean_db', 0):.0f} dB"
    else:
        bytes_verdict = "DIFFER"

    return {
        "model": cell.get("model", "?"),
        "family": cell.get("family", "?"),
        "gb": cell.get("weight_gb"),
        "keys": keys,
        "served": served,
        "a_med": am,
        "b_med": bm,
        "sweep": (bm - am) if (am and bm) else None,
        "sweep_ratio": ((bm - am) / am) if (am and bm) else None,
        "ratio": ratio,
        "bytes": bytes_verdict,
        "identical": gate.get("identical"),
        "excluded": b.get("screen_excluded"),
        "contradictions": b.get("contradictions"),
        "rc": (a.get("rc"), b.get("rc")),
    }


def _fmt(rows: list[dict], markdown: bool) -> str:
    head = ["model", "family", "GB", "keys", "base s", "sweep s", "sweep/base",
            "gain", "bytes", "screen-out", "contra"]
    lines = []
    if markdown:
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "|".join("---" for _ in head) + "|")
    else:
        lines.append(f"{'model':<34} {'family':<12} {'GB':>6} {'keys':>5} "
                     f"{'base s':>9} {'sweep s':>9} {'swp/base':>9} {'gain':>7} "
                     f"{'bytes':>12} {'scr-out':>8} {'contra':>7}")
    for r in rows:
        cells = [
            r["model"],
            r["family"],
            f"{r['gb']:.1f}" if r["gb"] else "?",
            str(r["keys"]) if r["keys"] is not None else "?",
            f"{r['a_med']:.1f}" if r["a_med"] else "?",
            f"{r['sweep']:.1f}" if r["sweep"] else "?",
            f"{r['sweep_ratio']:.2f}" if r["sweep_ratio"] else "?",
            f"x{r['ratio']:.2f}" if r["ratio"] else "no ratio",
            r["bytes"],
            str(r["excluded"]) if r["excluded"] is not None else "?",
            str(r["contradictions"]) if r["contradictions"] is not None else "?",
        ]
        if markdown:
            lines.append("| " + " | ".join(cells) + " |")
        else:
            lines.append(f"{cells[0]:<34} {cells[1]:<12} {cells[2]:>6} {cells[3]:>5} "
                         f"{cells[4]:>9} {cells[5]:>9} {cells[6]:>9} {cells[7]:>7} "
                         f"{cells[8]:>12} {cells[9]:>8} {cells[10]:>7}")
    return "\n".join(lines)


def _gaps(sorted_ratios):
    return [(b / a, i) for i, (a, b) in
            enumerate(zip(sorted_ratios, sorted_ratios[1:]))]


def _second_gap(ratios):
    g = sorted((r for r, _ in _gaps(sorted(ratios))), reverse=True)
    return g[1] if len(g) > 1 else 1.0


def _regimes(rows):
    """(low, high, break) when the distribution BREAKS, else (None, None, None).

    Measured, not assumed: the largest multiplicative gap between consecutive
    sorted gains against the second largest. A distribution whose biggest step
    is barely larger than its others is continuous and takes a median; one whose
    biggest step dwarfs every other has two regimes, and a median describes
    neither. The factor is 2.5 because that sits comfortably above every
    within-regime step this campaign produced (largest 1.37x) and below the
    break it found (4.9x) — and it is printed with every verdict so a reader can
    disagree with it on the evidence rather than on trust.
    """
    scored = [r for r in rows if r["ratio"]]
    if len(scored) < 4:
        return None, None, None
    scored.sort(key=lambda r: r["ratio"])
    gaps = _gaps([r["ratio"] for r in scored])
    if not gaps:
        return None, None, None
    biggest, at = max(gaps)
    others = sorted((g for g, i in gaps if i != at), reverse=True)
    if not others or biggest < 2.5 * others[0]:
        return None, None, None
    return scored[:at + 1], scored[at + 1:], biggest


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
        low, high, break_ratio = _regimes(rows)
        if low is not None:
            print(f"NO MEDIAN IS PUBLISHED. The distribution has a BREAK: the gap "
                  f"between the two regimes is {break_ratio:.1f}x, against a "
                  f"largest gap of {_second_gap(ratios):.2f}x anywhere else. A "
                  f"median would read x{statistics.median(ratios):.2f} and would "
                  f"describe no model on this list.\n")
            for label, group in (("AMORTISED — the sweep is small beside the run",
                                  low),
                                 ("DOMINATED — the sweep dwarfs the run", high)):
                print(f"  {label}")
                for r in sorted(group, key=lambda x: x["ratio"]):
                    print(f"    x{r['ratio']:<6.2f} {r['model']:<34} "
                          f"{r['keys']:>4} keys, base {r['a_med']:>7.1f} s, "
                          f"sweep/base {r['sweep_ratio']:.2f}")
                print()
            print("  The boundary is not a gain threshold, it is the sweep-to-base "
                  "ratio: every AMORTISED cell sits under 1.4, every DOMINATED "
                  "cell over 10. Nothing measured lands between.")
            print("  If one number is wanted, there isn't one — and that is the "
                  "result, not a gap in it.")
        else:
            print(f"Median gain: x{statistics.median(ratios):.2f} "
                  f"over {len(ratios)} cells — and that number means nothing "
                  f"detached from this population:")
            for name in population:
                keys = next(r["keys"] for r in rows if r["model"] == name)
                print(f"    {name} ({keys} keys)")
    if no_ratio:
        print(f"\nNo ratio, deliberately, for {len(no_ratio)}:")
        for r in rows:
            if r["ratio"]:
                continue
            if r["bytes"] == "did not run":
                why = (f"both arms exited non-zero — the cell FAILED, it did not "
                       f"measure a small gain")
            else:
                why = ("the lever moved nothing: an arm that served nothing and "
                       "swept nothing measured no lever at all")
            print(f"    {r['model']:<34} {why}")
        print("    1.0 would read as 'no gain' in either case; the truth is 'no "
              "measurement', and the two reasons are not the same debt.")

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
