#!/usr/bin/env python3
"""The catalogue against the certified directory: one line per hub model.

WHAT THIS DOCUMENT IS FOR

The certified directory is the only place in this engine where a proof exists.
It currently covers a handful of shapes met by a handful of models, and the
target is the catalogue — forty-seven models on the hub. This report says, model
by model, where each one stands, and it is built so that a cell which has not
been measured SAYS SO rather than carrying a plausible number.

THE RULE THAT SHAPES EVERY COLUMN

A cell reads "not measured" or it reads a measurement. There is no third kind.
An estimate appears only where its BASIS is named in the same cell, and only
where that basis is a recorded run rather than a model of one.

That rule cost a shortcut on 2026-09-10 and the negative result is kept here so
nobody re-derives it: a STATIC count of distinct autotuned-op shapes, taken from
`graph.json`, does NOT predict the key count. Two containers with 54 static
shapes each measured 8 and 137 keys — a 17x spread across the catalogue. The
static count is not a weak predictor, it is not a predictor, and the column it
would have filled stays empty instead.

ORDERING

By known cost ascending, where "known" means a recorded run of that cell. Five
of forty-seven have one. The rest are ordered by weight as a declared crude
proxy whose only job is to try cheap things first — it is a guess, it is labelled
one, and it is NOT what protects the night. The budget guard is: a cell whose
KNOWN cost exceeds what remains is refused at the door, and a cell with no known
cost is never refused on a guess (`tools/precision_zoo_campaign.py`,
`cell_cost_estimate`).

Usage:
    python tools/certified_catalogue_report.py --out validation_outputs/<dir>
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CERTIFIED = REPO / "src" / "neurobrix" / "config" / "autotune" / "nvidia" / "volta"

HEADER = """# The catalogue against the certified directory — {today}

**Everything in this document concerns `nvidia/volta` and nothing else.**
Everywhere else the runtime sweeps at load and the guarantee is the consensus
screen, where *screened* means the candidates agreed with each other — never
that anything was proved against an oracle. That distinction is not a nuance
here, it is what this whole document is about:
`docs/reference/what-certified-means.md`.

A cell that has not been measured says **not measured**. No cell carries a
number whose basis is not named in the cell beside it.
"""


def _hub(snapshot: Path) -> list[dict]:
    """Parse the `neurobrix hub` listing captured beside this report."""
    rows, started = [], False
    for line in snapshot.read_text().splitlines():
        if line.startswith("---"):
            started = True
            continue
        if not started or not line.strip() or line.startswith("Total:"):
            continue
        if line.startswith(("Install:", "Installed locally:")):
            continue
        m = re.match(r"^(\S+)\s+(\S+)\s+([\d.]+\s*[GM]B)\s+(.*?)\s+(\d+)\s*(installed)?\s*$", line)
        if not m:
            continue
        size = m.group(3).replace(" ", "")
        gb = float(size[:-2]) / (1024 if size.endswith("MB") else 1)
        rows.append({
            "hub": m.group(1),
            "slug": m.group(1).split("/")[-1],
            "family": m.group(2),
            "gb": gb,
            "installed": bool(m.group(6)),
        })
    return rows


#: A campaign carrying one of these files measured something it should not have
#: — a stale tree, a moving tree, a perturbed cell — and its records are NOT
#: evidence. Reading one would put a number in this document whose basis is a
#: run its own author declared void.
INVALIDATION_MARKERS = ("INVALIDATED.md", "PERTURBATION_NOTE.md", "STALE.md")

#: Hub slug -> the name the measurement recorded, ONLY where the two genuinely
#: differ. Matching is otherwise exact on the lowercased slug. No prefix rule:
#: the first version of this tool used one and it silently attributed
#: `Qwen3-Coder-30B-A3B-Instruct`'s measurement to its int4g128-ffnonly variant,
#: which had never been run — a lying cell inside the document whose whole rule
#: is that no cell lies.
ALIASES = {
    "qwen3-30b-a3b-thinking": "qwen3-30b-a3b-thinking-2507",
}


def _excluded_campaigns(campaigns: Path) -> list[tuple[str, str]]:
    out = []
    for d in sorted(campaigns.glob("*/")):
        for marker in INVALIDATION_MARKERS:
            if (d / marker).exists():
                out.append((d.name, marker))
                break
    return out


def _campaign_cells(campaigns: Path) -> dict:
    """Per model: the measured key count, what was served, the gain, the
    screen's refusals. Only from records that completed, in campaigns nobody
    invalidated."""
    void = {name for name, _ in _excluded_campaigns(campaigns)}
    out = {}
    for result in campaigns.glob("*/proof/*/result.json"):
        if result.parent.parent.parent.name in void:
            continue
        try:
            d = json.loads(result.read_text())
        except (OSError, ValueError):
            continue
        a, b = d.get("A") or {}, d.get("B") or {}
        if a.get("rc") != 0 or b.get("rc") != 0:
            continue
        reps = lambda arm: [r["exec_s"] for r in (arm.get("reps") or [])
                            if r.get("rc") == 0 and isinstance(r.get("exec_s"), (int, float))]
        ra, rb = reps(a), reps(b)
        am = statistics.median(ra) if ra else a.get("exec_s")
        bm = statistics.median(rb) if rb else b.get("exec_s")
        keys = b.get("swept")
        gain = (bm / am) if (am and bm and keys) else None
        out[d.get("model", result.parent.name)] = {
            "keys": keys,
            "served": a.get("certified_served"),
            "gain": gain,
            "excluded": b.get("screen_excluded"),
            "contradictions": b.get("contradictions"),
            "identical": (d.get("gate") or {}).get("identical"),
            "cost_s": (a.get("wall_s") or 0) + (b.get("wall_s") or 0),
            "campaign": result.parent.parent.parent.name,
        }
    return out


def _certified_totals() -> tuple[int, dict]:
    per_file, total = {}, 0
    for f in sorted(CERTIFIED.glob("*.json")):
        try:
            n = len(json.loads(f.read_text()).get("entries", {}))
        except (OSError, ValueError):
            n = 0
        per_file[f.name] = n
        total += n
    return total, per_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--campaigns", type=Path,
                    default=Path("/home/mlops/nbx/campaigns"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    snapshot = args.out / "hub_snapshot.txt"
    if not snapshot.exists():
        print(f"missing {snapshot} — capture `neurobrix hub` beside the report first")
        return 1

    hub = _hub(snapshot)
    cells = _campaign_cells(args.campaigns)
    total_keys, per_file = _certified_totals()

    by_name = {name.lower(): c for name, c in cells.items()}

    def cell_for(row):
        slug = row["slug"].lower()
        return by_name.get(slug) or by_name.get(ALIASES.get(slug, ""))

    ranked = []
    for row in hub:
        c = cell_for(row)
        row["cell"] = c
        row["known_cost"] = c["cost_s"] if c else None
        ranked.append(row)
    ranked.sort(key=lambda r: (r["known_cost"] is None, r["known_cost"] or 0, r["gb"]))

    void = _excluded_campaigns(args.campaigns)
    lines = [HEADER.format(today=date.today().isoformat()), ""]
    if void:
        lines.append("Campaigns deliberately NOT read, because their own author "
                     "declared them void:\n")
        for name, marker in void:
            lines.append(f"* `{name}` — `{marker}`")
        lines.append("")
    lines.append(f"The directory holds **{total_keys:,} certified entries** across "
                 f"{len(per_file)} kernel/dtype files:\n")
    for name, n in sorted(per_file.items(), key=lambda kv: -kv[1]):
        lines.append(f"* `{name}` — {n:,}")
    measured = [r for r in ranked if r["cell"]]
    lines += ["", f"**{len(measured)} of {len(ranked)} models have been measured against it.** "
                  f"The other {len(ranked) - len(measured)} carry `not measured` in every "
                  f"column that would otherwise be a guess.", ""]

    lines += ["| # | model | family | GB | keys demanded | served certified | gain | screened out | known cost |",
              "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ranked, 1):
        c = r["cell"]
        if c:
            keys = str(c["keys"]) if c["keys"] is not None else "?"
            served = str(c["served"]) if c["served"] is not None else "?"
            gain = f"x{c['gain']:.2f}" if c["gain"] else "no ratio — the lever moved nothing"
            out = (f"{c['excluded']}" if c["excluded"] is not None else "?")
            cost = f"{c['cost_s']:.0f} s (measured, {c['campaign']})"
        else:
            keys = served = gain = out = "not measured"
            cost = "not measured"
        lines.append(f"| {i} | `{r['hub']}` | {r['family']} | {r['gb']:.1f} | {keys} | "
                     f"{served} | {gain} | {out} | {cost} |")

    lines += ["", "## What the measured rows say, and what they do not", ""]
    if measured:
        ratios = [r["cell"]["gain"] for r in measured if r["cell"]["gain"]]
        if ratios:
            lines.append(f"Median gain **x{statistics.median(ratios):.2f}** over "
                         f"{len(ratios)} cells. That number is inseparable from its "
                         f"population, which is exactly these models and no others:")
            for r in measured:
                if r["cell"]["gain"]:
                    lines.append(f"* `{r['hub']}` — {r['cell']['keys']} keys, "
                                 f"x{r['cell']['gain']:.2f}")
            lines += ["", "**It may not be compared with a median over a different "
                          "population.** An earlier campaign reported x13.9 over a set "
                          "that included `chatterbox` (674 keys) and `openaudio` (693); "
                          "this one does not carry them. Saying \"the median fell\" "
                          "against that number would describe which models were run, "
                          "in the grammar of a statement about the engine.", ""]
        lines += ["The law these rows show is the **key count**, not the model size: "
                  "54 GB at 203 keys gains x11.69 while 61 GB at 8 keys gains x1.54. "
                  "The table is ordered by cost, but read the keys column for the "
                  "explanation of any gain.", ""]

    lines += ["## Why the unmeasured rows have no estimate", "",
              "A static count of distinct autotuned-op shapes, read from each "
              "container's `graph.json`, was tried as a predictor of the key count "
              "on 2026-09-10 and **does not work**: two containers with 54 static "
              "shapes each measured 8 and 137 keys, and the ratio of measured to "
              "static ranges from 0.15 to 2.54 across the five known cells. It is "
              "not a weak predictor; it is not a predictor. Rather than fill the "
              "column with it, the column stays empty.", "",
              "The consequence for planning: the certification order is by KNOWN "
              "cost, and beyond the five known cells it is by weight — a declared "
              "crude proxy whose only job is to try cheap things first. What "
              "protects a night's budget is not that ordering but the guard: a cell "
              "whose known cost exceeds what remains is refused at the door, and a "
              "cell with no known cost is never refused on a guess.", ""]

    (args.out / "CATALOGUE.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out / 'CATALOGUE.md'} — {len(ranked)} models, "
          f"{len(measured)} measured, {total_keys:,} certified entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
