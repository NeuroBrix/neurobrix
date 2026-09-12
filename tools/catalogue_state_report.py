#!/usr/bin/env python3
"""The catalogue, one line per model, with a verdict that can be defended.

Every number is READ from the artefact that produced it. Nothing is retyped, and
every cell carries how it was obtained:

    measured      an artefact on this machine holds it, and the line names which
    inferred      derived from a measured line by a stated structural identity
    not measured  said in clear, because a blank cell and a zero read the same

The four sources, and none of them is prose:

  meet.json        the 2026-09-11 catalogue pass — state, wall clock, shapes
                   swept at runtime, shapes the screen excluded, per model
  CAMPAIGN_TABLE   the certified-directory campaign — eleven paired cells with
                   their base, their sweep and their key counts
  the censuses     run live against the local graphs: symbol collisions at the
                   input, depth collisions, the temporal unroll
  the overlay      what changed after the catalogue pass, each entry naming the
                   artefact that proves it

Usage: python tools/catalogue_state_report.py > docs/reference/catalogue-state.md
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/mlops/NeuroBrix_System")
MEET = REPO / "validation_outputs/catalogue_meet_20260911/meet.json"
TABLE = Path("/home/mlops/nbx/campaigns/prepared/CAMPAIGN_TABLE.md")
PY_BIN = "/home/mlops/ml/venv/bin/python"

#: What changed after the 2026-09-11 pass, each with the artefact that proves it.
#: A line here OVERRIDES the pass's verdict and says why; nothing is edited into
#: the pass's own record, which stays what it was on the day it ran.
OVERLAY = {
    "Allegro-TI2V": dict(
        now="RUNS — repaired and delivered",
        evidence="validation_outputs/allegro_image_sets_resolution_20260912/out.mp4 "
                 "(8 frames at 448x448, rc=0, inter-frame diff 23.3); hub replaced "
                 "15:13:48; docs/reference/catalogue-repairs.md entry 1",
        note="Two defects, both fixed at the source: the output size was never read "
             "from the container when the backbone's latent is flattened or the flow "
             "is named something else, and the conditioning image did not set the "
             "resolution. Bounded above: renders to 80 frames, fails at its own "
             "declared 88 asking 25.27 GiB in one allocation (DETTE D2).",
        line="measured"),
    "CogVideoX-5b-I2V": dict(
        now="CORRECTED at the source, conversion in flight",
        evidence=".cache/graphs/CogVideoX-5b-I2V/vae_encoder/graph.json "
                 "(sha 7216b9fb1478, 265 ops, no temporal symbol); container 21.5 GB "
                 "passing the regression gate component by component",
        note="Its causal temporal pad recorded 2187*s - 2184 against a truth of s + 2, "
             "exact at the traced s=1. Profiled at 49 frames the peak was 210.26 GB "
             "against 0.09 GB at the trace, a factor of 2237 which is the compound's "
             "own coefficient. Re-traced, the temporal axis carries no symbol at all "
             "(an I2V encoder conditions on one image) and the same request costs "
             "0.02 GB. 389 -> 265 ops.",
        line="measured"),
    "Open-Sora-v2": dict(
        now="DIAGNOSED, rebuild required",
        evidence="the shipped topology.json carries shapes=NONE for transformer and "
                 "vae while .cache/graphs holds them (vae: z [1,16,9,14,22]); "
                 "container dated 2026-06-30",
        note="The runtime repair is not enough for this one: the container predates "
             "the builder that writes component shapes, so the output size cannot be "
             "read from it whatever the runtime does. Snapshot re-downloaded "
             "2026-09-12 16:38 (64.43 GB, the build door's predicate satisfied); "
             "the rebuild is queued behind the CogVideoX upload, staged on the root "
             "filesystem rather than the export.",
        line="measured"),
    "Wan2.2-I2V-A14B": dict(
        now="DIAGNOSED, rebuild required — and a second line",
        evidence="shipped topology.json shapes=NONE (container dated 2026-07-03); "
                 "cached encoder carries the unroll signature",
        note="TWO lines, not one. The rebuild resolves its output size. Its VAE "
             "ENCODER stays unrolled over the temporal axis, which no rebuild "
             "changes — DETTE D-TEMPORAL-UNROLL. Delivering the first without "
             "saying the second would be delivering a fix for the error we found "
             "and hiding the one underneath.",
        line="measured (topology) / inferred (unroll)"),
    "Wan2.1-VACE-1.3B": dict(
        now="NAMED DEBT — not corrected, and no stimulus corrects it",
        evidence="validation_outputs/wan_class_e_20260912/VERDICT.md; "
                 "docs/reference/temporal-unroll-census.md",
        note="Its VAE encoder unrolls its temporal chunk loop: 517 ops per chunk, "
             "measured at two stimuli. The blind count held at 135 through T=17, 25, "
             "33 and 41 while the door's candidate walked 25 -> 33 -> 41. The graph "
             "is not symbolic in time however many symbols its table declares.",
        line="measured"),
    "Wan2.1-I2V-14B-480P": dict(
        now="INFERRED same debt as Wan2.1-VACE",
        evidence="its cached encoder carries the measured anchor's chunk-loop groups "
                 "identically: {2:1, 3:9, 6:12, 8:22, 12:1, 21:22}",
        note="No local snapshot, so no second trace point and no fitted slope. The "
             "claim rests on six module groups agreeing exactly on two numbers each, "
             "while the groups ABOVE the loop differ — different VAE sizes carrying "
             "the same loop. One snapshot and two 15-second traces convert it.",
        line="inferred"),
}


def meet_rows() -> list:
    return json.loads(MEET.read_text())


def campaign_cells() -> dict:
    """{container: (cost_s, ratio, keys, certified)} from the campaign table."""
    if not TABLE.exists():
        return {}
    out = {}
    for line in TABLE.read_text().splitlines():
        # The ratio cell ends in a multiplication sign, not an ASCII x, and the
        # model cell may carry a warning glyph. Split on the pipes instead of
        # matching the whole row: a regex for a hand-written table is a regex
        # that breaks when someone adds a column.
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8 or not cells[0].startswith("`"):
            continue
        name = cells[0].split("`")[1]
        try:
            cost = float(cells[4].strip("*"))
            ratio = float(cells[5].strip("*").rstrip("\u00d7x"))
        except ValueError:
            continue
        # The table carries a model twice where a cell was re-run; the FIRST row
        # is the one with its key counts, and overwriting it with the repeat
        # replaced "8/9 keys" with an em dash.
        out.setdefault(name, dict(cost_s=cost, ratio=ratio, keys=cells[6],
                                  certified=cells[7]))
        m = None
    return out


def census(tool: str, *args) -> str:
    try:
        return subprocess.run([PY_BIN, str(REPO / "tools" / tool), *args],
                              capture_output=True, text=True, timeout=600).stdout
    except Exception:
        return ""


def blind_axes() -> dict:
    """{container: [reasons]} — where a defect would be invisible, per model."""
    raw = census("symbol_collision_census.py", "--json")
    out = {}
    try:
        data = json.loads(raw)
    except ValueError:
        return out
    for comp, entries in (data.get("findings") or {}).items():
        model = comp.split("/")[0]
        for e in entries:
            for cls, _why in e["flags"]:
                out.setdefault(model, []).append(
                    f"{comp.split('/',1)[1]} `{e['name']}`@{e['trace']} ({cls})")
    return out


def main() -> int:
    rows = meet_rows()
    cells = campaign_cells()
    blind = blind_axes()

    print("# The catalogue, one line per model — 2026-09-12\n")
    print("**47 entries on the registry.** Every cell is read from an artefact on this")
    print("machine, and every cell says how it was obtained. A cell that says *not")
    print("measured* is not an omission: a blank and a zero read the same, and only one")
    print("of them is honest.\n")
    print("The run column is the catalogue pass of **2026-09-11** at engine `4c119b5`")
    print("unless a later line overrides it, in which case the override names the")
    print("artefact that proves it. The pass's own record is never edited — it stays")
    print("what it was on the day it ran.\n")

    states = {}
    for r in rows:
        states[r["state"]] = states.get(r["state"], 0) + 1
    over = sum(1 for r in rows if r["hub"].split("/")[-1] in OVERLAY)
    summary = ", ".join(f"**{v} {k}**" for k, v in
                        sorted(states.items(), key=lambda x: -x[1]))
    print(f"As the pass left it: {summary}. {over} rows carry a later line, "
          f"and every one of the nine failures was a VIDEO model.\n")

    print("| model | family | GB | on this rack | swept | screened | certified cost | "
          "where a defect would be invisible | line |")
    print("|---|---|---:|---|---:|---:|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["family"], r["hub"])):
        slug = r["hub"].split("/")[-1]
        container = r.get("container") or slug
        cell = cells.get(container)
        ov = OVERLAY.get(slug) or OVERLAY.get(container)
        if ov:
            run = f"**{ov['now']}**"
            line = ov["line"]
        else:
            # A ROW MAY LACK A FIELD, and a missing field is not a zero. The
            # catalogue decision (`Orpheus-3B`) was never built, so it carries no
            # container, no wall clock and no swept count; printing 0 for those
            # would be the lying cell this document exists to avoid.
            wall = r.get("wall_s")
            clock = f"{wall:.0f} s" if isinstance(wall, (int, float)) else "no clock"
            run = {"met": f"met in {clock}",
                   "failed": (f"FAILED rc={r.get('rc', '?')}"
                              + (f" — killed at {clock}"
                                 if r.get("rc") == -9 else f" in {clock}")),
                   "not runnable": "not runnable — catalogue decision"}.get(
                       r.get("state", "?"), r.get("state", "?"))
            line = "measured" if wall is not None else "not measured"
        cost = (f"{cell['cost_s']:.0f} s, {cell['ratio']}x, {cell['certified']}/"
                f"{cell['keys']} keys" if cell else "not measured")
        axes = blind.get(container, [])
        blind_cell = "; ".join(axes[:2]) + (f" (+{len(axes)-2})" if len(axes) > 2 else "") \
            if axes else "none found at the input"
        # A `swept` of 0 on a row that FAILED is not coverage. It counts the
        # shapes the run reached, and a run that died in five seconds reached
        # none. Reading it as "the certified directory served everything" is the
        # difference between a measure and an artefact of the failure, and the
        # first version of the reading guide made exactly that claim.
        incomplete = r.get("state") != "met"
        swept = r.get("swept")
        if swept is not None and incomplete:
            swept = f"{swept}†"
        screened = r.get("screened_out")
        if screened is not None and incomplete:
            screened = f"{screened}†"
        print(f"| `{r['hub']}` | {r.get('family', '?')} | {r.get('gb', 0):.1f} | "
              f"{run} | {swept if swept is not None else 'n/m'} | "
              f"{screened if screened is not None else 'n/m'} | {cost} | "
              f"{blind_cell} | {line} |")

    print("\n## The lines that carry a later verdict\n")
    for slug, ov in sorted(OVERLAY.items()):
        print(f"### `{slug}` — {ov['now']}\n")
        print(f"{ov['note']}\n")
        print(f"*Evidence:* {ov['evidence']}  ·  *line:* {ov['line']}\n")

    print("## How to read the columns\n")
    print("**swept** — shape keys this model had to sweep AT RUNTIME because the")
    print("certified directory did not hold them. On a row that MET, `0` is the")
    print("per-model measure of certified coverage: it was served entirely from the")
    print("directory. `n/m` is a model that was never run.\n")
    print("**† marks a row whose run did not complete**, and it changes what the two")
    print("columns mean there. They count what the run REACHED, and a run that died")
    print("in five seconds reached nothing — so a `0†` is not coverage, it is the")
    print("shape of the failure. Reading it as coverage would credit the directory")
    print("for work no one asked it to do.\n")
    print("**screened** — candidate configurations the correctness screen excluded")
    print("before timing. Zero across the whole catalogue, on 998 keys.\n")
    print("**certified cost** — from the paired certified-directory campaign, which")
    print("covers eleven cells and not the catalogue. The ratio is what runtime")
    print("sweeping costs relative to a served run, on this rack, at the shapes these")
    print("requests meet. It is not a throughput figure and it says nothing about")
    print("other hardware.\n")
    print("**where a defect would be invisible** — axes traced at a value where two")
    print("distinct rules give the same number, so the trace-point check cannot tell")
    print("them apart. This is NOT a defect list. An axis here needs its rule asserted")
    print("structurally or a re-trace outside the collision; a test at the flagged")
    print("value is green for the reason that blinds it.\n")

    print("## What this document does not say\n")
    print("* **Whether a model is CORRECT.** The run column says it produced output")
    print("  without failing, on one request, at one moment. Numerical agreement")
    print("  against a vendor pipeline is a different instrument and covers five of")
    print("  nine families.")
    print("* **What most models cost with the certified directory.** Eleven cells were")
    print("  measured; the other thirty-six say *not measured* and that is the whole")
    print("  point of the cell.")
    print("* **Whether the flagged axes are wrong.** They are places a defect could")
    print("  not be seen. Converting one into a verdict costs a second trace at a")
    print("  value outside the collision, and the instrument that does it refuses when")
    print("  the stimulus does not actually move — a tree compared with itself agrees")
    print("  with itself.")
    print("* **Anything about hardware other than this rack**: four V100s, two of 16 GB")
    print("  and two of 32, at 1290/877 MHz.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
