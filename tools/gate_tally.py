#!/usr/bin/env python3
"""gate_tally.py — THE gate tally, committed (CPU, reads only).

Per class (the validation regime: ONE class per gate, the card's real VRAM): the gate's wanted cells
(<gate>/gate2/gate_<cls>.list, native + triton), the cells this gate REPLAYED (latest row per (model, mode)
under <gate>/matrix_<cls>), the cells it had to replay (the computed scope, <scope>/queue_<cls>.txt), and for
every wanted cell not replayed the row CARRIED from the base gate with its provenance. A scope cell with no row
is MISSING (an error, never a pass). A red is named; a carried red is accepted only when the TREE's committed
per-class record (docs/reference/known-cause-reds.<cls>.yml) names it. The tally compares against the committed
reference, never against itself.

Quality (rule 2): every wanted (model, mode) is read against the committed judgments record
(--judgments, default <tree>/docs/reference/judgments.jsonl, tools/judgments.py) and the gate's own
<gate>/matrix_<cls>/judgments.jsonl; the pairs no judgment covers are NAMED ("NOT JUDGED"). A judgment is made
once, outside the gate, so a missing one is named, not counted as a red.

    gate_tally.py --gate <campaign gate dir> --base <base gate dir> --tree <tree> --classes 32g [--scope <dir>]
Exit 1 on a red, a missing scope cell, a never-measured cell, or a carried red without a known cause.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def latest_rows(matrix_dir):
    last = {}
    for f in sorted(glob.glob(f"{matrix_dir}/rows_card*.jsonl")):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            k = (r.get("model"), r.get("mode"))
            if k not in last or r.get("date", "") >= last[k].get("date", ""):
                last[k] = r
    return last


def wanted(list_path, modes=("native", "triton")):
    want = {}
    for l in open(list_path):
        p = l.split()
        if p and not p[0].startswith("#"):
            want[p[0]] = [m for m in (p[1].split(",") if len(p) > 1 else list(modes)) if m in modes]
    return want


def scope_cells(queue_path):
    cells = set()
    for l in open(queue_path):
        p = l.split()
        if p and not p[0].startswith("#"):
            for m in p[1].split(","):
                cells.add((p[0], m))
    return cells


def known(path, cls):
    """The TREE's committed per-class record (the harness's schema): {(model, mode, cls): cause}. A missing or
    unreadable record is an error, never an empty dict."""
    import yaml
    d = yaml.safe_load(open(path)) or {}
    return {(r["model"], r["mode"], cls): r["cause"] for r in d.get("reds") or []}


def judgments(paths):
    """{(model, mode): judgment} — the latest per pair over the committed record and the gate's own files."""
    out = {}
    for p in paths:
        p = Path(p)
        if not p.is_file():
            continue
        for l in p.read_text().splitlines():
            try:
                j = json.loads(l)
            except ValueError:
                continue
            if not all(k in j for k in ("model", "mode", "verdict", "date")):
                continue
            k = (j["model"], j["mode"])
            if k not in out or j["date"] >= out[k]["date"]:
                out[k] = j
    return out


def cell(r):
    return f"rc{r['rc']} {r.get('wall_s', '?')}s {r.get('date', '')[:16]}"


def tally(a) -> int:
    rc = 0
    modes = tuple(a.modes.split(","))
    for cls in a.classes.split(","):
        want = wanted(f"{a.gate}/gate2/gate_{cls}.list", modes)
        rows = latest_rows(f"{a.gate}/matrix_{cls}")
        overlaid = 0
        if a.overlay:
            for k, r in latest_rows(f"{a.overlay}/matrix_{cls}").items():
                rows[k] = r
                overlaid += 1
        base = latest_rows(f"{a.base}/matrix_{cls}")
        kn = known(f"{a.tree}/docs/reference/known-cause-reds.{cls}.yml", cls)
        scope = scope_cells(f"{a.scope}/queue_{cls}.txt") if a.scope else None
        judged = judgments([a.judgments or f"{a.tree}/docs/reference/judgments.jsonl",
                            f"{a.gate}/matrix_{cls}/judgments.jsonl"])
        green, red, carried_g, carried_r, missing, never, unjudged = 0, [], 0, [], [], [], []
        for m, ms in sorted(want.items()):
            for x in ms:
                if (m, x) not in judged:
                    unjudged.append(f"{m}:{x}")
                if (m, x) in rows:
                    r = rows[(m, x)]
                    if r["rc"] == 0:
                        green += 1
                    else:   # a replayed red stays red; the record's cause is printed beside it, the decision is the supervisor's
                        cause = kn.get((m, x, cls))
                        red.append(f"{m}:{x} {cell(r)} {r.get('error', '')[:60]}"
                                   + (f" | KNOWN CAUSE (record): {cause[:120]}" if cause else ""))
                elif scope is not None and (m, x) in scope:
                    missing.append(f"{m}:{x} (in the computed scope, no row)")
                elif (m, x) in base:
                    b = base[(m, x)]
                    if b["rc"] == 0:
                        carried_g += 1
                    else:
                        cause = kn.get((m, x, cls))
                        carried_r.append(f"{m}:{x} base {cell(b)} " + (f"known cause: {cause}" if cause else "NO KNOWN CAUSE"))
                else:
                    never.append(f"{m}:{x}")
        if scope is not None:
            for m, x in sorted(scope):
                if x not in want.get(m, []):
                    missing.append(f"{m}:{x} (in the computed scope but NOT in the gate list)")
        n_replayed = sum(1 for (m, x) in rows if m in want and x in want[m])
        n_want = sum(len(v) for v in want.values())
        print(f"{cls}: wanted {n_want} cells; replayed {n_replayed}: {green} green / {len(red)} red; "
              f"carried from base {carried_g + len(carried_r)}: {carried_g} green / {len(carried_r)} red; "
              f"missing {len(missing)}; never measured {len(never)}"
              + (f"; scope cells {len(scope)}" if scope is not None else "; scope not given")
              + (f"; overlay rows {overlaid}" if a.overlay else "")
              + f"; judged {n_want - len(unjudged)} / not judged {len(unjudged)}")
        for s in red:
            print("  RED", s)
        for s in carried_r:
            print("  CARRIED RED", s)
        for s in missing:
            print("  MISSING", s)
        for s in never:
            print("  NEVER MEASURED", s)
        for s in unjudged:
            print("  NOT JUDGED", s)
        if red or missing or never or any("NO KNOWN CAUSE" in s for s in carried_r):
            rc = 1
    return rc


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate", required=True, help="the gate: gate2/gate_<cls>.list and matrix_<cls>/ rows")
    ap.add_argument("--base", required=True, help="the base gate the unreplayed cells are carried from (its matrix_<cls>/)")
    ap.add_argument("--scope", default=None, help="the computed scope: <dir>/queue_<cls>.txt")
    # the landing tree's own rows (a candidate's replay/confirm): they take precedence over the frozen tree's row
    # for the same cell, so the tally judges the tree that lands, not the one the delta started from
    ap.add_argument("--overlay", default=None)
    ap.add_argument("--tree", required=True, help="the tree whose committed records name the carried reds and the judgments")
    ap.add_argument("--judgments", default=None, help="the committed judgments record (default <tree>/docs/reference/judgments.jsonl)")
    ap.add_argument("--classes", default="32g")
    ap.add_argument("--modes", default="native,triton")
    return ap


def main(argv=None) -> int:
    return tally(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
