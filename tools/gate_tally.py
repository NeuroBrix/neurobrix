#!/usr/bin/env python3
"""gate_tally.py — THE gate tally, committed (CPU, reads only).

Per class (the validation regime: ONE class per gate, the card's real VRAM), in the scope tool's gate layout
(tools/gate_scope.yml `gate:`): the gate's wanted cells (<gate>/gate2/gate_<cls>.list, native + triton), the cells
this gate REPLAYED (latest row per (model, mode) under <gate>/gate/matrix_<cls>), the cells it had to replay (the computed scope, <scope>/queue_<cls>.txt), and for
every wanted cell not replayed the row CARRIED from the base gate with its provenance. A scope cell with no row
is MISSING (an error, never a pass). A red is named; a carried red is accepted only when the TREE's committed
per-class record (docs/reference/known-cause-reds.<cls>.yml) names it. The tally compares against the committed
reference, never against itself.

Quality (rule 2): every wanted cell's OUTPUT (its row's sha256) is read against the committed judgments record
(--judgments, default <tree>/docs/reference/artefact-judgments.jsonl) and the gate's own
<gate>/gate/matrix_<cls>/judgments.jsonl (regression_matrix.py judge writes the sha); an output no judgment names is
NAMED ("NOT JUDGED"), an output judged other than works is RED. Time: a replayed cell slower than
`gate.slower_than_base_red` times its base row (tools/gate_scope.yml) is RED (the supervisor, 2026-10-26 23:20).

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
    """{output sha256: judgment} — the latest per OUTPUT over the committed record and the gate's own files. A judgment
    names the sha of what was looked at; one without a sha judged an output nobody can name, and judges nothing here
    (the supervisor, 2026-10-26 23:20: a pair judged once let every later output of that pair pass as judged)."""
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
            sha, date = j.get("sha256"), j.get("date") or j.get("run_date") or ""
            if not sha or j.get("verdict") in (None, "not judged"):
                continue
            if sha not in out or date >= out[sha]["_date"]:
                out[sha] = dict(j, _date=date)
    return out


def compute_s(r):
    """A run's compute seconds, the load spans removed (core/runtime/run_timings: `ttft_s` excludes every load): the
    output's ttft for a run without tokens, ttft + (tokens - 1) x tpot for a generation; None when the row has no
    timings line. Gate ebm (2026-10-27): four cells at once reading the shared storage tripled Wan2.2-I2V's load
    (74 -> 186 s) while its compute fell (72 -> 65 s); exec_s, which holds the in-request loads, called it a regression."""
    t = r.get("ttft_s")
    if t is None:
        return None
    if r.get("first_output") == "token" and r.get("tokens") and r.get("tpot_ms") is not None:
        return float(t) + (int(r["tokens"]) - 1) * float(r["tpot_ms"]) / 1000.0
    return float(t)


def slower(r, b, factor):
    """`(new, base, field)` when the row is over `factor` times its base on the same measure — the compute seconds
    when both rows carry them (loads removed: storage contention is not the engine's speed), else exec_s, else wall_s;
    None otherwise or when either row has no time."""
    for field, get in (("compute_s", compute_s), ("exec_s", lambda x: x.get("exec_s")), ("wall_s", lambda x: x.get("wall_s"))):
        n, o = get(r), get(b)
        if n is not None and o is not None:
            return (n, o, field) if o > 0 and float(n) > factor * float(o) else None
    return None


def cell(r):
    return f"rc{r['rc']} {r.get('wall_s', '?')}s {r.get('date', '')[:16]}"


def layout() -> dict:
    """THE gate layout (tools/gate_scope.yml `gate:`), the one the scope tool and gate_run read."""
    import yaml
    return (yaml.safe_load((Path(__file__).resolve().parent / "gate_scope.yml").read_text()) or {})["gate"]


def tally(a) -> int:
    rc = 0
    lay = layout()
    modes = tuple(a.modes.split(","))
    for cls in a.classes.split(","):
        want = wanted(f"{a.gate}/{lay['lists'].format(cls=cls)}", modes)
        rows_dir = f"{a.gate}/{lay['rows'].format(cls=cls)}"
        rows = latest_rows(rows_dir)
        overlaid = 0
        if a.overlay:
            for k, r in latest_rows(f"{a.overlay}/{lay['rows'].format(cls=cls)}").items():
                rows[k] = r
                overlaid += 1
        base = latest_rows(f"{a.base}/{lay['rows'].format(cls=cls)}")
        if not base:
            # an empty or mis-named base compares nothing and every cell passes its time check: refused by name
            print(f"{cls}: REFUSED — the base {a.base}/{lay['rows'].format(cls=cls)} holds no rows")
            return 2
        kn = known(f"{a.tree}/docs/reference/known-cause-reds.{cls}.yml", cls)
        scope = scope_cells(f"{a.scope}/queue_{cls}.txt") if a.scope else None
        judged = judgments([a.judgments or f"{a.tree}/docs/reference/artefact-judgments.jsonl",
                            f"{rows_dir}/judgments.jsonl"])
        factor = float(lay["slower_than_base_red"])
        green, red, carried_g, carried_r, missing, never, unjudged = 0, [], 0, [], [], [], []
        for m, ms in sorted(want.items()):
            for x in ms:
                out = rows.get((m, x)) or base.get((m, x))
                j = judged.get((out or {}).get("sha256") or "")
                if out is not None and out.get("rc") == 0 and j is None:
                    unjudged.append(f"{m}:{x} sha {str(out.get('sha256'))[:12]}")
                if (m, x) in rows:
                    r = rows[(m, x)]
                    # a base that FAILED has no time to compare (gate final 16g, 2026-10-27: CogVideoX-I2V's 405 s
                    # "slower" than ebm's 29 s refusal at the door)
                    b_row = base.get((m, x))
                    slow = slower(r, b_row, factor) if b_row is not None and b_row.get("rc") == 0 else None
                    if r["rc"] == 0 and j is not None and j["verdict"] != "works":
                        red.append(f"{m}:{x} {cell(r)} output {r['sha256'][:12]} JUDGED {j['verdict']}: {j.get('judged', '')[:100]}")
                    elif r["rc"] == 0 and slow:
                        red.append(f"{m}:{x} {cell(r)} SLOWER {slow[0]:.1f}s vs base {slow[1]:.1f}s ({slow[2]}, "
                                   f"over {factor}x; base {base[(m, x)].get('date', '')[:16]})")
                    elif r["rc"] == 0:
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
        print(f"{cls}: wanted {n_want} cells; replayed {n_replayed}: {green} ran clean (rc 0, zero miss) / {len(red)} red; "
              f"carried from base {carried_g + len(carried_r)}: {carried_g} ran clean / {len(carried_r)} red; "
              f"missing {len(missing)}; never measured {len(never)}"
              + (f"; scope cells {len(scope)}" if scope is not None else "; scope not given")
              + (f"; overlay rows {overlaid}" if a.overlay else "")
              + f"; outputs judged by sha {n_want - len(unjudged)} / not judged {len(unjudged)}")
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
    ap.add_argument("--gate", required=True, help="the gate, in the scope tool's layout: gate2/gate_<cls>.list, gate/matrix_<cls>/ rows")
    ap.add_argument("--base", required=True, help="the base gate the unreplayed cells are carried from (same layout)")
    ap.add_argument("--scope", default=None, help="the computed scope: <dir>/queue_<cls>.txt")
    # the landing tree's own rows (a candidate's replay/confirm): they take precedence over the frozen tree's row
    # for the same cell, so the tally judges the tree that lands, not the one the delta started from
    ap.add_argument("--overlay", default=None)
    ap.add_argument("--tree", required=True, help="the tree whose committed records name the carried reds and the judgments")
    ap.add_argument("--judgments", default=None, help="the committed per-output judgments record (default <tree>/docs/reference/artefact-judgments.jsonl)")
    ap.add_argument("--classes", default="32g")
    ap.add_argument("--modes", default="native,triton")
    return ap


def main(argv=None) -> int:
    return tally(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
