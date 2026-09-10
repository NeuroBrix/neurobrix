#!/usr/bin/env python3
"""Which timed rows were measured beside a neighbour, and are therefore not measurements.

WHY THIS EXISTS
---------------
`tools/rig_scheduler.py` granted a timed job the machine but did not HOLD it:
the test gating the untimed dispatch asks whether a timed job is ALREADY
running, so once one was, the branch was skipped and the loop below started
untimed jobs on the free cards at the next poll. Fixed 2026-09-09 in `7759b3f`.

Every timed row the scheduler recorded BEFORE that commit is suspect, and
suspicion is not a verdict: this tool reads the scheduler's own logs and says,
per row, whether another job was actually running inside its window. A row with
no neighbour is clean and stays; a row with one is not a measurement and is
re-measured.

The cost is not academic. The Voxtral rate row of 2026-09-09 12:25 ran its full
12 minutes inside a 30 GB byte gate's 78 and reported 7.869 tok/s over a 46 %
spread; re-measured under the fixed scheduler it reports 9.457 tok/s over 1.1 %.
The contaminated value understated throughput by 20 % — a wrong number in an
internal table is worse than no number, because it aims the next lever.

WHAT IT READS
-------------
The scheduler's log lines, which carry their own timeline:

    HH:MM:SS TIMED  <name> on card N, machine exclusive
    HH:MM:SS start  <name> on card(s) N (W GB declared)
    HH:MM:SS done   <name> (rc=R, M min, card N)
    HH:MM:SS yield  <name> -> the timed campaign '<other>' is ready

A timed row's window runs from its TIMED line to its done line. Any other job
whose [start, done] interval intersects that window is a neighbour.

BEYOND THE SCHEDULER (--flightrec)
----------------------------------
The scheduler's ledger only covers rows the scheduler ran. A timed row launched
by hand, in parallel, is exposed to the same hazard without appearing in any
scheduler log — and the standing instruction on this rig WAS to run cards in
parallel by default. The flight recorder is the rig-wide ledger that can see
those: every run carries `started_iso`, `ended_iso` and the cards it declared.

A record is treated as a TIMED measurement on three explicit markers, never on a
guess about its name:

    `--lock-clock` in the command   — a locked clock exists only to time something
    `bench_row.py` in the command   — the timed row harness
    `--cold-arms` in the command    — the campaign measuring cold starts

Overlap is computed on the wall-clock intervals. Two runs that overlap are
reported with the cards each declared: a neighbour on ANOTHER card still moves
the number (that is the whole finding), so a different card is not a defence.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

TIMED = re.compile(r"^(\d\d:\d\d:\d\d)\s+TIMED\s+(.+?) on card (\d+), machine exclusive")
START = re.compile(r"^(\d\d:\d\d:\d\d)\s+start\s+(.+?) on card\(s\) ([\d,]+)")
DONE = re.compile(r"^(\d\d:\d\d:\d\d)\s+done\s+(.+?) \(rc=(-?\d+),")
YIELD = re.compile(r"^(\d\d:\d\d:\d\d)\s+yield\s+(.+?) ->")


def _t(s: str) -> datetime:
    return datetime.strptime(s, "%H:%M:%S")


def audit_log(path: str) -> list:
    """Every timed row in one log, with the jobs that overlapped it."""
    starts, dones, timed = [], [], []
    for line in Path(path).read_text(errors="replace").splitlines():
        m = TIMED.match(line)
        if m:
            timed.append({"at": _t(m.group(1)), "name": m.group(2).strip(),
                          "card": int(m.group(3))})
            continue
        m = START.match(line)
        if m:
            starts.append({"at": _t(m.group(1)), "name": m.group(2).strip(),
                           "cards": m.group(3)})
            continue
        m = DONE.match(line)
        if m:
            dones.append({"at": _t(m.group(1)), "name": m.group(2).strip(),
                          "rc": int(m.group(3))})
            continue
        m = YIELD.match(line)
        if m:
            dones.append({"at": _t(m.group(1)), "name": m.group(2).strip(), "rc": None})

    def ended(name: str, after: datetime):
        for d in dones:
            if d["name"] == name and d["at"] >= after:
                return d["at"]
        return None

    rows = []
    for t in timed:
        real_end = ended(t["name"], t["at"])
        # A row with no `done` line never finished — the scheduler was killed
        # under it. It produced no artifact, so it is not a measurement to
        # replace; inventing an end time for it would overstate the damage.
        killed = real_end is None
        end = real_end or (t["at"] + timedelta(hours=1))
        neighbours = []
        for s in starts:
            if s["name"] == t["name"]:
                continue
            s_end = ended(s["name"], s["at"]) or (s["at"] + timedelta(hours=1))
            # intervals intersect
            if s["at"] < end and s_end > t["at"]:
                overlap = (min(s_end, end) - max(s["at"], t["at"])).total_seconds()
                neighbours.append({
                    "name": s["name"], "cards": s["cards"],
                    "started": s["at"].strftime("%H:%M:%S"),
                    "ended": s_end.strftime("%H:%M:%S"),
                    "overlap_s": int(overlap),
                    "covers_whole_row": s["at"] <= t["at"] and s_end >= end,
                })
        rows.append({
            "source": "scheduler",
            "log": path,
            "row": t["name"], "card": t["card"], "gpu": str(t["card"]),
            "window": ("from " + t["at"].strftime("%H:%M:%S") + ", never finished"
                       if killed else
                       f"{t['at'].strftime('%H:%M:%S')}-{end.strftime('%H:%M:%S')}"),
            "duration_s": None if killed else int((end - t["at"]).total_seconds()),
            "neighbours": neighbours,
            "verdict": ("KILLED-NO-ARTIFACT" if killed else
                        "NOT-A-MEASUREMENT" if neighbours else "CLEAN"),
        })
    return rows


TIMED_MARKERS = ("--lock-clock", "bench_row.py", "--cold-arms")


def audit_flightrec(rec_dir: str, before_iso: str = "") -> list:
    """Every timed run in the recorder's ledger, with the runs that overlapped it."""
    recs = []
    for f in glob.glob(str(Path(rec_dir) / "*.json")):
        if Path(f).name == "fsck.json":
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not d.get("started_iso") or not d.get("cmd"):
            continue
        recs.append(d)

    def span(d):
        try:
            a = datetime.fromisoformat(d["started_iso"])
        except Exception:
            return None
        b = None
        if d.get("ended_iso"):
            try:
                b = datetime.fromisoformat(d["ended_iso"])
            except Exception:
                b = None
        return a, b

    rows = []
    for d in recs:
        cmd = d.get("cmd") or ""
        markers = [m for m in TIMED_MARKERS if m in cmd]
        if not markers:
            continue
        sp = span(d)
        if not sp:
            continue
        a, b = sp
        if before_iso and d["started_iso"] >= before_iso:
            continue
        neighbours = []
        for o in recs:
            if o.get("id") == d.get("id"):
                continue
            osp = span(o)
            if not osp:
                continue
            oa, ob = osp
            # an unfinished neighbour is treated as ending at its own start:
            # claiming an unbounded run would manufacture overlaps
            ob_eff = ob or oa
            b_eff = b or a
            if oa <= b_eff and ob_eff >= a:
                neighbours.append({
                    "id": o.get("id"), "label": o.get("label"),
                    "gpu": o.get("gpu"), "started": o.get("started_iso"),
                    "ended": o.get("ended_iso"),
                })
        dur = int((b - a).total_seconds()) if b else None
        # A run that started and ended in the same second measured nothing — it
        # failed at launch. Counting it as a damaged measurement would inflate
        # the damage with rows that never produced a number.
        took_a_measurement = dur is None or dur > 1
        verdict = ("NO-MEASUREMENT-TAKEN" if not took_a_measurement else
                   "NOT-A-MEASUREMENT" if neighbours else "CLEAN")
        rows.append({
            "source": "flightrec",
            "row": d.get("label") or d.get("id"),
            "id": d.get("id"),
            "gpu": d.get("gpu"),
            "markers": markers,
            "window": f"{d.get('started_iso')} -> {d.get('ended_iso') or 'never finished'}",
            "duration_s": dur,
            "status": d.get("status"),
            "neighbours": neighbours if took_a_measurement else [],
            "verdict": verdict,
        })
    return sorted(rows, key=lambda r: r["window"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="*", default=None,
                    help="scheduler logs (default: every scheduler*.log under validation_outputs)")
    ap.add_argument("--out", default=None, help="write the inventory as JSON")
    ap.add_argument("--flightrec", default=None, metavar="DIR",
                    help="also audit the rig-wide recorder ledger (e.g. .flightrec)")
    ap.add_argument("--verbose", action="store_true",
                    help="print clean rows too (default: only the damaged ones)")
    ap.add_argument("--before", default="", metavar="ISO",
                    help="only recorder runs started before this ISO timestamp "
                         "(e.g. the fix's commit time)")
    a = ap.parse_args()

    logs = a.logs or sorted(set(glob.glob("validation_outputs/**/scheduler*.log",
                                          recursive=True)))
    if not logs:
        print("no scheduler log found — nothing to audit", file=sys.stderr)
        return 1

    rows = []
    for lg in logs:
        rows.extend(audit_log(lg))
    if a.flightrec:
        fr = audit_flightrec(a.flightrec, a.before)
        print(f"--- recorder ledger: {len(fr)} timed run(s)"
              + (f" started before {a.before}" if a.before else "") + " ---")
        dirty_fr = [r for r in fr if r["verdict"] == "NOT-A-MEASUREMENT"]
        nomeas = [r for r in fr if r["verdict"] == "NO-MEASUREMENT-TAKEN"]
        for r in (fr if a.verbose else dirty_fr):
            mark = {"CLEAN": "✓", "NO-MEASUREMENT-TAKEN": "–"}.get(r["verdict"], "✗")
            dur = "" if r["duration_s"] is None else f", {r['duration_s']}s"
            print(f"  {mark} {r['verdict']:20s} {str(r['row'])[:74]}")
            print(f"      {r['window']}{dur}  cards {r['gpu']}")
            for n in r["neighbours"][:3]:
                print(f"      beside: {str(n['label'])[:64]}  cards {n['gpu']}")
            if len(r["neighbours"]) > 3:
                print(f"      ... and {len(r['neighbours'])-3} more")
        print(f"\n  recorder: {len(fr)} timed run(s) — "
              f"{len([r for r in fr if r['verdict']=='CLEAN'])} clean, "
              f"{len(dirty_fr)} measured beside a neighbour, "
              f"{len(nomeas)} produced no measurement\n")
        rows.extend(fr)

    dirty = [r for r in rows if r["verdict"] == "NOT-A-MEASUREMENT"]
    for r in [x for x in rows if x.get("source") == "scheduler"]:
        mark = {"CLEAN": "✓", "KILLED-NO-ARTIFACT": "–"}.get(r["verdict"], "✗")
        print(f"  {mark} {r['verdict']:18s} {r['row']}")
        dur = "" if r["duration_s"] is None else f" ({r['duration_s']}s)"
        print(f"      window {r['window']}{dur} on card {r['card']}"
              f"  ·  {Path(r['log']).name}")
        for n in r["neighbours"]:
            whole = " — covers the WHOLE row" if n["covers_whole_row"] else ""
            print(f"      beside: {n['name']} on card(s) {n['cards']}, "
                  f"{n['started']}-{n['ended']}, overlap {n['overlap_s']}s{whole}")

    sched = [r for r in rows if r.get("source") == "scheduler"]
    killed = [r for r in sched if r["verdict"] == "KILLED-NO-ARTIFACT"]
    clean = [r for r in sched if r["verdict"] == "CLEAN"]
    dirty = [r for r in sched if r["verdict"] == "NOT-A-MEASUREMENT"]
    print(f"\nscheduler ledger: {len(sched)} timed row(s): {len(clean)} clean, "
          f"{len(dirty)} measured beside a neighbour, "
          f"{len(killed)} killed before producing an artifact")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))
        print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
