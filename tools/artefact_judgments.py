#!/usr/bin/env python3
"""docs/reference/artefact-judgments.jsonl — one outside judgment per artefact a gate row points to (supervisor 2026-10-15
18:05: recovered, zero GPU). Each line pairs the artefact (sha256) with its run (model, mode, engine, run date, wall time,
request — from its earliest gate row) and the verdict looked at from outside (works / broken and what was seen).

    artefact_judgments.py --machine dell --verdicts V1.txt V2.txt ... --rows ROWS... --out FILE [--merge MAC.jsonl]
Verdict files hold `<sha12> <works|broken> cells=N | <what was seen>` (the judging campaigns' verdicts.txt); a later line
for the same artefact supersedes an earlier one. A row whose artefact no verdict covers is listed with verdict "not judged".
--merge adds another machine's lines (same schema) as they are."""
import argparse, json, re, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--machine", required=True)
ap.add_argument("--verdicts", nargs="+", required=True)
ap.add_argument("--rows", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--merge", default=None)
ap.add_argument("--matrix-judgments", nargs="*", default=[],
                help="judgments.jsonl files written beside gate rows (regression_matrix.py judge): each judgment is the "
                     "verdict of its cell's artefact — the latest row of that model and mode in the same directory at or "
                     "before the judgment's date")
a = ap.parse_args()
verdict = {}
for f in a.verdicts:
    for l in open(f):
        m = re.match(r"([0-9a-f]{12}) (works|broken) \S*\s*\|\s*(.*)$", l.strip())
        if m:
            verdict[m.group(1)] = (m.group(2), m.group(3))
def _utc(stamp):
    """A judgment's date ('2026-10-15 12:11 CEST', or ISO) as a UTC 'YYYY-MM-DDTHH:MM:SS' string, or None."""
    import datetime
    if not stamp:
        return None
    s = str(stamp).strip()
    offsets = {"CEST": 2, "CET": 1, "UTC": 0, "Z": 0}
    zone = next((z for z in offsets if s.endswith(z)), None)
    core = s[: -len(zone)].strip() if zone else s
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            d = datetime.datetime.strptime(core[:19], fmt)
            return (d - datetime.timedelta(hours=offsets.get(zone, 0))).strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    return None


by_dir = {}
runs = {}
for f in a.rows:
    for l in open(f):
        try: r = json.loads(l)
        except ValueError: continue
        if r.get("rc") == 0 and r.get("sha256"):
            by_dir.setdefault(str(Path(f).parent), []).append(r)
            s = r["sha256"]
            if s not in runs or r.get("date", "") < runs[s].get("date", ""):
                runs[s] = r
for jf in a.matrix_judgments:
    rows_here = by_dir.get(str(Path(jf).parent), [])
    for line in open(jf):
        try:
            j = json.loads(line)
        except ValueError:
            continue
        when = _utc(j.get("date"))
        cands = [r for r in rows_here if r["model"] == j.get("model") and r["mode"] == j.get("mode")
                 and (when is None or str(r.get("date", ""))[:19] <= when)]
        if not cands or j.get("verdict") not in ("works", "broken"):
            continue
        r = max(cands, key=lambda r: r.get("date", ""))
        verdict[r["sha256"][:12]] = (j["verdict"], str(j.get("judged", "")))
out = []
for s, r in sorted(runs.items(), key=lambda kv: (kv[1]["model"], kv[1]["mode"], kv[1].get("date", ""))):
    v, seen = verdict.get(s[:12], ("not judged", ""))
    req = re.sub(r"/\S+/", "", " ".join(str(x) for x in (r.get("request") or [])))
    out.append({"machine": a.machine, "sha256": s, "model": r["model"], "mode": r["mode"], "engine": str(r.get("engine", ""))[:9],
                "run_date": r.get("date", ""), "wall_s": r.get("wall_s"), "request": req, "verdict": v,
                "judged": re.sub(r"/\S+/", "", seen)})
if a.merge:
    for l in open(a.merge):
        l = l.strip()
        if l:
            out.append(json.loads(l))
text = "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
for bad in ("/home", "/Users", "/mnt", "campagnes", "nbx/campaigns"):
    if bad in text:
        sys.exit(f"REFUSED: a local path or campaign name ({bad}) would enter a committed file")
Path(a.out).write_text(text)
c = {}
for o in out:
    c[(o["machine"], o["verdict"])] = c.get((o["machine"], o["verdict"]), 0) + 1
print(a.out, len(out), "artefacts", c)
