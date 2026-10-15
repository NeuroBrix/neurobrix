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
a = ap.parse_args()
verdict = {}
for f in a.verdicts:
    for l in open(f):
        m = re.match(r"([0-9a-f]{12}) (works|broken) \S*\s*\|\s*(.*)$", l.strip())
        if m:
            verdict[m.group(1)] = (m.group(2), m.group(3))
runs = {}
for f in a.rows:
    for l in open(f):
        try: r = json.loads(l)
        except ValueError: continue
        if r.get("rc") == 0 and r.get("sha256"):
            s = r["sha256"]
            if s not in runs or r.get("date", "") < runs[s].get("date", ""):
                runs[s] = r
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
