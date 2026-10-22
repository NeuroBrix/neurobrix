#!/bin/bash
# The Mac's section of docs/reference/benchmarks.md, re-derived from EVERY Mac run ever recorded (was
# campagnes/2026_09_22_apple/scripts/benchmarks_regen_mac.sh): (1) the artefact inventory and the Mac's judgments from the
# run register and the open campaigns (tools/mac/artefact_inventory.py, judgments_export.py); (2) the tree's
# docs/reference/artefact-judgments.jsonl keeps every other machine's line and takes the Mac's; (3) tools/benchmarks_md.py
# replaces the Mac section in place. Zero GPU. A run without an outside judgment prints "not judged", never green.
# Paths from atelier.json; the tree must be the repo or a worktree. usage: benchmarks_regen.sh <tree>
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd); AT() { python3 "$HERE/atelier.py" "$@" || exit 2; }
T=$(AT writable "$1"); V=$(AT python engine); W=$(AT tmp benchmarks_regen.$$); mkdir -p "$W"
$V "$HERE/mac/artefact_inventory.py" "$W/inventory.tsv"
$V "$HERE/mac/judgments_export.py" "$W/inventory.tsv" "$W/judgments_mac.jsonl"
J=$T/docs/reference/artefact-judgments.jsonl
$V - "$J" $W/judgments_mac.jsonl <<'PY'
import hashlib, json, sys
from pathlib import Path
j, mac = sys.argv[1], sys.argv[2]
keep = [l for l in open(j) if l.strip() and json.loads(l).get("machine") != "mac"]
new = []
for l in open(mac):
    if not l.strip():
        continue
    r = json.loads(l); a = Path(r.get("artefact") or "")
    w = r.get("wall_s")
    new.append(json.dumps({"machine": "mac", "sha256": hashlib.sha256(a.read_bytes()).hexdigest() if a.is_file() else None,
        "model": r["model"], "mode": r["mode"], "engine": r.get("engine"), "run_date": r.get("run_date"),
        "wall_s": float(w) if w not in (None, "") else None, "request": r.get("request"), "verdict": r["verdict"],
        "judged": r["judged"]}, ensure_ascii=False) + "\n")
open(j, "w").write("".join(keep + new)); print(f"{j}: {len(keep)} other lines kept, {len(new)} Mac lines")
PY
# the register's rows and the open campaigns' rows, each run once (a closed campaign's rows are both until it is removed)
python3 - "$W/rows.jsonl" "$(AT root registres)/runs/runs_mac.jsonl" $(find "$(AT root campaigns)" -path '*/harness/rows_card*.jsonl' | sort) <<'PY'
import json, sys
seen, out = set(), []
for f in sys.argv[2:]:
    try:
        lines = open(f).read().splitlines()
    except FileNotFoundError:
        continue
    for l in lines:
        if not l.strip():
            continue
        r = json.loads(l)
        k = (r.get("model"), r.get("mode"), r.get("date"), r.get("artefact"), r.get("command"))
        if k not in seen:
            seen.add(k); out.append(l)
open(sys.argv[1], "w").write("\n".join(out) + "\n"); print(len(out), "runs")
PY
ROWS="$W/rows.jsonl"
$V $T/tools/benchmarks_md.py --machine 'Mac M4 Pro 24 GB (Metal)' --card-class 0=18g --judgments $J --append --out $T/docs/reference/benchmarks.md $ROWS
rm -rf "$W"
