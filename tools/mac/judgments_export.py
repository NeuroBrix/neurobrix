"""The Mac's outside judgments in the judgments.jsonl schema (class, date, judged, mode, model, source, verdict) plus the
run they judge (was campagnes/2026_09_22_apple/scripts/judgments_mac_export_2026_10_15.py). Sources: the 2026-10-15
recovered judgments (`judgments_recovered_2026_10_15.input.json`, in the journals) and every `judged_*.json`
(PASS -> works, FAIL -> broken, OPEN left out and counted), matched on the inventory's artefacts (artefact_inventory.py).
Paths from atelier.json. usage: judgments_export.py <inventory.tsv> <out.jsonl>"""
import csv, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from atelier import root, writable  # noqa: E402
from artefact_inventory import judged_map  # noqa: E402

REG = root("registres")
rec = [f for f in (REG / "journaux").glob("**/judgments_recovered_2026_10_15.input.json") if f.is_file()]
today = json.load(open(rec[0])) if rec else {}
earlier = judged_map()
out, n_open = [], 0
for r in csv.DictReader(open(sys.argv[1]), delimiter="\t"):
    a = r["artefact"]
    if a in today:
        verdict, judged, when = today[a]["verdict"], today[a]["judged"], time.strftime("%Y-%m-%d %H:%M %Z")
    elif a in earlier:
        e = earlier[a]; im = e.get("image")
        if im not in ("PASS", "FAIL"):
            n_open += 1; continue
        verdict, judged, when = ("works" if im == "PASS" else "broken"), f"{e.get('instrument', 'looked at')}: {e.get('what', '')}", e.get("looked_at", "")
    else:
        continue
    out.append({"class": "18g", "date": when, "judged": judged, "mode": r["mode"], "model": r["model"], "source": "mac",
                "verdict": verdict, "run_date": r["date"], "engine": r["engine"], "wall_s": r["wall_s"], "request": r["request"],
                "artefact": a.split("campagnes/")[-1]})
(lambda o: (o.parent.mkdir(parents=True, exist_ok=True), o))(writable(sys.argv[2]))[1].write_text("".join(json.dumps(x) + "\n" for x in out))
print(len(out), "judgments,", sum(x["verdict"] == "broken" for x in out), "broken,", n_open, "OPEN left out")
