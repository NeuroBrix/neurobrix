"""The Mac's model runs paired with the outside judgment written for their artefact (supervisor 2026-10-15 18:05; was
campagnes/2026_09_22_apple/scripts/artefact_inventory_2026_10_15.py, which globbed the campaign's results/). Sources: the
run register (`<registres>/runs/runs_mac.jsonl`, every closed campaign's harness rows) plus the harness rows of the
campaigns still open under `<campaigns>/`; the verdicts already written in every `judged_*.json` of the journals and of
the open campaigns. A run counts when it ended rc 0 with an artefact (the artefact may be gone with its closed campaign:
the row and its verdict are the record). Paths from atelier.json. usage: artefact_inventory.py <out.tsv>"""
import json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from atelier import root, writable  # noqa: E402

CAMP = root("campaigns"); REG = root("registres")
APPLE = CAMP / "2026_09_22_apple"   # the campaign the judged_*.json artefact paths are relative to


def row_files():
    yield REG / "runs" / "runs_mac.jsonl"
    yield from sorted(CAMP.glob("**/harness/rows_card*.jsonl"))


def judged_files():
    yield from sorted((REG / "journaux").glob("**/judged_*.json"))
    yield from sorted(CAMP.glob("**/judged_*.json"))


def judged_map():
    out = {}
    for f in judged_files():
        if not f.is_file():
            continue
        for v in json.load(open(f)).values():
            a = v.get("artifact")
            if a:
                out[a if a.startswith("/") else str(APPLE / a)] = v
    return out


KIND = {".png": "image", ".jpg": "image", ".mp4": "video", ".gif": "video", ".wav": "audio", ".flac": "audio", ".txt": "text"}

if __name__ == "__main__":
    judged = {a: v.get("image") for a, v in judged_map().items()}
    rows, seen = [], set()
    for f in row_files():
        if not f.is_file():
            continue
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            a = r.get("artefact")
            if str(r.get("rc")) != "0" or not a or a in seen:
                continue
            seen.add(a)
            req = r.get("request"); req = " ".join(req) if isinstance(req, list) else str(req or "")
            rows.append((KIND.get(Path(a).suffix.lower(), "other"), r["model"], r["mode"], str(r.get("engine", ""))[:8],
                         str(r.get("date", ""))[:16], str(r.get("wall_s", "")), judged.get(a, "") or "", a, req,
                         r.get("campaign") or str(f.parent)))
    out = writable(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("kind\tmodel\tmode\tengine\tdate\twall_s\tverdict_already\tartefact\trequest\tharness\n" +
                   "\n".join("\t".join(x) for x in rows) + "\n")
    print(len(rows), "artefacts;", Counter((x[0], bool(x[6])) for x in rows))
