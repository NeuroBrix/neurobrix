"""catalog_manifest.py [--check]: docs/catalog/manifest.json, the catalogue the hub syncs from, built from the engine's
own records and nothing else (schema agreed with the hub, 2026-11-08 Q1; decided between agents 2026-11-09 17:18):

    {"schema": 1, "models": [{"slug", "family", "category", "container": {"sha256", "bytes"}, "neurotax", "licence",
                              "validation": [{"branch", "platform", "judged", "output_sha256"}]}]}

Sources: the catalogue (`docs/internal/catalogue.tsv`, header-named columns: hub slug, cache entry, NAS path, container
sha256), the container's own manifest (family, licence, neurotax; read in the engine's cache, never written), the NAS
container file (bytes), the committed verdict register (`docs/reference/artefact-judgments.jsonl`: per entry, mode and
machine, the latest row judged `works`) and `docs/reference/register-machines.tsv` (a register machine -> its platform).
No path, no machine name enters the file: one that would is refused by name. A catalogue entry whose container or
manifest is missing is refused by name, never skipped. `--check` rebuilds and compares with the committed file."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
OUT = REPO / "docs/catalog/manifest.json"
REGISTER = REPO / "docs/reference/artefact-judgments.jsonl"
MACHINES = REPO / "docs/reference/register-machines.tsv"
BRANCH = {"native": "pytorch", "triton": "triton"}        # the register's mode -> the engine's public branch name
COLS = {"slug": "hub slug", "entry": "cache entry", "nas": "NAS path (models/)", "sha": "container sha256 (NAS model.nbx)"}


def _tsv(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    return [h.strip() for h in rows[0]], [r for r in rows[1:] if any(c.strip() for c in r)]


def _catalogue():
    import atelier
    header, rows = _tsv(Path(atelier.root("communication")) / "catalogue.tsv")
    missing = [c for c in COLS.values() if c not in header]
    if missing:
        raise RuntimeError(f"ZERO FALLBACK: the catalogue has no column {missing} (header: {header})")
    idx = {k: header.index(c) for k, c in COLS.items()}
    return [{k: r[i].strip() for k, i in idx.items()} for r in rows]


def _platforms():
    header, rows = _tsv(MACHINES)
    return {r[header.index("machine")].strip(): r[header.index("platform")].strip() for r in rows}


def _validation(platforms):
    latest = {}
    for line in REGISTER.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sha = r.get("sha256")
        if r.get("verdict") != "works" or r.get("mode") not in BRANCH or not sha or sha == "None":
            continue
        if r["machine"] not in platforms:
            raise RuntimeError(f"ZERO FALLBACK: register machine {r['machine']!r} has no platform in {MACHINES.name}")
        k = (r["model"], r["mode"], r["machine"])
        date = str(r.get("run_date") or r.get("date") or "")
        if k not in latest or date >= latest[k][0]:
            latest[k] = (date, sha)
    out = {}
    for (entry, mode, machine), (date, sha) in sorted(latest.items()):
        out.setdefault(entry, []).append({"branch": BRANCH[mode], "platform": platforms[machine],
                                          "judged": date[:10], "output_sha256": sha})
    return out


def build() -> dict:
    import atelier
    cache, models = Path(atelier.root("cache")), Path(atelier.root("models"))
    platforms = _platforms()
    validation = _validation(platforms)
    out = []
    for c in _catalogue():
        man_path = cache / c["entry"] / "manifest.json"
        nbx = models / c["nas"] / "model.nbx"
        if not man_path.is_file() or not nbx.is_file():
            raise RuntimeError(f"ZERO FALLBACK: {c['slug']}: its container ({c['nas']}/model.nbx) or its manifest "
                               f"(cache entry {c['entry']}) is missing")
        man = json.loads(man_path.read_text())
        out.append({"slug": c["slug"], "family": man["family"], "category": c["nas"].split("/")[0],
                    "container": {"sha256": c["sha"], "bytes": nbx.stat().st_size},
                    "neurotax": man.get("neurotax_version"), "licence": man.get("license"),
                    "validation": validation.get(c["entry"], [])})
    doc = {"schema": 1, "models": sorted(out, key=lambda m: m["slug"].lower())}
    _refuse_leaks(doc, [str(cache), str(models)])
    return doc


def _refuse_leaks(doc, roots):
    text = json.dumps(doc)
    leaks = [s for s in (*roots, str(REPO), os.path.expanduser("~"), *_machines()) if s and s in text]
    if leaks:
        raise RuntimeError(f"ZERO FALLBACK: the manifest would carry {leaks}: no path or machine name goes public")


def _machines():
    header, rows = _tsv(MACHINES)
    return [f'"{r[header.index("machine")].strip()}"' for r in rows]


def main(argv) -> int:
    text = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    if "--check" in argv:
        same = OUT.is_file() and OUT.read_text() == text
        print("docs/catalog/manifest.json is current" if same else "docs/catalog/manifest.json is STALE: rebuild it")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote docs/catalog/manifest.json: {len(json.loads(text)['models'])} models")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
