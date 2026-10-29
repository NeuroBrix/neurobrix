#!/usr/bin/env python3
"""Model status, one line per catalogue model, for a hardware profile with memory classes (the Dell: `nvidia/volta`,
16g and 32g) — the NVIDIA half of docs/reference/model-status.md (the Mac's Apple table), same columns.

Every cell is READ from an artefact and says how; a cell with no artefact says `not measured` / `no certified-only
cell` / `not judged` — never a number:
    container     the cache entry's manifest (nbx version, trace date, NeuroTax version, the manifest's sha256[:12])
    census        rows of the committed census table per class (`derived`: the census is a derivation)
    keys served   census keys whose certified entry covers the class (`entry_for_memory_gb`), over the census rows
    native/triton the LATEST certified-only gate row per (model, mode, class): PASS = rc 0 and zero miss, else ERROR
    R29 verdict   the outside judgment of that row's artefact, by sha256 (artefact-judgments.jsonl); `not judged`
                  when no judgment names the sha
    status        derived from the above only: runs on both branches + judged works; request-limited when the
                  judgment says so (the gate floor's request forms no judgeable picture); judgment owed; red/missing

    model_status.py --tree <repo> --cache <dir> --catalogue <public tsv> --catalogue-full <internal tsv>
                    --rows <rows_card*.jsonl ...> --judgments <jsonl> --profile nvidia/volta --classes 16g=16,32g=32
                    --class-of-card 0=16g,1=16g,2=32g,3=32g --out <md>
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path


def load_rows(paths, class_of_card):
    """The latest row per (model, mode, class) over every rows file given; a row's class is its card's."""
    rows = {}
    for f in paths:
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            cls = class_of_card.get(str(r.get("gpu")))
            if cls is None:
                continue
            k = (r["model"], r["mode"], cls)
            if k not in rows or r.get("date", "") >= rows[k].get("date", ""):
                rows[k] = r
    return rows


def served_keys(census_rows, files, cls_gb, C):
    """(served, total): census keys whose certified entry covers the memory class."""
    n = 0
    for r in census_rows:
        key = r["key"]
        dts = [k for k in eval(key) if isinstance(k, str)]       # the key's dtype names (C.key_dtypes' table)
        dt = dts[-1] if dts else "any"
        ent = None
        for cand in [dt] + [d for d in ("fp16", "fp32", "bf16", "any") if d != dt]:
            e = files.get((r["kernel"], cand))
            if e and key in e:
                ent = e[key]
                break
        if ent is not None and C.entry_for_memory_gb(ent, cls_gb) is not None:
            n += 1
    return n, len(census_rows)


def cell_text(r):
    if r is None:
        return "no certified-only cell"
    ok = r.get("rc") == 0 and int(r.get("misses") or 0) == 0
    return f"{'PASS' if ok else 'ERROR'} {str(r.get('date', ''))[:10]} (gate {str(r.get('engine', ''))[:8]}, {float(r.get('wall_s') or 0):.0f} s)"


def judgment_text(r, judgments):
    if r is None or not r.get("sha256"):
        return "not judged"
    j = judgments.get(r["sha256"])
    if j is None or j.get("verdict") == "not judged":
        return "not judged"
    seen = str(j.get("judged", ""))[:110]
    return f"{'PASS' if j['verdict'] == 'works' else 'FAIL'} {str(j.get('run_date', ''))[:10]} ({seen})"


def status_of(cells, verdicts):
    present = [v for v in cells.values() if not v.startswith("no certified-only cell")]
    if not present or any(v.startswith("ERROR") for v in present):
        return "a cell is red or missing"
    if all(v.startswith("PASS") for v in verdicts.values()):
        return "runs on both branches, judged from outside"
    if any("request-limited" in v for v in verdicts.values()):
        return "runs on both branches; judged request-limited at the gate floor (a judgeable request is owed)"
    return "runs on both branches; judgment owed"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", required=True, type=Path, help="the repository whose census tables and certified directory are read")
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--catalogue", required=True, type=Path, help="the public table (unique name, repository, hub slug)")
    ap.add_argument("--catalogue-full", required=True, type=Path, help="the machine's table (cache entry per unique name)")
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--judgments", required=True, type=Path)
    ap.add_argument("--profile", required=True, help="vendor/profile, e.g. nvidia/volta")
    ap.add_argument("--classes", required=True, help="name=GB per memory class, e.g. 16g=16,32g=32")
    ap.add_argument("--class-of-card", required=True, help="card ordinal=class, e.g. 0=16g,1=16g,2=32g,3=32g")
    ap.add_argument("--machine", required=True, help="the heading's machine, e.g. 'NVIDIA V100 (nvidia/volta, 16 GB and 32 GB classes)'")
    ap.add_argument("--vendor-oracle", nargs="*", default=[], help="unique name=text of a vendor comparison with its artefact")
    ap.add_argument("--engine-src", type=Path, default=None, help="the engine package to import the certified reader from (default: the tree's src)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(a.engine_src or (a.tree / "src")))
    import neurobrix  # noqa: F401 — the engine provides nbxtensor's host services (neurobrix/_nbxtensor_host.py)
    from nbxtensor.ops import autotune_certified as C
    classes = dict(kv.split("=") for kv in a.classes.split(","))
    class_of_card = dict(kv.split("=") for kv in a.class_of_card.split(","))
    vendor = dict(kv.split("=", 1) for kv in a.vendor_oracle)
    cat = list(csv.DictReader(open(a.catalogue), delimiter="\t"))
    full = list(csv.DictReader(open(a.catalogue_full), delimiter="\t"))
    name_col = next(c for c in full[0] if c.startswith("unique name"))
    entry_of = {r[name_col]: r["cache entry"] for r in full}
    census = {cls: defaultdict(list) for cls in classes}
    for cls in classes:
        p = a.tree / "src/neurobrix/config/census" / a.profile / f"{cls}.jsonl"
        if not p.exists():
            raise SystemExit(f"ERROR: no census table at {p}")
        for l in open(p):
            r = json.loads(l)
            census[cls][r["model"]].append(r)
    files = {}
    for f in glob.glob(str(a.tree / "src/neurobrix/config/autotune" / a.profile / "*.json")):
        j = json.load(open(f))
        files[(j["kernel"], j["dtype"])] = j["entries"]
    rows = load_rows(a.rows, class_of_card)
    judgments = {}
    for l in open(a.judgments):
        j = json.loads(l)
        judgments[j["sha256"]] = j
    names = list(classes)
    out = [f"# Model status — {a.machine}\n",
           f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by `tools/model_status.py` from the artefacts named in each column: the committed "
           f"census tables and certified directory of the tree, the latest certified-only gate row per cell and memory class, the outside "
           f"judgments by artefact sha256, the catalogue table. A cell that says *not measured*, *no certified-only cell* or *not judged* has no "
           f"artefact — it is the queue, never a number.\n",
           "| model | Hugging Face repo | container | census (" + " / ".join(names) + " rows) | keys served / in table (" + "; ".join(names) + ") | vendor oracle | "
           + " | ".join(f"{m} {c} (certified-only cell)" for m in ("native", "triton") for c in names) + " | R29 verdict per branch (outside instrument, artefact) | hub slug | status |",
           "|" + "---|" * (8 + 2 * len(names))]
    counts = defaultdict(int)
    for r in cat:
        name = r["unique name"]; entry = entry_of.get(name)
        if entry is None or not (a.cache / entry / "manifest.json").exists():
            raise SystemExit(f"ERROR: {name}: no cache entry in the catalogue's full table or no manifest at {a.cache}/{entry}")
        mpath = a.cache / entry / "manifest.json"; m = json.loads(mpath.read_text())
        cont = (f"nbx {m.get('nbx_version', '?')}, traced {str(m.get('created_at', ''))[:10]}, NeuroTax {m.get('neurotax_version', '?')}, "
                f"manifest {hashlib.sha256(mpath.read_bytes()).hexdigest()[:12]}")
        cen = "derived: " + " / ".join(str(len(census[c][entry])) for c in names)
        keys = "; ".join("%d / %d" % served_keys(census[c][entry], files, int(classes[c]), C) for c in names)
        cells, verdicts = {}, {}
        for mode in ("native", "triton"):
            for c in names:
                cells[(mode, c)] = cell_text(rows.get((entry, mode, c)))
            judged_row = next((rows.get((entry, mode, c)) for c in reversed(names) if rows.get((entry, mode, c))), None)
            verdicts[mode] = judgment_text(judged_row, judgments)
        st = status_of(cells, verdicts); counts[st] += 1
        out.append("| `" + name + "` | " + r["repository (HF identifier or vendor)"] + " | " + cont + " | " + cen + " | " + keys + " | "
                   + vendor.get(name, "not measured") + " | " + " | ".join(cells[(mm, c)] for mm in ("native", "triton") for c in names)
                   + " | " + "; ".join(f"{mm}: {v}" for mm, v in verdicts.items()) + " | " + r["hub slug"] + " | " + st + " |")
    out.insert(3, "Status over the " + str(len(cat)) + " models: " + ", ".join(f"**{v} {k}**" for k, v in sorted(counts.items(), key=lambda x: -x[1])) + ".\n")
    a.out.write_text("\n".join(out) + "\n")
    print(f"{len(cat)} rows -> {a.out}; " + "; ".join(f"{v} {k}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
