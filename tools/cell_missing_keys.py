#!/usr/bin/env python3
"""Every key a red gate cell lacks, in ONE run — a diagnosis, never a result (supervisor, 2026-10-24 05:42).

A certified-only cell stops at its FIRST missing key, so each gate named one key and hid the next behind it: three gates
in a row each found one more census defect (the MoE combine, the codec norm, the complex sum), and every defect cost a
whole census. This tool replays the cell's own command (read from its gate row: the exact request, the same engine tree,
the same card) with the engine's key record on (`NBX_KEY_RECORD`, the launch-time key of every autotuned launch, the
launcher's own key function) and WITHOUT `--certified-only`, so the run goes on past a miss; then it classifies every key
the run formed against the tree's committed census table and certified directory for the card's memory class:

  ABSENT        not in the census table: a census defect to fix in the tool
  UNCERTIFIED   in the table, not certified for this class: a certification gap

The replay never touches what the engine serves: its outputs go to `--out`, the autotuner's replay cache is redirected
there (`NEUROBRIX_REPLAY_CACHE`) and removed after, nothing is written to the certified directory or the census table;
the row it read is not rewritten. Its artefact is never judged, never a gate cell: the tool says so in its report.

    cell_missing_keys.py --rows <gate>/matrix_32g --model Wan2.1-T2V-1.3B --mode triton --card 3 --out <campaign dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atelier import writable  # noqa: E402


def the_row(rows: Path, model: str, mode: str) -> dict:
    """The cell's LAST row in the gate's matrix (a replay appends): refused by name when there is none."""
    files = sorted(rows.glob("rows_card*.jsonl")) if rows.is_dir() else [rows]
    if not files:
        raise SystemExit(f"--rows {rows}: no rows_card*.jsonl — not a gate matrix")
    found = []
    for f in files:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("model") == model and r.get("mode") == mode:
                found.append(r)
    if not found:
        raise SystemExit(f"no row of {model} {mode} in {rows}")
    return sorted(found, key=lambda r: r.get("date") or "")[-1]


def diagnostic_command(row: dict, out: Path) -> list:
    """The row's command without `--certified-only`, its `--output` moved into `out`."""
    argv = shlex.split(row["command"])
    if "--certified-only" not in argv:
        raise SystemExit("the row's command is not a certified-only run: nothing to diagnose")
    argv = [a for a in argv if a != "--certified-only"]
    if "--output" in argv:
        i = argv.index("--output")
        argv[i + 1] = str(out / ("diagnostic_" + Path(argv[i + 1]).name))
    return argv


def classify(lines, tree: Path, memory_class: int, vendor: str, profile: str) -> dict:
    """{"absent": [...], "uncertified": [...], "served": n} for the recorded `<qual>::<key>` lines, read in the tree."""
    code = (
        "import json,sys\n"
        "from neurobrix.kernels import census_table as T, autotune_certified as C\n"
        "from neurobrix.kernels.autotune_certify import _read_file\n"
        "lines, cls, vendor, profile = json.loads(sys.stdin.read()), int(sys.argv[1]), sys.argv[2], sys.argv[3]\n"
        "out = {'absent': [], 'uncertified': [], 'served': 0}\n"
        "files = {}\n"
        "for ln in lines:\n"
        "    qual, key = ln.split('::', 1)\n"
        "    _path, rows = T.rows_for(qual, key, vendor, profile, cls)\n"
        "    if not rows:\n"
        "        out['absent'].append(ln); continue\n"
        "    if qual not in files:\n"
        "        d = C.file_for(vendor, profile, qual, 'x').parent\n"
        "        files[qual] = [_read_file(f) for f in sorted(d.glob(C.kernel_short(qual) + '.*.json'))]\n"
        "    if any(C.entry_covers(e, key, cls) for e in files[qual]):\n"
        "        out['served'] += 1\n"
        "    else:\n"
        "        out['uncertified'].append(ln)\n"
        "print(json.dumps(out))\n")
    p = subprocess.run([sys.executable, "-c", code, str(memory_class), vendor, profile], input=json.dumps(sorted(lines)), text=True,
                       capture_output=True, cwd=tree, env=dict(os.environ, PYTHONPATH=str(tree / "src")))
    if p.returncode:
        raise SystemExit(f"classifying the keys failed in {tree}: {p.stderr[-800:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", required=True, type=Path, help="the gate's matrix directory (or one rows_card*.jsonl)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, help="the cell's mode as its row names it (triton, native...)")
    ap.add_argument("--card", required=True, help="the card to replay on (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--class-gb", required=True, type=int, help="the card's memory class (16, 32): the table and "
                                                                 "certified class the keys are read against")
    ap.add_argument("--table", default="nvidia/volta", help="the vendor/profile the census table and the certified "
                                                           "directory are read for")
    ap.add_argument("--out", required=True, type=Path, help="where the diagnosis writes (an atelier root)")
    a = ap.parse_args(argv)
    row = the_row(a.rows, a.model, a.mode)
    tree = Path(row["engine_tree"])
    out = writable(a.out / f"{a.model}_{a.mode}")
    out.mkdir(parents=True, exist_ok=True)
    rec, replay = out / "keys.record", out / "replay_cache"
    rec.unlink(missing_ok=True)
    cmd = diagnostic_command(row, out)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.card), NBX_KEY_RECORD=str(rec),
               NEUROBRIX_REPLAY_CACHE=str(replay), PYTHONPATH=str(tree / "src"), PYTHONNOUSERSITE="1")
    env.pop("NBX_AUTOTUNE_CERTIFIED_ONLY", None)
    with open(out / "run.log", "w") as log:
        rc = subprocess.run(cmd, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    shutil.rmtree(replay, ignore_errors=True)           # what the replay swept is never kept
    lines = sorted({ln.strip() for ln in rec.read_text().splitlines() if ln.strip()}) if rec.exists() else []
    if not lines:
        raise SystemExit(f"the replay recorded no key (rc {rc}, log {out / 'run.log'}): nothing to classify")
    vendor, profile = a.table.split("/", 1)
    got = classify(lines, tree, a.class_gb, vendor, profile)
    report = {"model": a.model, "mode": a.mode, "card": a.card, "class_gb": a.class_gb, "engine": row.get("engine"),
              "tree": str(tree), "command": cmd, "rc": rc, "keys_formed": len(lines), **got,
              "note": "a DIAGNOSIS: its artefact is never judged and never a gate cell"}
    (out / "missing.json").write_text(json.dumps(report, indent=1))
    print(f"{a.model} {a.mode} card {a.card} ({a.class_gb} GB): rc {rc}, {len(lines)} key(s) formed, "
          f"{got['served']} certified, {len(got['uncertified'])} uncertified, {len(got['absent'])} ABSENT "
          f"from the census -> {out / 'missing.json'}")
    for ln in got["absent"]:
        print(f"  ABSENT       {ln}")
    for ln in got["uncertified"]:
        print(f"  UNCERTIFIED  {ln}")
    return 0 if rc == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
