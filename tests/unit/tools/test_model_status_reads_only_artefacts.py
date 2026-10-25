"""`tools/model_status.py`: every cell of the status table is read from an artefact; a cell without one says so.

What this test would do if the code were wrong: a cell PASS without a gate row, or a judged verdict without a judgment
naming the artefact's sha, would pass a table typed by its author — the assertions on `no certified-only cell`, `not
judged` and the ERROR of a row with a miss fail. Injection (a row with one miss read as PASS) seen red on 2026-10-25.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _artefacts(tmp):
    tree = tmp / "tree"; (tree / "src/neurobrix/config/census/nvidia/volta").mkdir(parents=True)
    (tree / "src/neurobrix/config/autotune/nvidia/volta").mkdir(parents=True)
    key = "(64, 32, 'fp16', 'fp16', 'fp16')"
    for cls in ("16g", "32g"):
        (tree / f"src/neurobrix/config/census/nvidia/volta/{cls}.jsonl").write_text(
            json.dumps({"model": "m1", "mode": "triton", "kernel": "k.mm_kernel", "key": key, "dtype": "fp16", "container": "c", "ops": "[]", "rungs_mb": "None", "tool": "t"}) + "\n")
    proof = {"machine": {"device": {"ordinal": 0, "visible_devices": "0", "name": "V100", "memory_mb": 16384}}}
    (tree / "src/neurobrix/config/autotune/nvidia/volta/mm_kernel.fp16.json").write_text(json.dumps(
        {"format": "nbx-autotune-certified/2", "vendor": "nvidia", "profile": "volta", "kernel": "k.mm_kernel", "dtype": "fp16",
         "entries": {key: {"config": {"kwargs": {}}, "excluded": [], "proof": proof, "variants": {}}}}))
    cache = tmp / "cache"; (cache / "m1").mkdir(parents=True)
    (cache / "m1/manifest.json").write_text(json.dumps({"nbx_version": "0.1", "created_at": "2026-10-01T00:00:00", "neurotax_version": "5.0"}))
    (tmp / "cat.tsv").write_text("repository (HF identifier or vendor)\tunique name\thub slug\nv/m1\tv/m1\tv/m1\n")
    (tmp / "full.tsv").write_text("unique name (HF or vendor repository[:weight file])\tcache entry\nv/m1\tm1\n")
    rows = [{"model": "m1", "mode": "native", "gpu": "0", "rc": 0, "misses": 0, "date": "2026-10-20T00:00:00Z", "engine": "abcdef12", "wall_s": 3.0, "sha256": "aa" * 32},
            {"model": "m1", "mode": "triton", "gpu": "2", "rc": 0, "misses": 1, "date": "2026-10-20T00:00:00Z", "engine": "abcdef12", "wall_s": 4.0, "sha256": "bb" * 32}]
    (tmp / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp / "judg.jsonl").write_text(json.dumps({"sha256": "aa" * 32, "verdict": "works", "judged": "looked at", "run_date": "2026-10-20T00:00:00Z"}) + "\n")
    return tree, cache


def _run(tmp, tree, cache):
    out = tmp / "status.md"
    r = subprocess.run([sys.executable, str(REPO / "tools/model_status.py"), "--tree", str(tree), "--cache", str(cache),
                        "--catalogue", str(tmp / "cat.tsv"), "--catalogue-full", str(tmp / "full.tsv"), "--rows", str(tmp / "rows.jsonl"),
                        "--judgments", str(tmp / "judg.jsonl"), "--profile", "nvidia/volta", "--classes", "16g=16,32g=32",
                        "--class-of-card", "0=16g,1=16g,2=32g,3=32g", "--machine", "test", "--engine-src", str(REPO / "src"), "--out", str(out)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    return out.read_text()


def test_every_cell_is_read_from_an_artefact_or_says_it_has_none(tmp_path):
    tree, cache = _artefacts(tmp_path)
    t = _run(tmp_path, tree, cache)
    row = next(l for l in t.splitlines() if l.startswith("| `v/m1`"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    # container, census, keys served (the 16g entry serves 16g only: 1/1; 32g: 0/1)
    assert cells[2].startswith("nbx 0.1, traced 2026-10-01, NeuroTax 5.0, manifest ") and cells[3] == "derived: 1 / 1" and cells[4] == "1 / 1; 0 / 1"
    assert cells[5] == "not measured"
    assert cells[6].startswith("PASS 2026-10-20 (gate abcdef12, 3 s)") and cells[7] == "no certified-only cell"       # native 16g / 32g
    assert cells[8] == "no certified-only cell" and cells[9].startswith("ERROR 2026-10-20")                           # triton: one miss
    assert "native: PASS 2026-10-20 (looked at)" in cells[10] and "triton: not judged" in cells[10]
    assert cells[12] == "a cell is red or missing"
