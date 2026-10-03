"""tools/model_status.py: one row per container of the catalogue, every cell read from a record.

What would this file do if the code were wrong? A container left out of the table -> the first test
RED; an empty catalogue or a note naming no container accepted -> the refusals, RED; the certified
count read without the memory class -> the coverage case, RED.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
import model_status as MS  # noqa: E402

DER = ("A triton | walked 2 derived 2: reproduced 2, missed 0, extra 0 | walked-with-op 9: reproduced 9, missed 0 | not-yet 0 \n"
       "A triton-sequential | walked 2 derived 2: reproduced 2, missed 0, extra 0 | walked-with-op 9: reproduced 9, missed 0 | not-yet 0 \n"
       "B triton | walked 3 derived 1: reproduced 1, missed 2, extra 0 | walked-with-op 9: reproduced 3, missed 6 | not-yet 1 \n")
KEY = "(64, 64, 64, 'fp32')"


def _world(tmp_path, names=("A", "B", "C"), notes=None):
    cache = tmp_path / "cache"
    for n in names:
        (cache / n).mkdir(parents=True)
        (cache / n / "manifest.json").write_text(json.dumps({"family": "llm", "nbx_version": "0.1",
                                                              "neurotax_version": "5.0", "created_at": "2026-09-26T00:00"}))
    rc = tmp_path / "rc"
    t = rc / "src/neurobrix/config/census/nvidia/volta"
    t.mkdir(parents=True)
    row = {"model": "A", "kernel": "neurobrix.kernels.ops.matmul.matmul_kernel", "key": KEY}
    for cls in (16, 32):
        (t / f"{cls}g.jsonl").write_text(json.dumps(row) + "\n")
    d = rc / "src/neurobrix/config/autotune/nvidia/volta"
    d.mkdir(parents=True)
    cert = {"config": {}, "proof": {"machine": {"device": {"memory_mb": 16384}}}}
    (d / "matmul_kernel.fp32.json").write_text(json.dumps({"entries": {KEY: cert}}))
    (tmp_path / "der.txt").write_text(DER)
    (tmp_path / "reg.yml").write_text("llm:\n  A:\n    hf_repo: o/A\n  B:\n    hf_repo: o/B\n  C:\n    hf_repo: o/C\n")
    (tmp_path / "rec.json").write_text(json.dumps({"A": {"compiled_mode": {"16g": {"result": "zero miss, rc 0", "date": "d"}}}}))
    (tmp_path / "notes.json").write_text(json.dumps(notes or {"B": "a measured cause"}))
    return ["--cache", str(cache), "--rc", str(rc), "--derivation", str(tmp_path / "der.txt"),
            "--registry", str(tmp_path / "reg.yml"), "--records", str(tmp_path / "rec.json"),
            "--notes", str(tmp_path / "notes.json"), "--out", str(tmp_path / "out.md")]


def test_every_container_has_its_row_and_the_class_coverage(tmp_path):
    MS.main(_world(tmp_path))
    out = (tmp_path / "out.md").read_text()
    rows = [l for l in out.splitlines() if l.startswith("| `") and " · o/" in l]
    assert [r.split("`")[1] for r in rows] == ["A", "B", "C"]
    a = rows[0].split(" | ")
    assert a[1] == "exact" and a[2] == "1/1" and a[3] == "0/1"          # proven on 16 GB only
    assert "walked" in rows[1].split(" | ")[1]
    assert "a measured cause" in out and "no two containers share a repo id" in out


def test_an_empty_catalogue_or_a_stray_note_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no container"):
        MS.main(_world(tmp_path, names=()))
    with pytest.raises(SystemExit, match="name no container"):
        MS.main(_world(tmp_path / "x", notes={"Z": "cause"}))
