"""The gate tally reads the TIME of a cell against its base and judges OUTPUTS by their sha (the supervisor,
2026-10-26 23:20). Before: a 2-20x slowdown of every Triton cell crossed gates ecr2..tha4 green (db94d33d), and a
pair judged once passed every later output of that pair as judged. What this test would do if the code were wrong:
the slow cell green, the pair-only judgment counted, an empty base accepted, a broken output green — each RED here.
The real case, kept beside it: gate ecr2 against its base gate_msnv2 tallied 98 clean before, 23 red SLOWER now."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

import gate_tally as T  # noqa: E402


def _gate(root, rows, judg=None):
    (root / "gate2").mkdir(parents=True)
    (root / "gate2" / "gate_32g.list").write_text("".join(f"{m} {x}\n" for m, x, *_ in rows))
    d = root / "gate" / "matrix_32g"
    d.mkdir(parents=True)
    with open(d / "rows_card0.jsonl", "w") as f:
        for m, x, ex, sha in rows:
            f.write(json.dumps({"model": m, "mode": x, "rc": 0, "exec_s": ex, "wall_s": ex + 5, "sha256": sha,
                                "date": "2026-10-26T20:00:00Z"}) + "\n")
    if judg:
        (d / "judgments.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judg))
    return root


def _tally(tmp_path, gate, base, record=()):
    tree = tmp_path / "tree" / "docs" / "reference"
    tree.mkdir(parents=True, exist_ok=True)
    (tree / "known-cause-reds.32g.yml").write_text("reds: []\n")
    (tree / "artefact-judgments.jsonl").write_text("".join(json.dumps(j) + "\n" for j in record))
    return T.main(["--gate", str(gate), "--base", str(base), "--tree", str(tmp_path / "tree"), "--classes", "32g"])


def _works(sha):
    return {"sha256": sha, "verdict": "works", "date": "2026-10-26 20:00", "model": "m", "mode": "triton"}


def test_a_cell_over_the_factor_of_its_base_is_red(tmp_path, capsys):
    base = _gate(tmp_path / "base", [("m", "triton", 10.0, "aa")])
    gate = _gate(tmp_path / "g", [("m", "triton", 16.0, "aa")])
    assert _tally(tmp_path, gate, base, [_works("aa")]) == 1
    assert "SLOWER 16.0s vs base 10.0s" in capsys.readouterr().out


def test_a_cell_within_the_factor_is_green(tmp_path):
    base = _gate(tmp_path / "base", [("m", "triton", 10.0, "aa")])
    gate = _gate(tmp_path / "g", [("m", "triton", 14.0, "aa")])
    assert _tally(tmp_path, gate, base, [_works("aa")]) == 0


def test_a_pair_judged_without_a_sha_judges_no_output(tmp_path, capsys):
    base = _gate(tmp_path / "base", [("m", "triton", 10.0, "aa")])
    gate = _gate(tmp_path / "g", [("m", "triton", 10.0, "bb")],
                 judg=[{"model": "m", "mode": "triton", "verdict": "works", "judged": "x", "date": "2026-10-26 20:00"}])
    _tally(tmp_path, gate, base, [_works("aa")])
    assert "NOT JUDGED m:triton sha bb" in capsys.readouterr().out


def test_an_output_judged_broken_is_red(tmp_path, capsys):
    base = _gate(tmp_path / "base", [("m", "triton", 10.0, "aa")])
    gate = _gate(tmp_path / "g", [("m", "triton", 10.0, "cc")])
    rec = [dict(_works("cc"), verdict="broken", judged="noise")]
    assert _tally(tmp_path, gate, base, rec) == 1
    assert "JUDGED broken" in capsys.readouterr().out


def test_an_empty_base_is_refused_by_name(tmp_path, capsys):
    gate = _gate(tmp_path / "g", [("m", "triton", 10.0, "aa")])
    (tmp_path / "empty").mkdir()
    assert _tally(tmp_path, gate, tmp_path / "empty", [_works("aa")]) == 2
    assert "holds no rows" in capsys.readouterr().out
