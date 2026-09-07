"""A cold env A/B's kernel choices classified against the certified directory: alike, a near-tie
(the certifier's second-best within 10 % of its best — the timer's noise), a contradicted
certified choice, an excluded setting picked at runtime, or an uncertified key (the runtime
sweep's own variance). CogVideoX-2b's proof row, 2026-09-07: six matmul/addmm near-ties,
three uncertified keys, the videos 40 dB apart — not a certification contradicted."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import precision_zoo_campaign as C  # noqa: E402

K = "neurobrix.kernels.ops.matmul.matmul_kernel::"
CFG_A = {"kwargs": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, "num_warps": 4, "num_stages": 3}
CFG_B = {"kwargs": {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, "num_warps": 4, "num_stages": 3}
CFG_X = {"kwargs": {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, "num_warps": 8, "num_stages": 2}


def _directory(src: Path, entries: dict):
    d = src / "neurobrix" / "config" / "autotune" / "nvidia" / "volta"; d.mkdir(parents=True)
    (d / "matmul_kernel.fp32.json").write_text(json.dumps({"entries": entries}))


def _replays(out: Path, a: dict, b: dict):
    for arm, st in (("A", a), ("B", b)):
        (out / "M" / f"{arm}_replay").mkdir(parents=True)
        (out / "M" / f"{arm}_replay" / "autotune_configs_cuda-70.json").write_text(json.dumps(st))


def test_choices_are_classified_by_the_entry_proof(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _directory(src, {
        "(1, 1, 1)": {"config": CFG_A, "proof": {"best_ms": 1.000, "second_ms": 1.004}, "excluded": []},        # near-tie
        "(2, 2, 2)": {"config": CFG_A, "proof": {"best_ms": 1.000, "second_ms": 1.500}, "excluded": []},        # clear margin
        "(3, 3, 3)": {"config": CFG_A, "proof": {"best_ms": 1.0, "second_ms": 1.1},
                      "excluded": [{"config": CFG_X, "deviation": 0.5, "tolerance": 1e-4}]},                    # excluded picked
        "(4, 4, 4)": {"config": CFG_A, "proof": {"best_ms": 1.0, "second_ms": 1.9}, "excluded": []},           # alike
    })
    _replays(out, {K + "(1, 1, 1)": CFG_A, K + "(2, 2, 2)": CFG_A, K + "(3, 3, 3)": CFG_A, K + "(4, 4, 4)": CFG_A, K + "(9, 9, 9)": CFG_A},
                  {K + "(1, 1, 1)": CFG_B, K + "(2, 2, 2)": CFG_B, K + "(3, 3, 3)": CFG_X, K + "(4, 4, 4)": CFG_A, K + "(9, 9, 9)": CFG_B})
    ch = C._choices_ab(out / "M", src)
    assert (ch["keys"], ch["certified"], ch["differ"]) == (5, 4, 4)
    assert ch["near_tie_count"] == 1 and abs(ch["near_tie"][0]["margin"] - 0.004) < 1e-9
    assert ch["contradicted_count"] == 1 and ch["contradicted"][0]["key"].endswith("(2, 2, 2)")
    assert ch["excluded_picked"] == [K + "(3, 3, 3)"]
    assert ch["differ_uncertified"] == [K + "(9, 9, 9)"]
    line = C.verdict({"lever": "env:x", "gate": {"ran": True, "identical": False}, "A": {}, "B": {}, "choices": ch})
    assert "EXCLUDED SETTING PICKED AT RUNTIME on 1 key(s)" in line
    assert "CERTIFIED CHOICE CONTRADICTED on 1 key(s)" in line and "margin 50 % = 500.0 µs on a 1000.0 µs kernel" in line
    assert "1 certified near-tie(s)" in line and "1 uncertified" in line


def test_alike_choices_say_so(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _directory(src, {"(1, 1, 1)": {"config": CFG_A, "proof": {"best_ms": 1.0, "second_ms": 1.5}, "excluded": []}})
    _replays(out, {K + "(1, 1, 1)": CFG_A}, {K + "(1, 1, 1)": CFG_A})
    ch = C._choices_ab(out / "M", src)
    assert ch["differ"] == 0
    assert "every kernel choice alike on 1 keys (1 certified)" in C.verdict({"lever": "env:x", "gate": {"ran": True, "identical": True}, "A": {}, "B": {}, "choices": ch})
