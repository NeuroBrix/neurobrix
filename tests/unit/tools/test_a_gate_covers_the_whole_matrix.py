"""A queue's gate covers every cell of the matrix: a gate whose card lists together miss a cell
refuses to start, by name.

2026-09-27: queue-9's gate ran the cell lists inherited from queue-8 (and a plan census over them),
never Sana_1600M_4Kpx_BF16 native — which queue-9 broke; queue-10's full gate found it on main.
Before this branch `run --gate-lists` does not exist and nothing refuses: these fail.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402


def _cache(tmp_path, names):
    for n in names:
        (tmp_path / "cache" / n).mkdir(parents=True)
        (tmp_path / "cache" / n / "manifest.json").write_text("{}")
    (tmp_path / "cache" / "not-a-container").mkdir()
    return tmp_path / "cache"


def test_the_full_matrix_is_every_container_in_every_mode(tmp_path):
    cache = _cache(tmp_path, ["a", "b"])
    assert R.full_matrix(cache) == {(m, mode) for m in "ab" for mode in R.MODES}


def test_a_partial_gate_is_refused_and_a_full_one_starts(tmp_path):
    cache = _cache(tmp_path, ["a", "b"])
    modes = ",".join(R.MODES)
    (tmp_path / "card0.list").write_text(f"a {modes}\n")
    (tmp_path / "card1.list").write_text("b native,triton\n")
    with pytest.raises(SystemExit, match="1 of 6 cells missing: b/triton-sequential"):
        R.refuse_a_partial_gate([tmp_path / "card0.list", tmp_path / "card1.list"], cache)
    (tmp_path / "card1.list").write_text(f"b {modes}\n")
    R.refuse_a_partial_gate([tmp_path / "card0.list", tmp_path / "card1.list"], cache)


def test_run_refuses_before_any_cell(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    cache = _cache(tmp_path, ["a"])
    monkeypatch.setattr(R, "CACHE", cache)
    (tmp_path / "l").write_text("a native\n")
    monkeypatch.setattr(R, "run_cell", lambda *a, **k: pytest.fail("a cell ran before the refusal"))
    args = NS(models="a", modes="native", gpu="0", out=str(tmp_path / "o"), src=str(tmp_path), timeout=60,
              rerun=False, gate_lists=[str(tmp_path / "l")])
    with pytest.raises(SystemExit, match="REFUSED"):
        R.cmd_run(args)
    assert not (tmp_path / "o").exists()
