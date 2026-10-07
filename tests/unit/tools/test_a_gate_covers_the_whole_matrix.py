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


# The owner, 2026-10-05 23:16: triton-sequential runs WHOLE only on small models; video and heavy models are gated
# compiled + Triton certified-only. The mode set is read from the family profile and the hardware profile.
HW = "c4140-4xv100-custom-nvlink"  # the rack's profile; its smallest card holds 16160 MB


def _container(cache, name, family, weight_bytes):
    import json
    w = cache / name / "components" / "c" / "weights"
    w.mkdir(parents=True)
    (cache / name / "manifest.json").write_text(json.dumps({"family": family}))
    with open(w / "w.bin", "wb") as f:
        f.truncate(weight_bytes)  # sparse: the size is read, never the bytes


def test_whole_sequential_only_where_the_data_allows_it(tmp_path):
    cache = tmp_path / "cache"
    _container(cache, "small_llm", "llm", 1 << 20)
    _container(cache, "heavy_llm", "llm", (16160 << 20) + 1)
    _container(cache, "small_video", "video", 1 << 20)
    full = R.full_matrix(cache, HW)
    assert {m for m, mo in full if mo == "triton-sequential"} == {"small_llm"}
    assert {(m, mo) for m, mo in full if mo != "triton-sequential"} == {
        (m, mo) for m in ("small_llm", "heavy_llm", "small_video") for mo in ("native", "triton")}
    # Without a profile the matrix is every mode (the census's and the reports' view), unchanged.
    assert len(R.full_matrix(cache)) == 9
    # A gate naming sequential on a video model is refused as naming an unknown cell.
    (tmp_path / "l").write_text("small_llm native,triton,triton-sequential\nheavy_llm native,triton\n"
                                "small_video native,triton,triton-sequential\n")
    with pytest.raises(SystemExit, match="unknown: small_video/triton-sequential"):
        R.refuse_a_partial_gate([tmp_path / "l"], cache, HW)


def test_a_gate_without_its_hardware_profile_is_refused(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    cache = tmp_path / "cache"
    _container(cache, "a", "llm", 1)
    monkeypatch.setattr(R, "CACHE", cache)
    (tmp_path / "l").write_text("a native,triton,triton-sequential\n")
    monkeypatch.setattr(R, "run_cell", lambda *a, **k: pytest.fail("a cell ran before the refusal"))
    args = NS(models="a", modes="native", gpu="0", out=str(tmp_path / "o"), src=str(tmp_path), timeout=60,
              rerun=False, gate_lists=[str(tmp_path / "l")], gate_hardware=None)
    with pytest.raises(SystemExit, match="--gate-hardware"):
        R.cmd_run(args)
