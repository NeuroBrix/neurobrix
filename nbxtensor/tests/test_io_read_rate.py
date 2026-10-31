"""nbxtensor.io.read_rate: a profile's cpu.io_workers is the SMALLEST worker count whose measured rate is within
5 % of the best — a worker past it holds a shard in host memory for nothing (Prism counts one per worker)."""
from pathlib import Path

import pytest

from nbxtensor.io import read_rate as M


def test_a_flat_rate_takes_one_worker():
    assert M.choose({1: 34.0, 2: 33.4, 4: 34.6, 8: 31.8}, 0.05) == 1     # a network-bound mount (the Mac's NFS)


def test_a_rate_that_scales_takes_the_count_where_it_stops():
    assert M.choose({1: 400, 2: 780, 4: 1400, 8: 2600, 16: 2650, 32: 2660}, 0.05) == 8


def test_the_pieces_cover_the_files_up_to_the_cap(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"x" * (M.PIECE + 5)); b.write_bytes(b"y" * 7)
    pieces, total = M.pieces([str(a), str(b)], 1 << 40)
    assert total == M.PIECE + 12 and [(Path(p).name, o, n) for p, o, n in pieces] == [
        ("a", 0, M.PIECE), ("a", M.PIECE, 5), ("b", 0, 7)]


def test_a_measure_reads_every_count_and_says_whether_it_was_steady(tmp_path):
    f = tmp_path / "w"
    f.write_bytes(b"z" * (3 * M.PIECE))
    m = M.measure([str(f)], 1 << 40, 4)
    assert set(m["rates"]) == {1, 2, 4} and m["bytes"] == 3 * M.PIECE and m["workers"] in (1, 2, 4)
    assert isinstance(m["steady"], bool)


def test_a_directory_without_weights_is_measured_on_a_witness_it_removes(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "WITNESS_BYTES", 2 * M.PIECE)
    seen = []
    monkeypatch.setattr(M, "measure", lambda files, *a: seen.append(files) or {"steady": True, "workers": 2})
    assert M.workers_for(tmp_path, max_workers=2) == 2
    assert seen and all(f[0].startswith(str(tmp_path)) for f in seen)    # the storage measured is the directory's
    assert list(tmp_path.iterdir()) == []


def test_the_first_trial_reads_one_shard_then_grows_while_not_stable(tmp_path, monkeypatch):
    for i in range(8):
        (tmp_path / f"x{i}").write_bytes(b"0" * (8 * M.PIECE))               # shards of 8 pieces, 64 in all
    sizes = []

    def measure(files, size, workers):
        sizes.append(size)
        return {"steady": size >= 32 * M.PIECE, "workers": 8}
    monkeypatch.setattr(M, "measure", measure)
    assert M.workers_for(tmp_path, max_workers=8, pattern="x*") == 8
    assert sorted(set(sizes)) == [8 * M.PIECE, 32 * M.PIECE]              # one shard first, quadrupled once


def test_the_first_trial_is_capped(tmp_path, monkeypatch):
    (tmp_path / "x").write_bytes(b"0" * (8 * M.PIECE))
    monkeypatch.setattr(M, "START_BYTES", 2 * M.PIECE)
    sizes = []
    monkeypatch.setattr(M, "measure", lambda f, size, w: sizes.append(size) or {"steady": True, "workers": 2})
    M.workers_for(tmp_path, max_workers=2, pattern="x")
    assert sizes[0] == 4 * M.PIECE                                         # never below four pieces


def test_no_stable_measure_at_any_size_is_refused_by_name(tmp_path, monkeypatch):
    (tmp_path / "x").write_bytes(b"0" * (8 * M.PIECE))
    monkeypatch.setattr(M, "measure", lambda *a: {"steady": False, "workers": 4})
    with pytest.raises(RuntimeError, match="no stable read-rate measure"):
        M.workers_for(tmp_path, max_workers=4, pattern="x")


def test_two_steady_draws_that_disagree_are_decided_by_a_third(tmp_path, monkeypatch):
    (tmp_path / "x").write_bytes(b"0" * (8 * M.PIECE))
    draws = iter([{"steady": True, "workers": 16}, {"steady": True, "workers": 8}, {"steady": False, "workers": 2},
                  {"steady": True, "workers": 16}])
    monkeypatch.setattr(M, "measure", lambda *a: next(draws))
    assert M.workers_for(tmp_path, max_workers=16, pattern="x") == 16   # 16, 8, 16 -> the median
