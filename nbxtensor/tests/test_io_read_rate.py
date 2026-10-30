"""nbxtensor.io.read_rate: a profile's cpu.io_workers is the SMALLEST worker count whose measured rate is within
5 % of the best — a worker past it holds a shard in host memory for nothing (Prism counts one per worker)."""
from pathlib import Path

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


def test_a_directory_without_weights_is_measured_on_a_scratch_file_it_removes(tmp_path):
    n = M.workers_for(tmp_path, max_bytes=2 * M.PIECE, max_workers=2)
    assert n is None or n in (1, 2)
    assert list(tmp_path.iterdir()) == []


def test_a_weight_cache_is_measured_on_its_shards(tmp_path):
    w = tmp_path / "m" / "components" / "c" / "weights"
    w.mkdir(parents=True)
    (w / "shard_000.safetensors").write_bytes(b"s" * (2 * M.PIECE))
    n = M.workers_for(tmp_path, max_bytes=2 * M.PIECE, max_workers=2)
    assert n is None or n in (1, 2)
    assert [p.name for p in w.iterdir()] == ["shard_000.safetensors"]


def test_the_count_is_the_median_of_three_steady_measures(tmp_path, monkeypatch):
    draws = iter([{"steady": True, "workers": 16}, {"steady": False, "workers": 2}, {"steady": True, "workers": 8},
                  {"steady": True, "workers": 16}])
    monkeypatch.setattr(M, "measure", lambda *a, **k: next(draws))
    (tmp_path / "x").write_bytes(b"0")
    assert M.workers_for(tmp_path, pattern="x") == 16          # 16, 8, 16 -> the median; the unsteady draw is not one


def test_fewer_than_two_steady_measures_state_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "measure", lambda *a, **k: {"steady": False, "workers": 4})
    (tmp_path / "x").write_bytes(b"0")
    assert M.workers_for(tmp_path, pattern="x") is None
