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
