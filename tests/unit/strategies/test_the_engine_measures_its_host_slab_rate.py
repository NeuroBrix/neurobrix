"""The host slab rate the time estimate prices a host-read sliced stretch at (`host_slab_gbps`) is measured BY THE
ENGINE inside its sliced runs — a probe outside the run does not reproduce it (the engine's take_slice on host tensors:
23-76 GB/s alone on the M4 Pro, 3.06 GB/s inside the sliced mochi run, 2026-10-28). ChunkedPiece counts the bytes it
cuts from host tensors and assembles for such a stretch, and their time (`chunked_piece.host_slab_gbps`). No card."""
from neurobrix.core.strategies import chunked_piece as CP


def test_no_cut_no_rate(monkeypatch):
    monkeypatch.setattr(CP, "HOST_SLAB", [0, 0.0])
    assert CP.host_slab_gbps() is None


def test_the_rate_is_bytes_over_time(monkeypatch):
    monkeypatch.setattr(CP, "HOST_SLAB", [6 * 10 ** 9, 2.0])
    assert CP.host_slab_gbps() == 3.0
