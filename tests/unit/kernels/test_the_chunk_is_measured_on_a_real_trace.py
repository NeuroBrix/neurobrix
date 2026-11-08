"""The expandable range's chunk is measured by replaying a REAL run's allocations (the supervisor 2026-11-02 17:15):
the library carries the trace as data, the probe replays it with the range off and at each candidate, and keeps the
smallest candidate within R16 OFF's noise.

What this test would do if the code were wrong: a probe that measures something else (the per-call churn it replaced,
which always favours the largest chunk), a criterion that keeps a candidate slower than the segments or skips a
smaller one within their noise, a trace without its card size or a probe that picks one of several traces silently —
RED. The device half replays the carried trace on a card and checks the answer is a whole number of granules."""
from __future__ import annotations

import os

import pytest

from nbxtensor.memory import trace_replay as T

MB = 1 << 20


def test_the_library_carries_a_trace_that_names_its_card():
    (path,) = T.traces()
    meta, events = T.load_trace(path)
    assert int(meta["card_mb"]) > 0 and {k for k, _, _ in events} == {"M", "A", "F"}
    assert sum(1 for k, _, _ in events if k == "F") <= sum(1 for k, _, _ in events if k != "F")


def test_a_trace_without_its_card_is_refused(tmp_path):
    p = tmp_path / "t.tsv"
    p.write_text("kind\tid\tnbytes\nM\t0\t512\n")
    with pytest.raises(RuntimeError, match="card_mb"):
        T.load_trace(p)


def _timed(monkeypatch, seconds):
    calls = []

    def child(trace, device, chunk):
        calls.append(chunk)
        return {"seconds": seconds[chunk].pop(0)}
    monkeypatch.setattr(T, "_child", child)
    return calls


def test_the_smallest_chunk_within_the_segments_noise_is_kept(monkeypatch):
    g = 2 * MB
    seconds = {0: [2.2, 2.3, 2.1], 2 * MB: [11.0] * 3, 4 * MB: [6.0] * 3, 8 * MB: [2.4, 2.35, 2.5],
               16 * MB: [2.25, 2.2, 2.3], 32 * MB: [2.0] * 3}
    calls = _timed(monkeypatch, seconds)
    chunk, times = T.chunk_by_replay(0, g, (1, 2, 4, 8, 16), 3, trace="x")
    assert chunk == 16 * MB, {k >> 20: v for k, v in times.items()}      # OFF 2.2 + spread 0.1: 8 MiB (2.4) is out
    assert 32 * MB not in calls, "the probe stops at the first candidate within the noise"


def test_no_chunk_within_the_noise_keeps_the_segments(monkeypatch):
    seconds = {0: [1.0] * 2, 2 * MB: [5.0] * 2, 4 * MB: [3.0] * 2}
    _timed(monkeypatch, seconds)
    assert T.chunk_by_replay(0, 2 * MB, (1, 2), 2, trace="x")[0] is None


def test_several_traces_are_named_never_picked(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "TRACES_DIR", tmp_path)
    for n in ("a.tsv", "b.tsv"):
        (tmp_path / n).write_text("# card_mb: 16384\nM\t0\t1\n")
    with pytest.raises(RuntimeError, match="names the one it replays"):
        T.chunk_by_replay(0, 2 * MB, (1,), 1)


def test_the_first_launch_probe_measures_on_the_trace(monkeypatch):
    from neurobrix.core.prism import facts_probe as P
    from nbxtensor.memory import vmm as V
    monkeypatch.setattr(V.Vmm, "load", staticmethod(lambda libs: object()))
    monkeypatch.setattr(V.Vmm, "supported", lambda self: True)
    monkeypatch.setattr(V.Vmm, "granularity", lambda self: (0, 2 * MB))
    seen = []
    monkeypatch.setattr(T, "chunk_by_replay", lambda dev, g, ks, reps, trace=None, off_first=None: (seen.append((g, tuple(ks))), (32 * MB, {}))[1])
    assert P.vmm_chunk_bytes(0, 3, (1, 16)) == 32 * MB and seen == [(2 * MB, (1, 16))]


@pytest.mark.skipif(not os.environ.get("CUDA_VISIBLE_DEVICES"), reason="needs a card")
def test_the_carried_trace_replays_on_a_card():
    from nbxtensor.backend import runtime as _rt
    from nbxtensor.memory.vmm import Vmm
    lib = Vmm.load(_rt.backend_table().get("driver_libs"))
    if lib is None or not Vmm(lib, 0).supported():
        pytest.skip("no VMM on this card")
    rc, g = Vmm(lib, 0).granularity()
    chunk, times = T.chunk_by_replay(0, g, (16,), 1)
    assert set(times) == {0, 16 * g} and (chunk is None or chunk % g == 0)
