"""Row 167 (the supervisor, 2026-11-19): a measured fact is kept once measured and never paid twice; a measure that
cannot conclude keeps its best figure with its spread; the fresh-home work is done once per machine and not serially
where it can share. Three launch-path facts under it:

* the expandable pool's chunk: a sweep where no candidate keeps up with the driver segments (a busy rack wrote 0 on
  its four cards, kept for ever) keeps its best candidate with its spread, never a 0 as if measured;
* the card models' chunk sweeps run side by side, each on its own first card (48 replay children ran one after
  another, 1024 s of a 1236 s fresh 4-card launch);
* the weight reader's width is kept with its spread after a load, and the next launch's reader starts there.

What each test would do if the code were wrong: 0 written for a busy sweep (red), a tie broken to the smaller chunk
(red), the two classes swept one after another (the barrier times out: red), a reader width not kept, kept over a
settled one, or a next reader starting at one (red)."""
from __future__ import annotations

import threading
from pathlib import Path

from neurobrix.core.prism import autodetect as A
from neurobrix.core.prism import facts_probe as P
from tests.unit.prism.test_class_facts_are_measured_once_per_card_model import (  # noqa: F401 — the fixture
    C16, C32, CALLS, _read, _write, hw)


def test_an_unconcluded_sweep_keeps_its_best_candidate_with_its_spread(monkeypatch):
    from nbxtensor.memory import trace_replay as T
    times = {0: [2.0, 2.1, 2.2], 2 << 20: [9.0, 9.5], 32 << 20: [3.0, 3.1], 128 << 20: [3.0, 3.1]}
    monkeypatch.setattr(T, "vmm_granularity", lambda d: 2 << 20)
    monkeypatch.setattr(T, "chunk_by_replay", lambda *a, **k: (None, times))
    got = P.vmm_chunk_bytes(0, 3, (1, 16, 64))
    assert got == 128 << 20                                   # never 0; a median tie: the larger, the fewer maps
    assert got.spread["concluded"] is False and got.spread["median_s"] == 3.05 and got.spread["bound_s"] == 2.2
    monkeypatch.setattr(T, "chunk_by_replay", lambda *a, **k: (32 << 20, times))
    assert P.vmm_chunk_bytes(0, 3, (1,)) == 32 << 20 and getattr(P.vmm_chunk_bytes(0, 3, (1,)), "spread", None) is None
    monkeypatch.setattr(T, "vmm_granularity", lambda d: None)
    assert P.vmm_chunk_bytes(0, 3, (1,)) is None              # no VMM: the one case the pool keeps segments (0)


def test_the_classes_sweep_side_by_side_and_a_busy_sweep_keeps_its_spread(hw, monkeypatch):
    both = threading.Barrier(2, timeout=10)

    def sweep(d, *a, **k):
        both.wait()                                           # both classes in flight at once, or this times out
        CALLS.append(("vmm", d))
        if d == 2:
            out = P.UnconcludedChunk(64 << 20)
            out.spread = {"concluded": False, "median_s": 5.0}
            return out
        return 32 << 20
    monkeypatch.setattr(P, "vmm_chunk_bytes", sweep)
    _write(hw, "default-four", [dict(C16, index=0), dict(C16, index=1), dict(C32, index=2), dict(C32, index=3)])
    A.ensure_measured_facts("default-four")
    devs = _read(hw, "default-four")["devices"]
    assert sorted(c for c in CALLS if c[0] == "vmm") == [("vmm", 0), ("vmm", 2)]       # identical cards: once
    assert devs[1]["measured"]["vmm_chunk_bytes"] == 32 << 20 and "vmm_chunk_spread" not in devs[1]["measured"]
    assert devs[3]["measured"]["vmm_chunk_bytes"] == 64 << 20
    assert devs[3]["measured"]["vmm_chunk_spread"]["concluded"] is False


def test_the_reader_s_width_is_kept_and_the_next_launch_starts_there(hw, monkeypatch):
    _write(hw, "default-one", [dict(C16, index=0)])
    m = {"workers": 8, "settled": True, "start": None, "rates": {1: 1e9, 2: 2e9, 4: 3e9, 8: 3.5e9, 16: 3.4e9},
         "draws": {8: [3.5e9, 3.3e9]}, "spread": (3.3e9, 3.5e9)}
    assert A.note_read_width("default-one", m)
    kept = _read(hw, "default-one")["host_measured"]["weight_read"]
    assert (kept["workers"], kept["low_gbps"], kept["high_gbps"], kept["rounds"], kept["settled"]) == (8, 3.3, 3.5, 2, True)
    assert kept["measured"]
    assert not A.note_read_width("default-one", dict(m, workers=16, draws={16: [9e9]}))   # never paid twice
    assert A.kept_read_width("default-one") == 8
    from neurobrix.core import workspace as W
    monkeypatch.setattr(W, "_READER", None)
    monkeypatch.setattr(W, "_LOAD_CEILING", 32)
    monkeypatch.setattr(A, "ACTIVE_HARDWARE_ID", "default-one")
    r = W.weight_reader()
    assert r.width == 8 and r.measured()["start"] == 8
    src = (Path(A.__file__).parents[1] / "runtime" / "graph_executor.py").read_text()
    assert "note_read_width(_ad.ACTIVE_HARDWARE_ID, weight_reader_measured())" in src


def test_an_unsettled_read_keeps_its_best_width_until_a_settled_search_replaces_it(hw):
    _write(hw, "default-one", [dict(C16, index=0)])
    grow = {"workers": 4, "settled": False, "start": None, "rates": {1: 1e9, 2: 2e9}, "draws": {1: [1e9], 2: [2e9]}}
    assert A.note_read_width("default-one", grow)
    assert _read(hw, "default-one")["host_measured"]["weight_read"]["workers"] == 2
    started = {"workers": 2, "settled": True, "start": 2, "rates": {}, "draws": {2: [2.1e9]}}
    assert not A.note_read_width("default-one", started)       # started from the kept figure: not a search
    quiet = {"workers": 8, "settled": True, "start": None, "rates": {8: 4e9}, "draws": {8: [4e9, 3.9e9]}}
    assert A.note_read_width("default-one", quiet)
    assert _read(hw, "default-one")["host_measured"]["weight_read"]["workers"] == 8
