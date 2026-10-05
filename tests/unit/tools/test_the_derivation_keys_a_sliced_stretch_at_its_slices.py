"""The census derives a stretch the plan runs in slices of a token axis at each length its slices
take — the runtime's own split (`chunk_extents` over the slice count, `ChunkedPiece.run`) — for the
ops of its sliced passes only. Without it a certified-only run of SANA-Video at 16 GB missed every
key of its sliced linear-attention blocks: they launch at 4 of the 11 latent frames, not 11.

Injection (seen red, then restored green): the lengths taken from the planned slice instead of
`chunk_extents` -> 11 frames in slices of 5 derive 5 and 1, where the runtime runs 4, 4 and 3."""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "unit" / "prism"))
import derived_census as D  # noqa: E402
import test_a_stretch_over_the_rung_runs_in_slices_of_its_token_axis as G  # noqa: E402


def _plan(slice_, passes=2, first=G.FIRST, last=G.LAST):
    return {"layer_stream_chunks": {"c": [{"first_op": first, "last_op": last, "symbol": "s1",
                                           "slice": slice_, "count": 0, "passes": passes,
                                           "peak_bytes": 0}]}}


def test_a_sliced_stretch_is_keyed_at_each_length_its_slices_take(monkeypatch):
    monkeypatch.setattr(D, "runtime_graph", lambda model, comp: G._graph())
    miss = collections.Counter()
    got = D.sliced_bindings("M", "c", {"s0": 2, "s1": 11}, _plan(5), miss)
    assert not miss, miss
    assert sorted(b["s1"] for _ops, b in got) == [3, 4], got       # 11 in 3 slices: 4, 4, 3
    for ops, b in got:
        assert b["s0"] == 2
        assert "aten.bmm::0" in ops and "aten.bmm::1" in ops      # both passes touch the axis
        assert "aten.mul::0" not in ops                           # the piece before the stretch


def test_an_even_split_adds_one_length_and_a_whole_one_none(monkeypatch):
    monkeypatch.setattr(D, "runtime_graph", lambda model, comp: G._graph())
    assert [b["s1"] for _o, b in D.sliced_bindings("M", "c", {"s0": 2, "s1": 12}, _plan(4),
                                                   collections.Counter())] == [4]
    assert D.sliced_bindings("M", "c", {"s0": 2, "s1": 3}, _plan(3), collections.Counter()) == []
    assert D.sliced_bindings("M", "c", {"s0": 2, "s1": 3}, {}, collections.Counter()) == []


def test_a_stretch_the_executed_graph_plans_otherwise_is_named(monkeypatch):
    monkeypatch.setattr(D, "runtime_graph", lambda model, comp: G._graph())
    miss = collections.Counter()
    assert D.sliced_bindings("M", "c", {"s0": 2, "s1": 11}, _plan(5, passes=3), miss) == []
    assert any("plans 2 passes" in k and "the plan 3" in k for k in miss), miss
    miss = collections.Counter()
    D.sliced_bindings("M", "c", {"s0": 2, "s1": 11}, _plan(5, first="aten.nothing::0"), miss)
    assert any("not in the graph the executor runs" in k for k in miss), miss
