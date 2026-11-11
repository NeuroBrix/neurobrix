"""A replayed decode step of an expert-store model gives the weight ratchet its call before it replays.

A serving daemon of Qwen3-Coder-30B-A3B on one 32 GB card answered its FIRST request at 0.08 s a token and every later
one at 17.7 s a token (2026-11-11). At each request the arena's dense weights are bound back to their host originals;
the expert-store plan's ratchet puts the card's copies back "before that block's first op" — through the per-op
callback — and a replayed stretch runs no op, so it called nothing: from the second request on (the decode's plan
already recorded) the block was never re-installed and the replay's input scan copied 434 host weights into the
recorded buffers at every token.

What this test does if the code is wrong: the callback is not called before the replay (the list holds the replay
alone), and it fails.
"""
from types import SimpleNamespace

from neurobrix.triton import replay as _replay
from neurobrix.triton.sequence import TritonSequence


def _sequence(segments, order):
    seq = object.__new__(TritonSequence)
    seq._ops = [SimpleNamespace(op_uid=f"op{i}") for i in range(6)]
    seq._expert_segments = lambda: segments
    seq._run_multi_device = lambda skip_kills, cb: order.append(("direct run", cb is not None))
    return seq


def test_the_ratchet_is_called_at_the_first_op_of_every_replayed_stretch(monkeypatch):
    order = []
    monkeypatch.setattr(_replay, "maybe_run", lambda seq, skip_kills, cb: order.append(("replay", seq._replay_segment)) or True)
    segments = [("replay", 0, 2, (), [False] * 6), ("direct", 2, 3, (), [False] * 6), ("tail", 3, 6, (), [False] * 6)]
    seq = _sequence(segments, order)
    seq._run_expert_tail(False, lambda op_idx, op: order.append(("ratchet", op_idx, op.op_uid)))
    assert order == [("ratchet", 0, "op0"), ("replay", 0), ("direct run", True), ("ratchet", 3, "op3"), ("replay", 2)]


def test_a_step_without_a_ratchet_replays_as_before(monkeypatch):
    order = []
    monkeypatch.setattr(_replay, "maybe_run", lambda seq, skip_kills, cb: order.append("replay") or True)
    seq = _sequence([("tail", 0, 6, (), [False] * 6)], order)
    seq._run_expert_tail(False, None)
    assert order == ["replay"]
