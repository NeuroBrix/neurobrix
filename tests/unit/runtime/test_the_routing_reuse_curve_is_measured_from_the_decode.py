"""The routing reuse curve (core/runtime/routing_reuse.py): the hit rate of a per-layer LRU of `s` slots over the
decode's routed sets, at every multiple of top_k up to the expert count, written once per model and component in the
engine store, never overwritten; a prefill call (more than top_k routed) is not a decode token.

What this test would do if the code were wrong: an LRU miscounted (a hand-computed rate differs — red); a prefill call
noted (red); a recorded curve overwritten (red); a ladder with a hard-coded step (top_k=3 reads 3, 6, 8 — red)."""
import json

from neurobrix.core.runtime import routing_reuse as RR


def test_the_lru_counts_hits_by_hand():
    calls = [[0, 1], [1, 2], [0, 3]]                  # 2 slots: 0,1 miss | 1 hit, 2 miss (evicts 0) | 0 miss, 3 miss
    assert RR.lru_hit_rate(calls, 2) == 1 / 6
    assert RR.lru_hit_rate(calls, 4) == 2 / 6          # nothing evicted: 1 and 0 hit


def test_the_ladder_is_the_model_s_own():
    assert RR.slot_ladder(3, 8) == [3, 6, 8]
    assert RR.slot_ladder(8, 128)[:3] == [8, 16, 24] and RR.slot_ladder(8, 128)[-1] == 128


def test_a_trace_notes_decode_calls_only_and_writes_once(tmp_path, monkeypatch):
    monkeypatch.setattr(RR, "_root", lambda: tmp_path)
    t = RR.ExpertTrace("m", "model", top_k=2, n_experts=4)
    t.note("L0", [0, 1]); t.note("L0", [1, 2]); t.note("L0", [0, 1, 2, 3])   # the last is a prefill
    assert t.calls == {"L0": [[0, 1], [1, 2]]}
    p = t.write("test")
    data = json.loads(p.read_text())
    assert data["tokens"] == 2 and data["curve"]["2"] == 0.25 and data["curve"]["4"] == 0.25
    t2 = RR.ExpertTrace("m", "model", top_k=2, n_experts=4)
    t2.note("L0", [3, 3])
    assert t2.write("again") is None and json.loads(p.read_text())["source"] == "test"
