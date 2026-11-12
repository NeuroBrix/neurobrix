"""The fused MoE op (`ops.moe_fused.fused_moe`) refuses by name what it cannot route before it touches a device; a store
call wider than its slots is cut into runs of whole experts; a slot store answers the op's three questions by default.

  * the refusals: no activations; a backend that cannot read through a pointer loaded from a table (every path
    addresses its experts that way, and such a backend answers zeros in silence); half of a blended routing; neither a
    router's scores nor a blended routing;
  * `_runs_of_experts`: the aligned blocks (sorted by expert) cut into runs of at most `n_slots` distinct experts,
    every block in exactly one run, an expert never split across two;
  * `memory.expert_slots.ExpertSlotStore`: `decodes_on_card` only when the card's residency is armed for that layer;
    `on_routed` and `on_enqueued` keep nothing.

What this test would do if the code were wrong: a refusal moved after the first device call meets the sentinel that
replaces `DeviceAllocator.set_device` and raises "past the door" instead of the refusal — the `pytest.raises` match
fails; a run holding more experts than slots, an expert split across two runs, or a block in no run — RED on the runs;
a store answering True for a layer its card's map does not hold — RED. No card.
"""
from __future__ import annotations

import pytest

pytest.importorskip("triton")


def _past_the_door(*_a, **_k):
    raise RuntimeError("past the door: the op went on to the device")


@pytest.fixture
def op(monkeypatch):
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.ops import moe_fused
    monkeypatch.setattr(DeviceAllocator, "set_device", staticmethod(_past_the_door))
    monkeypatch.setattr(moe_fused, "backend_loads_pointers_from_memory", lambda: True)
    return moe_fused


def test_no_activations_is_refused(op):
    with pytest.raises(RuntimeError, match="hidden_states is None"):
        op.fused_moe(object(), None, [], [], [], 2, 4)


def test_a_backend_that_cannot_load_a_pointer_is_refused(op, monkeypatch):
    monkeypatch.setattr(op, "backend_loads_pointers_from_memory", lambda: False)
    with pytest.raises(RuntimeError, match="cannot read through a pointer loaded"):
        op.fused_moe(object(), object(), [], [], [], 2, 4)


@pytest.mark.parametrize("given", ["indices", "weights"])
def test_half_a_blended_routing_is_refused(op, given):
    kw = {"topk_indices": object()} if given == "indices" else {"topk_weights": object()}
    with pytest.raises(RuntimeError, match="must be supplied together"):
        op.fused_moe(None, object(), [], [], [], 2, 4, **kw)


def test_no_routing_at_all_is_refused(op):
    with pytest.raises(RuntimeError, match="gate_scores is None and no blended routing"):
        op.fused_moe(None, object(), [], [], [], 2, 4)


@pytest.mark.parametrize("blocks,n_slots", [
    ([0, 0, 1, 2, 2, 3, 4], 2),
    ([5, 5, 5, 9], 1),
    ([0, 1, 2, 3, 4, 5, 6, 7], 3),
    ([3, 3, 7, 7, 7, 8], 8),
])
def test_a_store_call_is_cut_into_runs_of_whole_experts(blocks, n_slots):
    from nbxtensor.ops.moe_fused import _runs_of_experts
    runs = _runs_of_experts(blocks, n_slots, len(blocks))
    assert runs[0][0] == 0 and runs[-1][1] == len(blocks), runs
    for (lo, hi, experts), nxt in zip(runs, runs[1:] + [None]):
        assert 0 < len(experts) <= n_slots, runs
        assert experts == sorted(set(blocks[lo:hi])), (runs, "a run's experts are its blocks' experts")
        assert nxt is None or nxt[0] == hi, (runs, "the runs tile the blocks")
    named = [e for _, _, experts in runs for e in experts]
    assert named == sorted(set(blocks)), (runs, "an expert is in exactly one run")
    if len(set(blocks)) <= n_slots:
        assert len(runs) == 1, runs


def test_a_slot_store_answers_the_ops_questions_by_default():
    from nbxtensor.memory.expert_slots import ExpertSlotStore
    store = ExpertSlotStore.__new__(ExpertSlotStore)      # no arena: the questions read no device
    store.device_decode = False
    assert store.decodes_on_card("L0") is False
    store.device_decode, store.layer_index = True, {"L0": 0}
    assert store.decodes_on_card("L0") is True
    assert store.decodes_on_card("L1") is False
    assert store.on_routed("L0", [1, 2], lambda e: None) is None
    assert store.on_enqueued("L0", [1, 2]) is None
