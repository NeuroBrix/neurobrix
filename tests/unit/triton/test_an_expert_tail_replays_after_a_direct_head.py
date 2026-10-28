"""S3 (a): a decode step of an expert-store LM runs its head (the store-served MoE layers) directly and replays its tail
(the layers whose experts other cards hold whole) — Qwen3-30B-A3B triton on 4 V100 (2026-10-28): 1431 launches
verified byte-equal, the 64-token text identical, tpot 383 -> 278 ms.

1. The cut: just after the last MoE op the store serves; the boundary slots are what the tail reads and does not make.
2. A device synchronisation made while recording is an action of the plan (`transfer_tensor` waits for the source
   card before a peer copy; replayed without it, error 700 that CUDA_LAUNCH_BLOCKING hid).

What each test would do if the code were wrong: a cut before a store-served MoE op or after a held one, or a boundary
missing a head output (red); a sync not recorded (red)."""
from types import SimpleNamespace

import pytest

from neurobrix.triton.sequence import TritonSequence


def _op(uid, op_type, ins=(), outs=()):
    return SimpleNamespace(op_uid=uid, op_type=op_type, all_input_slots=tuple(ins), output_slots=tuple(outs))


def test_the_tail_starts_after_the_last_store_served_moe_and_reads_the_head_s_outputs():
    seq = object.__new__(TritonSequence)
    seq._num_weights, seq._num_inputs = 2, 1                 # slots 0-1 weights, 2 input, 3+ intermediates
    seq.dag = {"ops": {u: {"attributes": {"expert_gate_weight_ids": [g]}} for u, g in
                       (("moe0", "L0.e0"), ("moe1", "L1.e0"), ("moe2", "L2.e0"))}}
    seq._expert_cards = {"L2.e0": ("cuda:1", None)}            # layer 2's experts held on another card
    seq._ops = [_op("a", "aten::mm", (0, 2), (3,)), _op("moe0", "custom::moe_fused", (3,), (4,)),
                _op("b", "aten::add", (4, 3), (5,)), _op("moe1", "custom::moe_fused", (5,), (6,)),
                _op("c", "aten::mm", (6, 1), (7,)), _op("moe2", "custom::moe_fused", (7,), (8,)),
                _op("d", "aten::add", (8, 5), (9,))]
    cut, boundary = seq._expert_tail_cut()
    assert cut == 4, cut                                      # after moe1, the last op the store serves
    assert boundary == (5, 6), boundary                       # the head's outputs the tail reads; 7 and 8 it makes


def test_no_tail_without_a_held_layer_after_the_store():
    seq = object.__new__(TritonSequence)
    seq._num_weights, seq._num_inputs = 0, 0
    seq.dag = {"ops": {"moe0": {"attributes": {"expert_gate_weight_ids": ["L0.e0"]}}}}
    seq._expert_cards = {}
    seq._ops = [_op("moe0", "custom::moe_fused", (0,), (1,))]
    assert seq._expert_tail_cut() is None


def _has_gpu():
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_gpu(), reason="needs a device")
def test_a_sync_made_while_recording_is_an_action_of_the_plan():
    from neurobrix.triton import replay as R
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    R._install_seams()
    R.STATE.records, R.STATE.recording = [], True
    try:
        DeviceAllocator.sync_device()
    finally:
        R.STATE.recording = False
    assert (R._SYNC, (int(DeviceAllocator.get_device()),)) in R.STATE.records, R.STATE.records
