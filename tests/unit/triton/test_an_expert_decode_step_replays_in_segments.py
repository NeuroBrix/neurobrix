"""An expert-store decode step with a card-held tail runs its store-served MoE ops directly and replays every stretch
between them and the tail, each segment with the slots it reads and does not produce (S3, 2026-10-28: Qwen3-30B-A3B on
4 V100, 17 segments, 8 direct; steady step 84 -> 79 ms, text identical). No card."""
from types import SimpleNamespace

from neurobrix.triton.sequence import TritonSequence


def _op(t, ins, outs, uid):
    return SimpleNamespace(op_type=t, all_input_slots=tuple(ins), output_slots=tuple(outs), op_uid=uid)


def test_the_store_moe_ops_split_the_step_and_each_segment_names_its_boundary():
    seq = TritonSequence.__new__(TritonSequence)
    # slots: 0-1 weights, 2 input, 3.. intermediates
    seq._num_weights, seq._num_inputs = 2, 1
    seq._ops = [_op("aten::mm", [2, 0], [3], "a"),                  # 0
                _op("custom::moe_fused", [3], [4], "m0"),          # 1  store-served
                _op("aten::add", [4, 3, 1], [5], "b"),             # 2
                _op("custom::moe_fused", [5], [6], "m1"),          # 3  store-served
                _op("aten::mul", [6], [7], "c"),                   # 4  tail from here
                _op("custom::moe_fused", [7], [8], "m2")]          # 5  held on another card
    seq.dag = {"component_name": "model", "ops": {
        "m0": {"attributes": {"expert_gate_weight_ids": ["g0"]}},
        "m1": {"attributes": {"expert_gate_weight_ids": ["g1"]}},
        "m2": {"attributes": {"expert_gate_weight_ids": ["g2"]}}}}
    seq._expert_cards = {"g2": ("cuda:1", ())}
    seq._expert_store = SimpleNamespace(device_decode=False)
    segs = seq._expert_segments()
    assert [(k, a, b, bnd) for k, a, b, bnd, _ in segs] == [
        ("replay", 0, 1, ()), ("direct", 1, 2, ()), ("replay", 2, 3, (3, 4)), ("direct", 3, 4, ()), ("tail", 4, 6, (6,))]
    assert [i for i, s in enumerate(segs[2][4]) if not s] == [2]


def test_a_store_whose_residency_runs_on_the_card_replays_the_whole_step():
    seq = TritonSequence.__new__(TritonSequence)
    seq._num_weights, seq._num_inputs = 2, 1
    seq._ops = [_op("aten::mm", [2, 0], [3], "a"), _op("custom::moe_fused", [3], [4], "m0"),
                _op("aten::mul", [4], [5], "c")]
    seq.dag = {"component_name": "model", "ops": {"m0": {"attributes": {"expert_gate_weight_ids": ["g0"]}}}}
    seq._expert_cards = {}
    seq._expert_store = SimpleNamespace(device_decode=True)
    segs = seq._expert_segments()
    assert [(k, a, b) for k, a, b, _bnd, _skip in segs] == [("tail", 0, 3)]
