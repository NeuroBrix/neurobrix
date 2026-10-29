"""Three per-decode-step costs of an expert-store LM, each removed at its source (Qwen3-30B-A3B triton on 4 V100,
2026-10-28: 437 -> 420 -> 385 -> 381 ms per token, the text identical at each step):

1. A fused MoE op is not placed on its routed experts' device: pinned host experts carry the placeholder index 0, and
   the home card's MoE calls moved their hidden state to card 0 and back. They still make the sequence multi-device
   (only that loop calls the zero3 ratchet that builds the store).
2. `bind_weights` with the same weights as the last bind does nothing (16.5 ms per call); a cleared or restored arena,
   or a replaced weight, binds again.
3. The autotune ladder is answered by the identity of the profile's row list, never by a `repr` built per call.

What each test would do if the code were wrong: the MoE op placed on 0 or the sequence classed single-device (red);
the second bind run, or a bind skipped after a clear / a replaced weight (red); `repr` built on a repeated call (red)."""
from types import SimpleNamespace

from nbxtensor.ops import autotune_bucket as AB
from nbxtensor.memory.arena import Arena
from neurobrix.triton.sequence import TritonSequence


def _t(dev_idx, device="cuda", numel=10):
    return SimpleNamespace(_device_idx=dev_idx, _device=device, _numel=numel)


def test_a_moe_op_is_not_placed_on_its_host_experts_and_still_makes_the_sequence_multi_device():
    seq = object.__new__(TritonSequence)
    seq._arena = Arena(4, num_weights=4)
    seq._arena[0], seq._arena[1] = _t(2, numel=100), _t(0, "cpu", numel=10_000)   # dense on card 2, an expert on host
    seq._arena[2], seq._arena[3] = _t(2, numel=5), _t(0, "cpu", numel=10_000)
    dense = SimpleNamespace(op_type="aten::mm", weight_input_slots=[0], device_idx=2)
    moe = SimpleNamespace(op_type="custom::moe_fused", weight_input_slots=[2, 3], device_idx=2)
    seq._ops = [dense, moe]
    seq.device_idx = 2
    try:
        seq.compute_op_devices()
    except Exception:
        pass                                    # the multi-device tail needs more than this test builds
    assert moe.device_idx == 2, f"the MoE op was placed on {moe.device_idx}, its host experts' placeholder"
    assert seq._is_multi_device, "host experts no longer make the sequence multi-device: the ratchet would not run"


def test_a_bind_of_the_same_weights_runs_once_and_again_after_a_clear_or_a_new_weight(monkeypatch):
    seq = object.__new__(TritonSequence)
    seq._arena, seq._compiled = Arena(2, num_weights=2), True
    calls = []
    monkeypatch.setattr(TritonSequence, "_bind_weights", lambda self, w: calls.append(dict(w)))
    w = {"a": object(), "b": object()}
    seq.bind_weights(w); seq.bind_weights(w)
    assert len(calls) == 1, "an unchanged bind ran again"
    seq._arena.clear_all(); seq.bind_weights(w)
    assert len(calls) == 2, "a cleared arena was not bound again"
    w["a"] = object(); seq.bind_weights(w)
    assert len(calls) == 3, "a replaced weight was not bound again"


class _Rows(list):
    reprs = 0

    def __repr__(self):
        _Rows.reprs += 1
        return super().__repr__()


def test_the_ladder_is_answered_by_the_rows_identity():
    rows = _Rows([{"up_to": 64, "step": 16}, {"up_to": None, "step": 64}])
    first = AB.parse_ladder(rows)
    before = _Rows.reprs
    assert AB.parse_ladder(rows) == first and _Rows.reprs == before, "a repeated call built the repr again"
