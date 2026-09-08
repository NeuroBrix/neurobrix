"""A constant whose every consumer computes in fp32 is bound in fp32 once at load: the
fp32-internal wrap cast layer_norm's weight and bias to fp32 at EVERY call (whisper-large-v3-
turbo: 900 a transcription, the copy census of 2026-09-08). The rule reads the graph: every
consumer an AMP_FP32 op or a contract island, none narrowed by the contract."""
from __future__ import annotations

from neurobrix.triton.dtype import fp32_constant_names


def _dag():
    return {"ops": {
        "aten.layer_norm::0": {"op_type": "aten::layer_norm", "input_tensor_ids": ["x", "param::ln.weight", "param::ln.bias"]},
        "aten.rms_norm::0": {"op_type": "aten::rms_norm", "input_tensor_ids": ["y", "param::norm.weight"]},
        "aten.linear::0": {"op_type": "aten::linear", "input_tensor_ids": ["x", "param::fc.weight", "param::fc.bias"]},
        "aten.layer_norm::1": {"op_type": "aten::layer_norm", "input_tensor_ids": ["z", "param::shared.weight"]},
        "aten.mul::0": {"op_type": "aten::mul", "input_tensor_ids": ["z", "param::shared.weight"]},
        "aten.group_norm::0": {"op_type": "aten::group_norm", "input_tensor_ids": ["w", "buffer::gn.weight"]},
        "aten.layer_norm::2": {"op_type": "aten::layer_norm", "input_tensor_ids": ["v", "param::narrowed.weight"]},
    }}


def test_constants_consumed_only_by_fp32_internal_ops_are_named():
    names = fp32_constant_names(_dag())
    assert names == {"ln.weight", "ln.bias", "norm.weight", "gn.weight", "narrowed.weight"}


def test_a_constant_with_one_consumer_outside_the_fp32_set_is_not():
    assert "shared.weight" not in fp32_constant_names(_dag())      # layer_norm AND a mul
    assert "fc.weight" not in fp32_constant_names(_dag())


def test_a_narrowed_consumer_still_qualifies():
    """The contract narrows an fp32-class op's OUTPUT to the compute dtype; the engine's wrap
    still pre-casts its inputs to fp32 (`force_cast_back=True` on the same wrap), so its
    constants are cast per call just the same — whisper's 1,318 narrowed encoder ops."""
    names = fp32_constant_names(_dag(), narrow_op_uids={"aten.layer_norm::2"})
    assert "narrowed.weight" in names and "ln.weight" in names


def test_an_islanded_op_counts_as_fp32():
    names = fp32_constant_names(_dag(), fp32_op_uids={"aten.linear::0"})
    assert {"fc.weight", "fc.bias"} <= names
