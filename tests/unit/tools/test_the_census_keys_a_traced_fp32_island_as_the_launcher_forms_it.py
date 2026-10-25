"""The derived census keys an op by the operands the KERNEL receives. For the trace's own fp32 island (an AMP_FP16 op,
not self-managed, traced float32 in a half graph) both engines take the fp32 wrap a contract island takes, so the key
derivation must read it through the same set — or it keys the half dtypes the launcher never forms (Allegro-TI2V's
and CogVideoX's conditioning conv: depthwise_conv2d_kernel keyed fp16 in the table, launched fp32, chain52 2026-10-25;
the Apple class alike, the Mac 10:30). Seen red before `traced_island_uids` existed (the keys came out at the compute
dtype: the one-model re-derivation of CogVideoX-2b on 79b8c8a8 kept 'fp16','fp16','fp16')."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
from derived_census import traced_island_uids  # noqa: E402


def _op(uid, op_type, ins, outs):
    return {"op_uid": uid, "op_type": op_type, "input_dtypes": ins, "output_dtypes": outs}


OPS = {
    "aten.convolution::1": _op("aten.convolution::1", "aten::convolution", ["float32", "float32"], ["float32"]),
    "aten.convolution::2": _op("aten.convolution::2", "aten::convolution", ["float16", "float16"], ["float16"]),
    "aten.conv2d::0": _op("aten.conv2d::0", "aten::conv2d", ["float32", "float32"], ["float32"]),   # self-managed
    "aten.bmm::0": _op("aten.bmm::0", "aten::bmm", ["float32", "float32"], ["float32"]),            # self-managed
    "aten.matmul::0": _op("aten.matmul::0", "aten::matmul", ["float32", "float32"], ["float32"]),
    "aten.layer_norm::0": _op("aten.layer_norm::0", "aten::layer_norm", ["float32", "float32"], ["float32"]),  # AMP_FP32
    "aten._to_copy::0": _op("aten._to_copy::0", "aten::_to_copy", ["float32"], ["float16"]),
}


def test_the_island_set_is_wrap_op_s_order():
    for graph, c in (("float16", "float16"), ("bfloat16", "bfloat16"), ("float16", "bfloat16")):
        got = traced_island_uids(OPS, graph, c)
        assert got == {"aten.convolution::1", "aten.matmul::0"}, (graph, c, got)


def test_no_island_under_an_fp32_compute_or_graph():
    assert traced_island_uids(OPS, "float16", "float32") == frozenset()
    assert traced_island_uids(OPS, "float32", "float16") == frozenset()
