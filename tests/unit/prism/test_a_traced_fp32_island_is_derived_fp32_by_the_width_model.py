"""The width model (Prism's plan prices, the DERIVED census's keys) mirrors the engines' dtype policies op by op.
Both policies keep a vendor's fp32 island inside a half graph in fp32 (`core/dtype/traced_island`, 2026-10-25): an
AMP_FP16 op whose traced floating inputs and output are all float32 computes and stores fp32 on any hardware. The
mirror must say the same, or the census derives the half key the launcher never forms — Allegro-TI2V's conditioning
conv (`aten.convolution::0`, traced fp32) missed its fp32 depthwise key at the gate of 2026-10-25 08:25 (chain52).

Seen red on the tree that carried the engines' predicate but not the mirror (model-status-nvidia afe108cc):
the conv derived at the compute dtype, its operands cast to it.
"""
from __future__ import annotations

from neurobrix.core.prism.runtime_widths import runtime_dtypes, runtime_widths, conservative_contract

NONE = conservative_contract("test: no contract")


def _t(shape, dtype):
    return {"shape": list(shape), "dtype": dtype}


def _op(uid, op_type, ins, outs, in_dtypes, out_dtypes):
    return {"op_uid": uid, "op_type": op_type, "input_tensor_ids": list(ins), "output_tensor_ids": list(outs),
            "input_dtypes": list(in_dtypes), "output_dtypes": list(out_dtypes),
            "attributes": {"args": [{"type": "tensor", "tensor_id": t} for t in ins]}}


def _dag(graph):
    # A half component: an fp32 island conv (mask traced fp32 by the vendor) and a plain half conv beside it.
    T = {"input::m": _t([1, 1, 1, 96], "float32"), "param::k": _t([1, 1, 1, 2], "float32"),
         "input::x": _t([1, 8, 4, 4], graph), "param::w": _t([8, 8, 3, 3], graph),
         "m1": _t([1, 1, 1, 48], "float32"), "y": _t([1, 8, 4, 4], graph)}
    O = [_op("conv::0", "aten::convolution", ["input::m", "param::k"], ["m1"], ["float32", "float32"], ["float32"]),
         _op("conv::1", "aten::convolution", ["input::x", "param::w"], ["y"], [graph, graph], [graph])]
    return {"tensors": T, "ops": {o["op_uid"]: o for o in O}, "execution_order": [o["op_uid"] for o in O],
            "input_tensor_ids": ["input::m", "input::x"], "output_tensor_ids": ["m1", "y"], "torch_dtype": graph}


def _casts(dag, engine, c, bf16):
    w = runtime_widths(dag, c, engine, attention_operands="narrowest", has_native_bf16=bf16, stores_fp64=False,
                       contract=NONE, tiling=None)
    return w


def test_the_island_conv_is_derived_fp32_in_both_engines_on_fp16_and_bf16_hardware():
    for graph, c, bf16 in (("float16", "float16", False), ("bfloat16", "bfloat16", True), ("float16", "bfloat16", True)):
        dag = _dag(graph)
        for engine in ("triton", "compiled"):
            d = runtime_dtypes(dag, c, engine, attention_operands="narrowest", has_native_bf16=bf16,
                               stores_fp64=False, contract=NONE, tiling=None)
            assert d["m1"] == "float32", (engine, graph, c, d["m1"])        # the island: fp32 compute and store
            assert d["y"] == c, (engine, graph, c, d["y"])                  # the half conv beside it: the compute dtype


def test_an_fp32_graph_has_no_island():
    d = runtime_dtypes(_dag("float32"), "float32", "triton", attention_operands="narrowest", has_native_bf16=False,
                       stores_fp64=False, contract=NONE, tiling=None)
    assert d["m1"] == "float32" and d["y"] == "float32"
