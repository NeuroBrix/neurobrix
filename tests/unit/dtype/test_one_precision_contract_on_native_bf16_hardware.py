"""ONE precision contract, both engines (the owner, 2026-10-25 15:26): on hardware that computes bf16 natively the
model's tensors are bf16, weights and activations; fp32 only for a traced island, a calibration island, or a kernel's
internal accumulation that is not stored. The Triton branch kept fp32 past an island because its self-managed GEMMs
and convolutions took their operands as they came and stored wide (2 017 policy sites on Sana's transformer, the Mac,
06:40), and `bmm` / decode rows (M <= 4) stored fp32 on every hardware.

Seen red before the change: wrap_op passed a self-managed `mm` through untouched under bf16 (the operands' fp32 kept),
`matmul_out_dtype(bf16, M=1, force_fp32=True, native_bf16=True)` was float32, and the width model derived a non-island
`bmm` after an island as float32. Under fp16 compute without native bf16 (V100) nothing moves: the in-kernel widening is
the compiled branch's fp32-internal rule there, and the keys of the Volta census stay what they are.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism.runtime_widths import runtime_dtypes, conservative_contract

ISLAND = {"input_dtypes": ["float32", "float32"], "output_dtypes": ["float32"]}
HALF = {"input_dtypes": ["bfloat16", "bfloat16"], "output_dtypes": ["bfloat16"]}


def test_the_store_rule_keeps_half_on_native_bf16_and_fp32_without_it():
    from neurobrix.kernels.launch_keys import matmul_out_dtype
    from neurobrix.kernels.nbx_tensor import NBXDtype as D
    assert matmul_out_dtype(D.bfloat16, 1, True, True) == D.bfloat16        # decode row, forced: still bf16 natively
    assert matmul_out_dtype(D.bfloat16, 4096, False, True) == D.bfloat16
    assert matmul_out_dtype(D.float16, 1, True, False) == D.float32         # V100: fp16 stores fp32 (overflow)
    assert matmul_out_dtype(D.float16, 4096, False, False) == D.float32
    assert matmul_out_dtype(D.float32, 1, True, True) == D.float32          # an fp32 operand (an island) stays


def test_the_triton_wrap_narrows_a_self_managed_op_to_bf16_unless_it_is_an_island(monkeypatch):
    pytest.importorskip("triton")
    from neurobrix.kernels.nbx_tensor import NBXDtype
    from neurobrix.triton import dtype as T
    eng = T.TritonDtypeEngine(NBXDtype.bfloat16, has_native_bf16=True, graph_dtype="bfloat16", stores_fp64=False)
    monkeypatch.setattr(eng, "_wrap_fp32", lambda f: "fp32")
    monkeypatch.setattr(eng, "_wrap_lower_precision", lambda f: "lower")
    f = lambda *a: None
    for op in ("mm", "bmm", "addmm", "conv2d", "_convolution"):
        assert eng.wrap_op(op, f, op_uid=f"aten.{op}::1", op_record=HALF) == "lower", op
        assert eng.wrap_op(op, f, op_uid=f"aten.{op}::0", op_record=ISLAND) == "fp32", op
    # the upsample stays self-managed (it follows its input, which the contract already narrowed)
    assert eng.wrap_op("upsample_nearest2d", f, op_uid="aten.upsample_nearest2d::0", op_record=HALF) is f
    # fp16 compute without native bf16: self-managed as before (the in-kernel widening is the compiled rule there)
    v100 = T.TritonDtypeEngine(NBXDtype.float16, has_native_bf16=False, graph_dtype="float16", stores_fp64=False)
    monkeypatch.setattr(v100, "_wrap_lower_precision", lambda f: "lower")
    assert v100.wrap_op("mm", f, op_uid="aten.mm::1", op_record={"input_dtypes": ["float16", "float16"], "output_dtypes": ["float16"]}) is f


def _t(shape, dtype):
    return {"shape": list(shape), "dtype": dtype}


def _op(uid, op_type, ins, outs, in_d, out_d):
    return {"op_uid": uid, "op_type": op_type, "input_tensor_ids": list(ins), "output_tensor_ids": list(outs),
            "input_dtypes": in_d, "output_dtypes": out_d,
            "attributes": {"args": [{"type": "tensor", "tensor_id": t} for t in ins]}}


def test_the_width_model_derives_the_op_after_an_island_at_the_compute_dtype():
    # an island bmm (rotary: fp32 in, fp32 out) whose output feeds a plain bmm, in a bf16 graph
    T = {"input::p": _t([1, 1, 64, 32], "float32"), "param::f": _t([1, 1, 32, 64], "float32"),
         "rot": _t([1, 1, 64, 64], "float32"), "input::x": _t([1, 1, 64, 64], "bfloat16"), "y": _t([1, 1, 64, 64], "bfloat16")}
    O = [_op("aten.bmm::0", "aten::bmm", ["input::p", "param::f"], ["rot"], ["float32", "float32"], ["float32"]),
         _op("aten.bmm::1", "aten::bmm", ["rot", "input::x"], ["y"], ["bfloat16", "bfloat16"], ["bfloat16"])]
    dag = {"tensors": T, "ops": {o["op_uid"]: o for o in O}, "execution_order": [o["op_uid"] for o in O],
           "input_tensor_ids": ["input::p", "input::x"], "output_tensor_ids": ["y"], "torch_dtype": "bfloat16"}
    casts = {}
    d = runtime_dtypes(dag, "bfloat16", "triton", attention_operands="narrowest", has_native_bf16=True,
                       stores_fp64=False, contract=conservative_contract("t"), tiling=None, casts=casts)
    assert d["rot"] == "float32"                                   # the island keeps fp32
    assert d["y"] == "bfloat16", d                                  # its consumer stores at the compute dtype
    assert ("rot", "bfloat16") in casts.get("aten.bmm::1", [])     # and reads the island's output cast to bf16
    # V100: fp16 compute, no native bf16 — the consumer stores fp32 as before (the hardware gate)
    dag16 = dict(dag, torch_dtype="float16")
    d16 = runtime_dtypes(dag16, "float16", "triton", attention_operands="narrowest", has_native_bf16=False,
                         stores_fp64=False, contract=conservative_contract("t"), tiling=None)
    assert d16["y"] == "float32"
