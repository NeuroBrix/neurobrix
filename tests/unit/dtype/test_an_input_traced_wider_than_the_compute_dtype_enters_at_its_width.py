"""A graph input traced WIDER than the compute dtype enters at its traced width; one at or below it enters at the compute dtype.

Both engines cast every runtime input through one sentence of the DtypeEngine
(`TritonDtypeEngine._target_dtype_for_input`; `tensor_resolver.get_input_dtype` + `DtypeEngine.storage_dtype` on the ATen
branch), and Prism prices the seed with the same sentence (`runtime_widths`). The sentence used to be "a floating input →
the compute dtype": a float64 timestep fed to a bf16 component on Metal entered as bf16, which holds integers exactly only
to 256 — 777 became 776, the sinusoidal embedding was another timestep's, and Allegro rendered its halo (the Mac's
block-0 comparison of 2026-10-18). fp16 holds integers to 2048, so the V100 never showed it. The two CFG engines also
cast the timestep to the state's dtype before the executor; they now hand it in the scheduler's dtype.

What these tests would do if the code were wrong: the rule tests read fp16 for an fp32 input (red); the executed door
feeds 2049.0 and 4097.0 in float32 to an fp16 component — exact in fp32, rounded to 2048 and 4096 by an fp16 entry — and
reads them back negated (red at 2048).
"""
from __future__ import annotations

import pytest
import torch

from neurobrix.core.prism.runtime_widths import conservative_contract, runtime_dtypes

NONE = conservative_contract("test: no contract")


def _meta(dtype):
    return {"input::t": {"dtype": dtype, "shape": [2], "is_input": True, "input_name": "t"}}


@pytest.mark.parametrize("traced,compute,stores_fp64,expected", [
    ("float32", "float16", False, "float32"),       # wider: enters at its width
    ("float64", "float16", False, "float32"),       # wider, the branch holds no fp64: its storage form
    ("float64", "float16", True, "float64"),
    ("float16", "bfloat16", False, "bfloat16"),     # at the compute width: the compute dtype (as before)
    ("float16", "float16", False, "float16"),
    ("int64", "float16", False, "int64"),           # non-floating: the graph's own
])
def test_the_triton_engine_s_entry_rule(traced, compute, stores_fp64, expected):
    pytest.importorskip("triton")
    from neurobrix.kernels.nbx_tensor import NBXDtype
    from neurobrix.triton.dtype import TritonDtypeEngine
    eng = TritonDtypeEngine(NBXDtype[compute], has_native_bf16=True, graph_dtype=compute, stores_fp64=stores_fp64)
    assert eng._target_dtype_for_input("input::t", _meta(traced)) == NBXDtype[expected]


@pytest.mark.parametrize("traced,compute,expected", [
    ("float32", torch.float16, torch.float32), ("float64", torch.float16, torch.float64),
    ("float16", torch.bfloat16, torch.bfloat16), ("float16", torch.float16, torch.float16),
    ("int64", torch.float16, torch.int64),
])
def test_the_compiled_resolver_s_entry_rule(traced, compute, expected):
    from neurobrix.core.runtime.graph.execution_context import ExecutionContext
    from neurobrix.core.runtime.graph.tensor_resolver import TensorResolver
    ctx = ExecutionContext(dag={"tensors": _meta(traced)}, device="cpu", dtype=compute)
    assert TensorResolver(ctx).get_input_dtype("t") == expected


@pytest.mark.parametrize("engine", ["compiled", "sequential", "triton", "triton_sequential"])
@pytest.mark.parametrize("traced,stores_fp64,expected", [
    ("float32", False, "float32"), ("float64", False, "float32"), ("float64", True, "float64"), ("float16", False, "float16")])
def test_prism_prices_the_seed_by_the_same_rule(engine, traced, stores_fp64, expected):
    dag = {"tensors": {**_meta(traced), "o": {"dtype": traced, "shape": [2]}},
           "ops": {"neg::0": {"op_type": "aten::neg", "input_tensor_ids": ["input::t"], "output_tensor_ids": ["o"],
                              "attributes": {"args": [{"type": "tensor", "tensor_id": "input::t"}]}}},
           "execution_order": ["neg::0"], "input_tensor_ids": ["input::t"], "output_tensor_ids": ["o"], "torch_dtype": "float16"}
    got = runtime_dtypes(dag, "float16", engine, attention_operands="narrowest", has_native_bf16=False, contract=NONE,
                         stores_fp64=stores_fp64)
    assert got["input::t"] == expected


# --- the executed door -------------------------------------------------------------------------------------------
def _tensor(tid, **extra):
    base = {"tensor_id": tid, "shape": [2], "dtype": "float32", "device": "cpu", "producer_op_uid": None,
            "output_index": None, "consumer_op_uids": [], "is_parameter": False, "is_input": False,
            "weight_name": None, "input_name": None, "output_name": None}
    base.update(extra)
    return base


def _graph():
    """neg(t): an fp32 input [2] into an fp16 component; `neg` is in no precision class, so it runs at its operand's width."""
    tensors = {"input::t": _tensor("input::t", is_input=True, input_name="t", consumer_op_uids=["aten.neg::0"]),
               "aten.neg::0::out_0": _tensor("aten.neg::0::out_0", producer_op_uid="aten.neg::0", output_index=0,
                                             output_name="out")}
    ops = {"aten.neg::0": {"op_uid": "aten.neg::0", "op_type": "aten::neg", "input_tensor_ids": ["input::t"],
                           "output_tensor_ids": ["aten.neg::0::out_0"],
                           "attributes": {"args": [{"type": "tensor", "tensor_id": "input::t"}], "kwargs": {}}}}
    return {"component_name": "transformer", "format": "tensor_dag", "version": "0.1", "torch_dtype": "float16",
            "input_names": ["t"], "tensors": tensors, "ops": ops, "execution_order": ["aten.neg::0"],
            "input_tensor_ids": ["input::t"], "output_tensor_ids": ["aten.neg::0::out_0"]}


def _cuda() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# On the host the engines place fp16 components in fp32, so a CPU run cannot narrow the input and the door would be
# green whatever the rule: it runs on a card only (seen red on 738bfb137: -2048, -4096 in every CUDA mode).
MODES = [("compiled", "cuda"), ("triton", "cuda"), ("triton_sequential", "cuda")]


@pytest.mark.skipif(not _cuda(), reason="the entry narrowing happens on a card: the host places fp16 components in fp32")
@pytest.mark.parametrize("mode,device", MODES)
def test_an_fp32_input_value_survives_the_entry_of_an_fp16_component(mode, device):
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    ex = GraphExecutor(family="video", vendor="nvidia", arch="volta", device=device, dtype="float16", mode=mode)
    ex.load_graph_from_dict(_graph())
    ex._component_name = "transformer"
    x = torch.tensor([2049.0, 4097.0], dtype=torch.float32)      # exact in fp32, 2048 / 4096 in fp16
    if mode.startswith("triton"):
        from neurobrix.kernels.nbx_tensor import NBXTensor
        t = NBXTensor.from_numpy(x.numpy()).to_cuda()
    else:
        t = x.to(device)
    out = ex.run({"t": t})["out"]
    got = torch.from_numpy(out.numpy()).float() if hasattr(out, "to_cuda") else out.detach().float().cpu()
    assert got.tolist() == [-2049.0, -4097.0], f"{mode}/{device}: the input was narrowed at entry: {got.tolist()}"
