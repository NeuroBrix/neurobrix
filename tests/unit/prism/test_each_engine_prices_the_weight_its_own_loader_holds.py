"""Prism prices each engine's weights at the dtype THAT engine's loader holds them in.

`runtime_widths._AtenRules` inherited the Triton loader's rule (`stored_dtype_in_compute`) for the
COMPILED engine's weights: an fp16-stored weight under an fp32 plan stayed fp16 in the width pass,
while the ATen loader (`core/io/weight_loader.WeightLoader`) converts every floating weight to the
placement dtype. Every activation downstream of such a weight (an embedding's output, a widest-operand
op) was priced at half the width the compiled engine executes it at. Embedded constants had the same
defect: the compiled engine binds them through `DtypeEngine.convert_constant` (an fp32 constant in a
half graph is the vendor's own and is KEPT), not through the Triton loader's rule (fp32 -> C).

Each assertion below compares the width pass with the LOADER'S OWN function run on a real tensor —
never with a literal dtype.

SEEN RED (2026-10-12, each injection applied alone, then reverted and re-run green):
  * `_AtenRules.weight_dtype` deleted (the compiled engine back on the Triton rule): red —
    test_an_fp16_weight_under_an_fp32_plan_is_priced_as_the_compiled_loader_holds_it,
    test_an_embedded_constant_is_priced_as_the_compiled_engine_binds_it.
  * the shard-wide shortcut restored in `WeightLoader._load_with_pinned_dma` (no tensor of a shard
    converted when its first floating tensor already has the plan's dtype): red —
    test_a_mixed_shard_is_held_tensor_by_tensor.
"""
from __future__ import annotations

import base64

import pytest
import torch

from neurobrix.core.dtype.config import compiled_weight_dtype, is_floating_name
from neurobrix.core.dtype.engine import DtypeEngine
from neurobrix.core.io.weight_loader import WeightLoader
from neurobrix.core.prism.runtime_widths import conservative_contract, runtime_dtypes
from neurobrix.core.runtime.expert_files import held_dtype
from neurobrix.kernels.nbx_tensor import parse_dtype
from neurobrix.triton.weight_loader import stored_dtype_in_compute

NONE = conservative_contract("test: no record")


def _name(dt) -> str:
    return str(dt).replace("torch.", "")


def _dag(weight_dtype: str, graph: str = "float32", **weight_meta):
    """input ids -> embedding(weight) -> out: the output follows the weight (widest float operand)."""
    T = {"input::ids": {"shape": [1, 3], "dtype": "int64", "is_input": True},
         "param::w": {"shape": [8, 4], "dtype": weight_dtype, **weight_meta},
         "e0": {"shape": [1, 3, 4], "dtype": weight_dtype}}
    op = {"op_uid": "emb::0", "op_type": "aten::embedding", "input_tensor_ids": ["param::w", "input::ids"],
          "output_tensor_ids": ["e0"], "attributes": {"args": [
              {"type": "tensor", "tensor_id": "param::w"}, {"type": "tensor", "tensor_id": "input::ids"}]}}
    return {"tensors": T, "ops": {"emb::0": op}, "execution_order": ["emb::0"],
            "input_tensor_ids": ["input::ids"], "output_tensor_ids": ["e0"], "torch_dtype": graph}


def _widths(dag, engine, c):
    return runtime_dtypes(dag, c, engine, has_native_bf16=False, attention_operands="widest", stores_fp64=False,
                          contract=NONE)


def _compiled_loader_holds(stored: str, c: str) -> str:
    """What the ATen loader's own conversion produces for one weight (`_convert_weights_dtype`)."""
    loader = WeightLoader.__new__(WeightLoader)
    t = torch.zeros(8, 4, dtype=getattr(torch, stored))
    return _name(loader._convert_weights_dtype({"w": t}, getattr(torch, c))["w"].dtype)


@pytest.mark.parametrize("stored,c", [("float16", "float32"), ("bfloat16", "float32"),
                                      ("float32", "float16"), ("bfloat16", "float16"),
                                      ("float16", "float16")])
def test_an_fp16_weight_under_an_fp32_plan_is_priced_as_the_compiled_loader_holds_it(stored, c):
    held = _compiled_loader_holds(stored, c)
    for engine in ("compiled", "sequential"):
        w = _widths(_dag(stored), engine, c)
        assert w["param::w"] == held, (engine, stored, c)
        assert w["e0"] == held                       # the activation it feeds follows it
    # The expert-file views read the same rule.
    assert held_dtype(stored, c, "compiled").name == held


@pytest.mark.parametrize("stored,c", [("float16", "float32"), ("bfloat16", "float32"),
                                      ("float32", "float16"), ("bfloat16", "float16")])
def test_the_triton_engine_keeps_its_own_loader_s_rule(stored, c):
    held = stored_dtype_in_compute(parse_dtype(stored), parse_dtype(c)).name
    for engine in ("triton", "triton_sequential"):
        assert _widths(_dag(stored), engine, c)["param::w"] == held
    assert held_dtype(stored, c, "triton").name == held


def test_the_two_engines_differ_where_their_loaders_do():
    """fp16 stored, fp32 plan: Triton keeps fp16 (in-kernel PROMOTE_B), compiled loads fp32."""
    assert _widths(_dag("float16"), "triton", "float32")["param::w"] == "float16"
    assert _widths(_dag("float16"), "compiled", "float32")["param::w"] == "float32"


@pytest.mark.parametrize("traced,c,graph", [("float32", "float16", "float16"),   # vendor fp32 constant: kept
                                            ("float32", "float16", "float32"),   # graph-dtype constant: at C
                                            ("float16", "float32", "float16"),
                                            ("bfloat16", "float16", "float32")])
def test_an_embedded_constant_is_priced_as_the_compiled_engine_binds_it(traced, c, graph):
    eng = DtypeEngine(getattr(torch, c), graph_dtype=getattr(torch, graph))
    bound = _name(eng.convert_constant(torch.zeros(8, 4, dtype=getattr(torch, traced))).dtype)
    dag = _dag(traced, graph=graph, constant=True, constant_data=base64.b64encode(b"x").decode())
    assert _widths(dag, "compiled", c)["param::w"] == bound


def test_a_mixed_shard_is_held_tensor_by_tensor(tmp_path):
    """A shard whose FIRST floating tensor already has the plan's dtype: every other floating
    tensor is still held at the plan's dtype — what Prism prices."""
    from safetensors.torch import save_file
    path = tmp_path / "shard_000.safetensors"
    save_file({"a.weight": torch.zeros(4, dtype=torch.float32),
               "b.weight": torch.zeros(4, dtype=torch.bfloat16),
               "c.ids": torch.zeros(4, dtype=torch.int64)}, str(path))
    loader = WeightLoader.__new__(WeightLoader)
    loader._only = None
    got = loader._load_with_pinned_dma(str(path), "cpu", torch.float32, False)
    for k, stored in (("a.weight", "float32"), ("b.weight", "bfloat16"), ("c.ids", "int64")):
        assert _name(got[k].dtype) == compiled_weight_dtype(stored, "float32"), k
    assert _name(got["b.weight"].dtype) == "float32"


def test_the_floating_predicate_is_torch_s():
    """`is_floating_name` (torch-free, read by Prism) answers what `torch.dtype.is_floating_point`
    answers, over every dtype torch defines (drift door)."""
    dts = {d for d in vars(torch).values() if isinstance(d, torch.dtype)}
    assert dts
    assert [d for d in dts if is_floating_name(_name(d)) != d.is_floating_point] == []
