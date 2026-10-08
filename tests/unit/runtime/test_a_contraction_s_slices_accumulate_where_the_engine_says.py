"""A contraction split over the axis it reduces is summed where the DtypeEngine says, in both engines.

A stretch run in slices of a token axis (`core/strategies/chunked_piece`) sums the partials of a
contraction over the slices. The dtype of that sum is a dtype decision, so it belongs to the
DtypeEngine (`accumulation_dtype`), never to the pass dtype the partials arrive in: float32 for a
half partial, the dtype itself for float32 and float64, the sum stored once in the op's own dtype.

The rule is one pure function per engine module (`contraction_accumulator_dtype`, torch-free twins
in core/dtype/engine.py and triton/dtype.py) plus its width form for Prism
(`contraction_accumulator_bytes`); the executor answers with the engine its mode runs; Prism prices
the accumulator at that width.

Injections (seen red, then restored green — STATE.md of the op_tiler campaign, 2026-10-05):
  A. the Triton twin returns the dtype itself for a half dtype -> the twin test red.
  B. `LayerPartitioner._accumulator_bytes` prices the store width -> the pricing test red.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
from neurobrix.core.dtype import engine as DE
from neurobrix.kernels.nbx_tensor import NBXDtype
from neurobrix.triton import dtype as TD

FLOATS = ("float16", "bfloat16", "float32", "float64")
WIDTH = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}


def test_the_two_engines_hold_one_rule():
    for name in FLOATS:
        assert DE.contraction_accumulator_dtype(name) == TD.contraction_accumulator_dtype(name), name
        assert TD.contraction_accumulator_bytes(WIDTH[name]) == WIDTH[
            TD.contraction_accumulator_dtype(name)], name
    assert TD.contraction_accumulator_dtype("float16") == "float32"
    assert TD.contraction_accumulator_dtype("bfloat16") == "float32"
    for bad in ("int64", "bool", "complex64"):
        for f in (DE.contraction_accumulator_dtype, TD.contraction_accumulator_dtype):
            with pytest.raises(ValueError, match=bad):
                f(bad)
    with pytest.raises(ValueError, match="1-byte"):
        TD.contraction_accumulator_bytes(1)


def test_each_engine_answers_in_its_own_dtype_type():
    aten = DE.DtypeEngine(torch.float16)
    tri = TD.TritonDtypeEngine(NBXDtype.float16, has_native_bf16=False, has_fp64=False)
    for name in FLOATS:
        a = aten.accumulation_dtype(getattr(torch, name))
        t = tri.accumulation_dtype(getattr(NBXDtype, name))
        assert isinstance(a, torch.dtype) and isinstance(t, NBXDtype)
        assert str(a).replace("torch.", "") == t.name == TD.contraction_accumulator_dtype(name)


@pytest.mark.parametrize("mode", ["sequential", "compiled", "triton", "triton_sequential"])
def test_the_executor_asks_the_engine_its_mode_runs(mode):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prism"))
    import test_a_stretch_over_the_rung_runs_in_slices_of_its_token_axis as S
    ex = S._CpuExecutor(family="image", vendor="nvidia", arch="volta", device="cpu",
                        dtype="bfloat16", mode=mode)
    ex.load_graph_from_dict(S._graph())
    half = NBXDtype.bfloat16 if mode.startswith("triton") else torch.bfloat16
    got = ex.accumulation_dtype(half)
    want = NBXDtype.float32 if mode.startswith("triton") else torch.float32
    assert got == want and type(got) is type(want), (mode, got)


def test_prism_prices_the_accumulator_at_the_engine_s_width():
    """K^T V of the linear-attention block, [B, D, D], at a 2-byte compute dtype: the running sum
    and its replacement in float32 (4 bytes each) plus the partial cast to float32."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prism"))
    import test_a_stretch_over_the_rung_runs_in_slices_of_its_token_axis as S
    from neurobrix.core.prism.layer_partition import LayerPartitioner
    sm = {"s0": S.B, "s1": 4096}
    p = LayerPartitioner(S._half_graph_of(S._graph, "float16"), {w: S.D * 2 for w in S.WEIGHTS},
                         symbol_map=sm, compute_dtype_bytes=2)
    numel = S.B * S.D * S.D
    assert p._accumulator_bytes("aten.bmm::0::out_0", sm) == numel * (4 + 4 + 4)
    p4 = LayerPartitioner(S._graph(), {w: S.D * 4 for w in S.WEIGHTS}, symbol_map=sm,
                          compute_dtype_bytes=4)
    assert p4._accumulator_bytes("aten.bmm::0::out_0", sm) == numel * (4 + 4)
