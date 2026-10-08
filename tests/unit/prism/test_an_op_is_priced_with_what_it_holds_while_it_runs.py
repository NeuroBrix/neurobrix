"""Prism prices what an op holds WHILE it runs, from the functions the runtime sizes it with.

`core/prism/op_transients.op_transient_bytes` is the one price of an op's transient: the
TilingEngine's attention route and scores, conv2d band, tiled conv2d and conv3d fold
(`tiling_sizes`, the functions `kernels/launch_keys` and the wrappers call), and the DtypeEngine's
cast copies (`itemsize.cast_copy_bytes` at the engines' AMP execution dtype). The placement walk
(`ActivationProfiler.estimate_peak_memory(transients=, dtypes=)`) adds it to the op's in-op figure.

What this test does if the code were wrong: a price that drifts from the sizing function (a live
count, a route, a band rule, a cast term dropped) differs from the function's own bytes on the
case that exercises it -> red, naming the kind; a walk that forgets the transient reports the
outputs-only peak -> red (seen red with the cast term removed from `op_transient_bytes`, green
again once restored).
"""
from __future__ import annotations

import pytest

from neurobrix.core.dtype.itemsize import cast_copy_bytes, representation_bytes
from neurobrix.core.module import tiling_sizes as ts
from neurobrix.core.prism.op_transients import TransientContext, context_for, op_transient_bytes
from neurobrix.core.prism.profiler import ActivationProfiler

GiB = 1024 ** 3
_VOLTA_MU = {"operand_dtype": "float16", "flash": [{"head_dim_le": 128, "block_m": 64, "block_n": 64,
                                                     "warps_m": 4}]}


def _ctx(**kw):
    base = dict(engine="triton", compute_dtype="float16", sdpa_budget_bytes=GiB, sdpa_min_chunk_rows=128,
                sdpa_max_chunks=16, band_bytes=ts.conv2d_band_bytes())
    base.update(kw)
    return TransientContext(**base)


def _sdpa(q, k, v, dt="float16", ctx=None):
    return op_transient_bytes("u", {"op_type": "aten::scaled_dot_product_attention"}, [q, k, v], [q],
                              [dt, dt, dt], dt, ctx or _ctx())


def _conv(x, w, y, attrs, dt="float16", uid="u", ctx=None):
    return op_transient_bytes(uid, {"op_type": "aten::convolution", "attributes": attrs}, [x, w], [y],
                              [dt, dt], dt, ctx or _ctx())


# --- price == the sizing function's bytes, per transient kind -------------------------------------

def test_attention_math_route():
    B, H, T, D = 1, 8, 1024, 64                      # 32 MiB of scores: under the budget
    assert ts.sdpa_route(B, H, T, T, D, D, GiB, 128, 16) == ("math", 0)
    assert _sdpa([B, H, T, D], [B, H, T, D], [B, H, T, D]) == ts.sdpa_transient_bytes("math", 0, B, H, T, T)
    assert ts.sdpa_transient_bytes("math", 0, B, H, T, T) > 0


def test_attention_chunked_route():
    B, H, T, D = 1, 24, 16156, 64                    # mochi-scale scores: over the budget, chunks fit
    route, rows = ts.sdpa_route(B, H, T, T, D, D, GiB, 128, 64)
    assert route == "chunked" and rows
    got = _sdpa([B, H, T, D], [B, H, T, D], [B, H, T, D], ctx=_ctx(sdpa_max_chunks=64))
    assert got == ts.sdpa_transient_bytes("chunked", rows, B, H, T, T)


def test_attention_flash_route_on_the_matrix_unit():
    B, H, T, D = 1, 24, 16156, 64
    ctx = _ctx(matrix_unit=_VOLTA_MU)
    assert ts.sdpa_route(B, H, T, T, D, D, GiB, 128, 16, unit_flash=True) == ("flash", 0)
    assert _sdpa([B, H, T, D], [B, H, T, D], [B, H, T, D], ctx=ctx) == ts.sdpa_transient_bytes(
        "flash", 0, B, H, T, T)
    # fp32 operands are not the unit's: the route falls back to what the budget says.
    route, rows = ts.sdpa_route(B, H, T, T, D, D, GiB, 128, 16)
    assert _sdpa([B, H, T, D], [B, H, T, D], [B, H, T, D], dt="float32", ctx=ctx) == ts.sdpa_transient_bytes(
        route, rows, B, H, T, T)


def test_conv2d_band_and_its_floor():
    attrs = {"stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 1}
    big = ([1, 128, 4096, 8192], [128, 128, 3, 3], [1, 128, 4096, 8192])          # 8 GiB fp16 output
    small = ([1, 64, 256, 256], [64, 64, 3, 3], [1, 64, 256, 256])
    band = ts.conv2d_band_bytes()
    want = ts.conv2d_band_transient_bytes(1, 128, 8192, 2, 128, 4096, 8192, 2, 3, 1, 1, band)
    assert want > 0 and _conv(*big, attrs) == want
    assert _conv(*small, attrs) == 0 == ts.conv2d_band_transient_bytes(1, 64, 256, 2, 64, 256, 256, 2, 3, 1, 1, band)
    # depthwise runs no band (launch_keys.conv2d_route tests it first)
    dw = ([1, 128, 4096, 8192], [128, 1, 3, 3], [1, 128, 4096, 8192])
    assert _conv(*dw, {**attrs, "groups": 128}) == 0


def test_a_planned_tiled_conv2d():
    attrs = {"stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 1}
    ctx = _ctx(tiled_ops={"c": 8})
    got = op_transient_bytes("c", {"op_type": "aten::convolution", "attributes": attrs},
                             [[1, 128, 2048, 2048], [128, 128, 3, 3], [128]], [[1, 128, 2048, 2048]],
                             ["float16"] * 3, "float16", ctx)
    assert got == ts.tiled_conv2d_transient_bytes(1, 128, 2048, 2048, 2, 128, 2048, 2048, 2, 3, 1, 1, 1, 1,
                                                  True, 8) > 0


@pytest.mark.parametrize("chunked", [False, True])
def test_conv3d_one_shot_and_chunked(chunked):
    # mochi-1-preview's VAE at 7 frames, a decoder conv (shape class of the 11 039 MB Mac case)
    x, w = [1, 256, 9, 240, 424], [256, 256, 3, 3, 3]
    attrs = {"stride": [1, 1, 1], "padding": [0, 1, 1], "dilation": [1, 1, 1], "groups": 1}
    ctx = _ctx(conv3d_chunks=frozenset({"u"}) if chunked else frozenset())
    got = _conv(x, w, [1, 256, 7, 240, 424], attrs, ctx=ctx)
    assert got == ts.conv3d_transient_bytes(x, w, [1, 1, 1], [0, 1, 1], [1, 1, 1], 2, 2, chunked) > 0


def test_an_fp32_op_copies_its_half_inputs_and_holds_its_fp32_result():
    # layer_norm under fp16 compute: the fp32-internal wrap upcasts the input and casts back.
    n = 2 * 4096 * 1536
    got = op_transient_bytes("u", {"op_type": "aten::layer_norm"}, [[2, 4096, 1536]], [[2, 4096, 1536]],
                             ["float16"], "float16", _ctx())
    assert got == cast_copy_bytes("float16", "float32", n) + cast_copy_bytes("float16", "float32", n)
    assert got == 2 * representation_bytes("float32", n)


def test_a_half_op_copies_its_fp32_input_and_a_self_managed_one_copies_nothing():
    n = 4096 * 4096
    got = op_transient_bytes("u", {"op_type": "aten::linear"}, [[4096, 4096], [4096, 4096]], [[4096, 4096]],
                             ["float32", "float16"], "float16", _ctx())
    assert got == cast_copy_bytes("float32", "float16", n)
    assert op_transient_bytes("u", {"op_type": "aten::mm"}, [[4096, 4096], [4096, 4096]], [[4096, 4096]],
                              ["float32", "float16"], "float32", _ctx()) == 0
    # full-precision compute: no AMP wrap, no copy
    assert op_transient_bytes("u", {"op_type": "aten::layer_norm"}, [[4096, 4096]], [[4096, 4096]],
                              ["float32"], "float32", _ctx(compute_dtype="float32")) == 0


def test_the_compiled_engine_prices_casts_and_no_tiling_split():
    attrs = {"stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 1}
    ctx = _ctx(engine="compiled")
    assert _conv([1, 128, 4096, 8192], [128, 128, 3, 3], [1, 128, 4096, 8192], attrs, ctx=ctx) == 0
    n = 2 * 4096 * 1536
    assert op_transient_bytes("u", {"op_type": "aten::layer_norm"}, [[2, 4096, 1536]], [[2, 4096, 1536]],
                              ["float16"], "float16", ctx) == 2 * representation_bytes("float32", n)


def _contract(safe, fp32=(), narrow=()):
    from neurobrix.core.prism.runtime_widths import PrecisionContract
    return PrecisionContract(safe, frozenset(fp32), frozenset(narrow), "test")


def test_the_execution_dtype_reads_the_component_s_precision_contract():
    # The engines' own contract rules (core/dtype/engine.py `_resolve_args`, triton/dtype.py
    # `wrap_op`), on the record `runtime_dtypes` reads: a component with a safe fp16 contract
    # runs its matmuls and fp16-IO norms at fp16 under the compiled engine — no fp32 copy of a
    # weight (Allegro's transformer, DeepSeek-Coder-V2-Lite's lm_head, 2026-10-09).
    n = 2048 * 102400
    mm = ({"op_type": "aten::mm"}, [[23, 2048], [2048, 102400]], [[23, 102400]], ["float16", "float16"])
    ln = ({"op_type": "aten::native_layer_norm"}, [[2, 4096, 1536]], [[2, 4096, 1536]], ["float16"])

    def price(case, ctx, uid="u"):
        op, ins, outs, dts = case
        return op_transient_bytes(uid, op, ins, outs, dts, "float16", ctx)

    unsafe = _ctx(engine="compiled", contract=_contract(False))
    safe = _ctx(engine="compiled", contract=_contract(True))
    # compiled, no contract: the V100 fp32 upcast copies both operands and holds the fp32 result
    assert price(mm, unsafe) == (representation_bytes("float32", 23 * 2048) + representation_bytes("float32", n)
                                 + representation_bytes("float32", 23 * 102400))
    assert price(mm, safe) == 0 and price(ln, safe) == 0
    assert price(ln, unsafe) == 2 * representation_bytes("float32", 2 * 4096 * 1536)
    # an island computes in fp32 under any contract, either engine
    island = _contract(True, fp32={"isl"})
    assert price(mm, _ctx(engine="compiled", contract=island), uid="isl") == price(mm, unsafe)
    assert price(mm, _ctx(contract=island), uid="isl") == price(mm, unsafe)
    # Triton: the contract moves no compute but the islands (mm is self-managed, the norm is fp32)
    assert price(mm, _ctx(contract=_contract(True))) == 0
    assert price(ln, _ctx(contract=_contract(True))) == price(ln, unsafe)


def test_the_context_reads_the_attention_keys_the_wrapper_reads():
    from neurobrix.core.config.loader import get_vendor_config
    from neurobrix.core.prism.structure import DeviceBrand, DeviceSpec

    class _P:
        devices = [DeviceSpec(0, "V100-16", 16384, "7.0", ["float32", "float16"], "volta", DeviceBrand.NVIDIA)]
    mem = get_vendor_config("nvidia", "volta")["memory"]
    ctx = context_for(_P(), "triton", "torch.float16")
    assert ctx.compute_dtype == "float16"
    assert ctx.sdpa_budget_bytes == ts.sdpa_device_scores_budget(
        int(mem["sdpa_math_max_scores_bytes"]), float(mem["sdpa_math_scores_device_fraction"]), 16384)
    assert ctx.sdpa_min_chunk_rows == int(mem["sdpa_math_min_chunk_rows"])
    assert ctx.sdpa_max_chunks == int(mem["sdpa_math_max_chunks"])
    assert ctx.band_bytes == ts.conv2d_band_bytes()


# --- the walk adds the transient to the op's in-op figure -----------------------------------------

def _attention_dag(B, H, T, D):
    sh = [B, H, T, D]
    t = lambda: {"shape": list(sh), "dtype": "float16"}
    return {
        "tensors": {"q": t(), "k": t(), "v": t(), "o": t()},
        "ops": {"a": {"op_type": "aten::scaled_dot_product_attention",
                      "input_tensor_ids": ["q", "k", "v"], "output_tensor_ids": ["o"]}},
        "execution_order": ["a"], "input_tensor_ids": ["q", "k", "v"], "output_tensor_ids": ["o"],
    }


def test_the_walk_adds_each_op_s_transient_to_its_peak():
    B, H, T, D = 1, 8, 1024, 64
    dag = _attention_dag(B, H, T, D)
    prof = ActivationProfiler(dag)
    dtypes = {k: "float16" for k in dag["tensors"]}
    bare = prof.estimate_peak_memory(dtype_bytes=2)
    priced = prof.estimate_peak_memory(dtype_bytes=2, transients=_ctx(), dtypes=dtypes)
    want = ts.sdpa_transient_bytes("math", 0, B, H, T, T)
    assert priced.transient_by_op == {"a": want}
    assert priced.peak_bytes == bare.peak_bytes + want


def test_a_walk_with_transients_and_no_dtypes_is_refused():
    with pytest.raises(ValueError, match="runtime dtypes"):
        ActivationProfiler(_attention_dag(1, 1, 8, 8)).estimate_peak_memory(dtype_bytes=2, transients=_ctx())


# --- the ladder is data -----------------------------------------------------------------------------

def test_the_memory_ladder_is_read_from_tiling_yml():
    from neurobrix.core.config.loader import get_tiling_policy
    from neurobrix.core.prism.memory_budget import memory_ladder_mb
    rungs = get_tiling_policy()["memory_tiers"]["ladder_gb"]
    assert memory_ladder_mb() == sorted({int(g) * 1024 for g in rungs})
