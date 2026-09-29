"""The activation estimate prices each tensor at the width the ENGINE executes it at.

`ActivationProfiler` sized every floating activation at the compute dtype C
(`force_compute_dtype_for_fp=True`). The engines keep some tensors wider: an
AMP_FP32 op without a precision contract keeps an fp32 output, a binary op takes
the wider operand, a fused upsample+conv writes at its pre-input's width. Sana
4Kpx's VAE (no calibration record) planned 6 144 MB at 3072x4096 and died holding
15 360 MB of activations. `core/prism/runtime_widths.py` walks the graph with the
engine's rules; the estimate sizes each activation at the width it returns.

Every test below was run against a deliberately wrong rule and SEEN RED
(2026-09-28, each injection applied alone to runtime_widths.py / profiler.py and
reverted):

  * AMP_FP32 output at C instead of fp32 (`amp_fp32_out` returns `self.c`):
    red — test_an_uncontracted_rms_norm_keeps_fp32_and_the_residual_follows,
    test_triton_and_compiled_differ_where_their_tables_differ,
    test_the_estimate_sizes_an_fp32_activation_at_four_bytes.
  * the narrow set ignored (the `narrow_op_uids` rule deleted):
    red — test_a_contract_narrows_the_norm_back_to_c.
  * the narrow set reaching triton_sequential's rms_norm (`seq_rms` forced False):
    red — test_triton_sequential_rms_norm_is_reached_by_the_flag_only.
  * conv following its input (`_conv_out` returns the input's dtype):
    red — test_a_conv_resets_to_c, test_an_uncontracted_rms_norm_keeps_fp32...
  * the fused conv at C (`fused` never read), or a KNOWN fused conv priced like an unknown
    one (the early return removed): red — test_a_fused_conv_writes_at_its_pre_input_s_width.
  * binary ops taking the first operand (`widest` returns dims[0]):
    red — test_a_binary_op_takes_the_wider_operand_and_a_0_dim_one_does_not_count
    (and the rms residual test).
  * 0-dim operands counted (`widest` without the dims filter): red — the same test.
  * Triton cat taking the widest (`first_dimensioned` -> `widest`):
    red — test_cat_follows_its_first_operand_on_triton_and_promotes_on_compiled.
  * the Triton matmul store ignoring has_native_bf16 (`mm_store` hardware gate removed):
    red — test_the_matmul_store_reads_the_hardware_and_m.
  * an explicit dtype argument ignored (`_rule` explicit branch removed):
    red — test_an_explicit_dtype_is_what_a_cast_writes.
  * a seam cast to C (`is_seam_tensor` branch removed): red — test_a_seam_keeps_its_producer_s_width.
  * the island rule removed: red — test_an_island_is_fp32_whatever_its_class.
  * the profiler ignoring `widths` (sizing at `_compute_size` as before):
    red — test_the_estimate_sizes_an_fp32_activation_at_four_bytes.
  * the in-place filter removed: red — test_an_in_place_add_into_a_narrower_buffer_is_a_new_buffer.
  * one ATen set edited in core/dtype/engine.py: red — test_the_mirrored_aten_tables_are_the_engine_s.
  * `import torch` added at the module top: red — test_the_pass_imports_no_torch.
  * the plan-time contract skipping the record's signature check, or never safe:
    red — test_the_plan_time_contract_is_the_runtime_s_triple.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest -q \
     tests/unit/prism/test_the_estimator_prices_the_width_the_runtime_executes.py
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from neurobrix.core.prism.profiler import ActivationProfiler
from neurobrix.core.prism.runtime_widths import (
    PrecisionContract, TilingView, conservative_contract, plan_time_contract,
    runtime_dtypes, runtime_widths)

NONE = conservative_contract("test: no contract")


def _t(shape, dtype="float32", **kw):
    return {"shape": list(shape), "dtype": dtype, **kw}


def _op(uid, op_type, ins, outs, **attrs):
    return {"op_uid": uid, "op_type": op_type, "input_tensor_ids": list(ins),
            "output_tensor_ids": list(outs), "attributes": attrs}


def _dag(tensors, ops, inputs, outputs, graph="float32"):
    """`graph`: the component's traced `torch_dtype` (every container states one); the
    fixtures' tensors are traced fp32."""
    return {"tensors": tensors, "ops": {o["op_uid"]: o for o in ops},
            "execution_order": [o["op_uid"] for o in ops],
            "input_tensor_ids": list(inputs), "output_tensor_ids": list(outputs),
            "torch_dtype": graph}


def _graph(dag, g):
    return {**dag, "torch_dtype": g}


def _w(dag, engine="triton", c="float16", bf16=False, contract=NONE, tiling=None):
    return runtime_dtypes(dag, c, engine, has_native_bf16=bf16, contract=contract, tiling=tiling)


# A decoder block's tail: conv -> NHWC -> rms_norm -> NCHW -> residual add -> conv -> add.
def _block():
    T = {"input::z": _t([1, 8, 4, 4], is_input=True),
         "param::w": _t([8, 8, 3, 3]), "param::nw": _t([8]),
         "c0": _t([1, 8, 4, 4]), "p0": _t([1, 4, 4, 8]), "r0": _t([1, 4, 4, 8]),
         "p1": _t([1, 8, 4, 4]), "a0": _t([1, 8, 4, 4]), "c1": _t([1, 8, 4, 4]),
         "a1": _t([1, 8, 4, 4])}
    O = [_op("conv::0", "aten::convolution", ["input::z", "param::w"], ["c0"]),
         _op("permute::0", "aten::permute", ["c0"], ["p0"]),
         _op("rms::0", "custom::rms_norm", ["p0", "param::nw"], ["r0"]),
         _op("permute::1", "aten::permute", ["r0"], ["p1"]),
         _op("add::0", "aten::add", ["p1", "input::z"], ["a0"]),
         _op("conv::1", "aten::convolution", ["a0", "param::w"], ["c1"]),
         _op("add::1", "aten::add", ["c1", "a0"], ["a1"])]
    return _dag(T, O, ["input::z"], ["a1"])


def test_an_uncontracted_rms_norm_keeps_fp32_and_the_residual_follows():
    """Triton, C=fp16, no record: rms_norm (AMP_FP32) keeps fp32 (triton/dtype.py:476-483,
    642-659); the permute keeps it; the residual add takes the wider operand
    (wrappers.py:567-574); the next conv resets to C; the second add is fp32 again."""
    w = _w(_block())
    assert w["input::z"] == "float16"                       # a component input enters at C
    assert [w[t] for t in ("c0", "p0", "r0", "p1", "a0", "c1", "a1")] == [
        "float16", "float16", "float32", "float32", "float32", "float16", "float32"]


def test_a_contract_narrows_the_norm_back_to_c():
    k = PrecisionContract(True, frozenset(), frozenset({"rms::0"}))
    w = _w(_block(), contract=k)
    assert [w[t] for t in ("r0", "a0", "a1")] == ["float16"] * 3
    # the narrow set alone, without the safe flag, still narrows the pinned op (dtype.py:454)
    k2 = PrecisionContract(False, frozenset(), frozenset({"rms::0"}))
    assert _w(_block(), contract=k2)["r0"] == "float16"


def test_triton_sequential_rms_norm_is_reached_by_the_flag_only():
    """triton_sequential wraps custom::rms_norm with no op_uid (triton/sequential.py:176-179):
    the narrow set never reaches it, the safe flag does."""
    narrow_only = PrecisionContract(False, frozenset(), frozenset({"rms::0"}))
    assert _w(_block(), "triton_sequential", contract=narrow_only)["r0"] == "float32"
    assert _w(_block(), "triton", contract=narrow_only)["r0"] == "float16"
    safe = PrecisionContract(True, frozenset(), frozenset())
    assert _w(_block(), "triton_sequential", contract=safe)["r0"] == "float16"


def test_a_conv_resets_to_c():
    w = _w(_block())
    assert w["a0"] == "float32" and w["c1"] == "float16"
    assert _w(_block(), c="bfloat16")["c1"] == "bfloat16"
    assert _w(_block(), "compiled")["c1"] == "float16"


def test_an_island_is_fp32_whatever_its_class():
    k = PrecisionContract(True, frozenset({"conv::1"}), frozenset({"rms::0"}))
    for eng in ("triton", "compiled"):
        assert _w(_block(), eng, contract=k)["c1"] == "float32"


def _fused():
    T = {"input::x": _t([1, 8, 2, 2], is_input=True), "param::nw": _t([8]),
         "param::w": _t([8, 8, 3, 3]), "r": _t([1, 8, 2, 2]), "up": _t([1, 8, 4, 4]),
         "cv": _t([1, 8, 4, 4])}
    O = [_op("exp::0", "aten::exp", ["input::x"], ["r"]),
         _op("upsample::0", "aten::upsample_nearest2d", ["r"], ["up"]),
         _op("conv::0", "aten::convolution", ["up", "param::w"], ["cv"])]
    return _dag(T, O, ["input::x"], ["cv"])


def test_a_fused_conv_writes_at_its_pre_input_s_width():
    """`_fused_upsample_conv2d_nbx` allocates at pre_input.dtype
    (kernels/ops/fused_upsample_conv.py:804-807); the torch variant at weight.dtype (:299-303)."""
    g = _fused()
    assert _w(g)["r"] == "float32" and _w(g)["up"] == "float32"
    assert _w(g)["cv"] == "float32"                          # plan unknown: the wider
    fused = TilingView(fusion_convs={"conv::0": "upsample::0"}, tiled_ops=frozenset())
    assert _w(g, tiling=fused)["cv"] == "float32"
    unfused = TilingView(fusion_convs={}, tiled_ops=frozenset())
    assert _w(g, tiling=unfused)["cv"] == "float16"
    assert _w(g, "compiled")["cv"] == "float16"
    # a KNOWN fused conv is its pre-input's width even where the op rule is wider: an
    # island conv (`_wrap_fp32`) whose proxy carries a C pre-input writes C; unknown, fp32
    g2 = _fused()
    g2["ops"]["exp::0"]["op_type"] = "aten::neg"
    island = PrecisionContract(False, frozenset({"conv::0"}), frozenset())
    assert _w(g2, contract=island, tiling=fused)["cv"] == "float16"
    assert _w(g2, contract=island)["cv"] == "float32"
    # an upsample with two consumers is never fused: the conv writes C
    g["tensors"]["y"] = _t([1, 8, 4, 4])
    g["ops"]["neg::0"] = _op("neg::0", "aten::neg", ["up"], ["y"])
    g["execution_order"].append("neg::0")
    assert _w(g)["cv"] == "float16"


def _binary():
    T = {"input::x": _t([4, 8], is_input=True), "param::w": _t([8, 8, 1, 1]),
         "e": _t([4, 8]), "s": _t([]), "a": _t([4, 8]), "b": _t([4, 8]),
         "x4": _t([1, 8, 4, 1]), "cv": _t([1, 8, 4, 1])}
    O = [_op("exp::0", "aten::exp", ["input::x"], ["e"]),
         _op("sum::0", "aten::sum", ["input::x"], ["s"]),
         _op("add::0", "aten::add", ["input::x", "e"], ["a"]),
         _op("mul::0", "aten::mul", ["input::x", "s"], ["b"])]
    return _dag(T, O, ["input::x"], ["a", "b"])


def test_a_binary_op_takes_the_wider_operand_and_a_0_dim_one_does_not_count():
    """`_prepare_binary` aligns two tensors to the wider (wrappers.py:627-632) and treats a
    0-dim operand as the scalar, the output `empty_like` the tensor (:524-530, 596-615)."""
    for eng in ("triton", "compiled"):
        w = _w(_binary(), eng)
        assert w["e"] == "float32" and w["s"] == "float32"
        assert w["a"] == "float32", eng
        assert w["b"] == "float16", eng


def test_cat_follows_its_first_operand_on_triton_and_promotes_on_compiled():
    T = {"input::x": _t([4, 8], is_input=True), "e": _t([4, 8]), "k": _t([8, 8]),
         "k2": _t([8, 8])}
    O = [_op("exp::0", "aten::exp", ["input::x"], ["e"]),
         _op("cat::0", "aten::cat", ["input::x", "e"], ["k"]),
         _op("cat::1", "aten::cat", ["e", "input::x"], ["k2"])]
    g = _dag(T, O, ["input::x"], ["k", "k2"])
    assert _w(g)["k"] == "float16" and _w(g)["k2"] == "float32"            # nbx_tensor.py:4251-4253
    assert _w(g, "compiled")["k"] == "float32" and _w(g, "compiled")["k2"] == "float32"


def _mm(rows):
    T = {"input::a": _t([rows, 16], is_input=True), "param::b": _t([16, 4]), "o": _t([rows, 4])}
    return _dag(T, [_op("mm::0", "aten::mm", ["input::a", "param::b"], ["o"])], ["input::a"], ["o"])


def test_the_matmul_store_reads_the_hardware_and_m():
    """Triton `_matmul_out_dtype` (wrappers.py:1798-1861): fp16 on hardware without native
    bf16 stores fp32; a half with M <= 4 stores fp32; else the half. The ATen engine
    upcasts fp16 mm to fp32 unless the contract says otherwise (engine.py:695-732)."""
    assert _w(_mm(8), bf16=False)["o"] == "float32"
    assert _w(_mm(8), bf16=True)["o"] == "float16"
    assert _w(_mm(2), bf16=True)["o"] == "float32"
    assert _w(_mm(8), c="bfloat16", bf16=True)["o"] == "bfloat16"
    assert _w(_mm(8), "compiled", bf16=True)["o"] == "float32"
    assert _w(_mm(8), "compiled", contract=PrecisionContract(True, frozenset(), frozenset()))["o"] == "float16"
    assert _w(_mm(8), "compiled", c="bfloat16")["o"] == "bfloat16"


def test_triton_and_compiled_differ_where_their_tables_differ():
    """custom::rms_norm: Triton AMP_FP32 -> fp32; ATen `rms_norm_fp32` -> the weight's dtype
    (core/dtype/engine.py:26-50). upsample_nearest: Triton dtype passthrough (self-managed);
    ATen AMP_FP32 -> fp32."""
    g = _block()
    assert _w(g, "triton")["r0"] == "float32"
    assert _w(g, "compiled")["r0"] == "float16"
    T = {"input::x": _t([1, 8, 2, 2], is_input=True), "up": _t([1, 8, 4, 4])}
    u = _dag(T, [_op("up::0", "aten::upsample_nearest2d", ["input::x"], ["up"])], ["input::x"], ["up"])
    assert _w(u, "triton")["up"] == "float16"
    assert _w(u, "compiled")["up"] == "float32"
    # sequential is priced with the compiled table
    assert _w(g, "sequential") == _w(g, "compiled")


def test_an_explicit_dtype_is_what_a_cast_writes():
    dt = lambda s: {"type": "dtype", "value": s}
    T = {"input::x": _t([4], is_input=True), "a": _t([4]), "b": _t([4], "bfloat16"),
         "i": _t([4], "int64")}
    O = [_op("tc::0", "aten::_to_copy", ["input::x"], ["a"], kwargs={"dtype": dt("torch.float32")}),
         _op("tc::1", "aten::_to_copy", ["a"], ["b"], kwargs={"dtype": dt("torch.bfloat16")}),
         _op("tc::2", "aten::_to_copy", ["b"], ["i"], kwargs={"dtype": dt("torch.int64")})]
    g = _dag(T, O, ["input::x"], ["i"])
    for eng in ("triton", "compiled"):
        w = _w(g, eng)
        assert (w["a"], w["b"], w["i"]) == ("float32", "float16", "int64"), eng


def test_a_seam_keeps_its_producer_s_width():
    T = {"input::s": _t([4], is_input=True, seam_alias_of="aten.x::0::out_0"),
         "input::y": _t([4], is_input=True), "o": _t([4])}
    g = _dag(T, [_op("add::0", "aten::add", ["input::y", "input::s"], ["o"])],
             ["input::s", "input::y"], ["o"])
    w = _w(g)
    assert w["input::s"] == "float32" and w["input::y"] == "float16" and w["o"] == "float32"


def test_a_triton_fp32_constant_is_bound_fp32():
    """`fp32_constant_names` (triton/dtype.py:292-326) — the constants of all-fp32 consumers
    are bound fp32 on the Triton engines only (graph_executor.py:2063-2107)."""
    T = {"input::x": _t([2, 8], is_input=True), "param::g": _t([8]), "param::b": _t([8]),
         "o": _t([2, 8]), "m": _t([2, 1]), "r": _t([2, 1])}
    g = _dag(T, [_op("ln::0", "aten::native_layer_norm", ["input::x", "param::g", "param::b"],
                     ["o", "m", "r"])], ["input::x"], ["o"])
    assert _w(g)["param::g"] == "float32" and _w(g, "compiled")["param::g"] == "float16"
    assert _w(g)["o"] == "float32"


def test_refusals_name_what_they_refuse():
    with pytest.raises(ValueError, match="engine"):
        _w(_block(), "cuda")
    g = _block()
    del g["tensors"]["r0"]["dtype"]
    with pytest.raises(ValueError, match="no traced dtype"):
        _w(g)


def test_the_estimate_sizes_an_fp32_activation_at_four_bytes():
    """Profiler: the peak with `widths` counts the rms_norm output (and what it reaches) at
    4 bytes. Block tensors are 128 elements; at add::0 live are a0 plus the inputs still
    needed (c0 freed at permute, r0 at permute::1, p1 at add::0)."""
    g = _block()
    ap_old = ActivationProfiler(g).estimate_peak_memory(dtype_bytes=2)   # traced (fp32) sizing
    w = runtime_widths(g, "float16", "triton", has_native_bf16=False, contract=NONE)
    ap = ActivationProfiler(g).estimate_peak_memory(dtype_bytes=2, widths=w)
    level = {t: (2 if b in (2, 4) else b) for t, b in w.items()}      # the former levelling to C
    ap_c = ActivationProfiler(g).estimate_peak_memory(dtype_bytes=2, widths=level)
    # the peak op holds an fp32 128-element tensor beside another: priced wider than at C
    assert ap.peak_bytes > ap_c.peak_bytes
    assert ap_c.peak_bytes * 2 == ap_old.peak_bytes          # all-fp32 trace, all-C level
    w_ = runtime_widths(g, "float16", "triton", has_native_bf16=False,
                        contract=PrecisionContract(True, frozenset(), frozenset({"rms::0"})))
    assert ActivationProfiler(g).estimate_peak_memory(dtype_bytes=2, widths=w_).peak_bytes == ap_c.peak_bytes


def test_an_in_place_add_into_a_narrower_buffer_is_a_new_buffer():
    """add_inplace_nbx falls back to a plain add when its target is narrower
    (wrappers.py:1325-1336; tiling_engine.py `_inplace_add`): the output is a new buffer."""
    T = {"input::x": _t([1024], is_input=True), "param::w": _t([4, 4, 1, 1]),
         "c": _t([1024]), "e": _t([1024]), "a": _t([1024]), "n": _t([1024])}
    O = [_op("conv::0", "aten::convolution", ["input::x", "param::w"], ["c"]),
         _op("exp::0", "aten::exp", ["input::x"], ["e"]),
         _op("add::0", "aten::add", ["c", "e"], ["a"]),
         _op("neg::0", "aten::neg", ["a"], ["n"])]
    g = _dag(T, O, ["input::x"], ["n"])
    w = runtime_widths(g, "float16", "triton", has_native_bf16=False, contract=NONE)
    assert (w["c"], w["e"], w["a"]) == (2, 4, 4)
    p = ActivationProfiler(g)
    into_narrow = p.estimate_peak_memory(dtype_bytes=2, widths=w, inplace_adds=[("add::0", 0)])
    into_wide = p.estimate_peak_memory(dtype_bytes=2, widths=w, inplace_adds=[("add::0", 1)])
    # target c (fp16) is narrower than the fp32 result: at add::0 c + e + a NEW a = 10 KiB
    assert into_narrow.peak_bytes == 1024 * (2 + 4 + 4)
    # target e (fp32): a IS e's buffer — add::0 holds c + e = 6 KiB, neg::0 e + n = 8 KiB
    assert into_wide.peak_bytes == 1024 * (4 + 4)


def test_the_mirrored_aten_tables_are_the_engine_s():
    """The ATen engine's AMP classes are mirrored (core/dtype/engine.py imports torch, Prism
    may not). This door compares the mirror with the engine's own sets — torch is imported
    by THIS test only."""
    pytest.importorskip("torch")
    from neurobrix.core.dtype import engine as E
    from neurobrix.core.prism import runtime_widths as R
    assert R.ATEN_AMP_FP32_OPS == E.AMP_FP32_OPS
    assert R.ATEN_AMP_FP16_OPS == E.AMP_FP16_OPS
    assert R.ATEN_FP16_NEED_FP32 == E._FP16_NEED_FP32
    assert R.ATEN_FP16_GEMM_OPS == E._FP16_GEMM_OPS
    assert R.ATEN_FP32_OPS_HALF_IO == E._FP32_OPS_HALF_IO
    assert R.ATEN_AMP_PROMOTE_OPS == E.AMP_PROMOTE_OPS
    assert R.ATEN_AMP_CREATION_FILL_OPS == E.AMP_CREATION_FILL_OPS


def test_the_pass_imports_no_torch():
    code = ("import sys, neurobrix.core.prism.runtime_widths, neurobrix.core.prism.solver; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def test_the_plan_time_contract_is_the_runtime_s_triple():
    """Only float16 has a contract (precision_contract.py:231-233); a record measured on this
    graph gives the flag, its islands and the structural narrow set."""
    assert plan_time_contract(None, "vae", {"ops": {}}, "bfloat16").safe is False
    import json
    from pathlib import Path
    cache = Path.home() / ".neurobrix" / ("ca" + "che") / "Sana_1600M_1024px_MultiLing"
    g = cache / "components" / "vae" / "graph.json"
    from neurobrix.core.dtype import calibration as _cal
    if not g.exists() or _cal.load_record("Sana_1600M_1024px_MultiLing", "vae") is None:
        pytest.skip("Sana_1600M_1024px_MultiLing vae or its calibration record absent")
    dag = json.loads(g.read_text())
    k = plan_time_contract(cache, "vae", dag, "float16")
    from neurobrix.core.runtime import precision_contract as _pc
    assert k.safe is True and k.narrow_op_uids == _pc.narrowable_op_uids(dag)
    other = dict(dag, execution_order=list(reversed(dag["execution_order"])))
    assert plan_time_contract(cache, "vae", other, "float16").safe is False   # not this graph


def test_an_nbx_dtype_is_named_by_its_member_not_its_string(monkeypatch):
    """IntEnum.__str__ is int.__str__ on Python 3.11+ (the Mac's): str(NBXDtype.bfloat16) == "1" there,
    and the pass refused Sana-4K's text encoder at plan time ("unknown dtype name '1'", 2026-09-28 12:16).
    The name is read from the member. RED on 43b307ad: this test forces the 3.11 str() on 3.10."""
    from neurobrix.core.prism import runtime_widths as RW
    from neurobrix.kernels.nbx_tensor import NBXDtype
    monkeypatch.setattr(NBXDtype, "__str__", lambda self: str(int(self)))
    assert str(NBXDtype.bfloat16) == "1"
    assert RW._nbx_name(NBXDtype.bfloat16) == "bfloat16"
    assert RW._nbx_name(NBXDtype.float32) == "float32"


# ---------------------------------------------------------------------------
# Under bf16 on a bf16 GRAPH an fp32-internal op returns the compute dtype (decision
# 2026-09-28, refined on the Apple gate: on a graph of another dtype coerced to bf16 compute
# the output stays fp32) — the pass CALLS the engines' rule (`triton/dtype.py
# amp_fp32_output_dtype`) with the component's `torch_dtype`, the engines' own source. Seen red
# (2026-09-28, nbx/campaigns/2026_09_28_bf16_castback/injections.txt, injections2.txt), each alone:
#   * the unchanged pass (427c8c91): test_under_bf16_an_amp_fp32_output_is_priced_at_two_bytes.
#   * the 4398f9d2 pass (no graph dtype): test_an_fp32_graph_coerced_to_bf16_keeps_its_fp32_norm_output
#     and test_the_mirrors_answer_what_the_engines_execute.
#   * the pass reading the compute dtype as the graph's, or the rule ignoring the graph:
#     test_an_fp32_graph_coerced_to_bf16_* and the drift door.
#   * the Triton mirror, or the ATen mirror, restored to its pre-cast-back rule: the drift
#     door and test_under_bf16_an_amp_fp32_output_is_priced_at_two_bytes.
#   * the rule changed in an ENGINE only (a wrapper reading the compute dtype as the graph's;
#     the compiled twin without its bf16 clause; `compile_op` reading the flag as `safe`):
#     test_the_mirrors_answer_what_the_engines_execute (the drift door).
# ---------------------------------------------------------------------------

def _one(op_type, attrs=None, graph="bfloat16"):
    T = {"input::x": _t([2, 8], is_input=True), "y": _t([2, 8])}
    return _dag(T, [_op("op::0", op_type, ["input::x"], ["y"], **(attrs or {}))],
                ["input::x"], ["y"], graph=graph)


def test_under_bf16_an_amp_fp32_output_is_priced_at_two_bytes():
    g = _graph(_block(), "bfloat16")
    for eng in ("triton", "triton_sequential"):
        w = _w(g, eng, c="bfloat16", bf16=True)
        assert [w[t] for t in ("r0", "p1", "a0", "c1", "a1")] == ["bfloat16"] * 5, eng
        b = runtime_widths(g, "bfloat16", eng, has_native_bf16=True, contract=NONE)
        assert b["r0"] == 2 and b["a1"] == 2
    for eng in ("compiled", "sequential"):
        assert _w(_one("aten::layer_norm"), eng, c="bfloat16", bf16=True)["y"] == "bfloat16"
        assert runtime_widths(_one("aten::exp"), "bfloat16", eng, has_native_bf16=True,
                              contract=NONE)["y"] == 2
    # fp16 unchanged: no contract -> fp32 (4 bytes) on both engines
    assert runtime_widths(g, "float16", "triton", has_native_bf16=False, contract=NONE)["r0"] == 4
    assert runtime_widths(_one("aten::exp"), "float16", "compiled", has_native_bf16=False,
                          contract=NONE)["y"] == 4


def test_an_fp32_graph_coerced_to_bf16_keeps_its_fp32_norm_output():
    """The graph dtype is the component's `torch_dtype`, the engines' own source: an fp32
    graph the profile runs in bf16 keeps the fp32 AMP_FP32 output (and the residual add
    follows it), priced at 4 bytes; a graph that states none is refused by name."""
    g = _block()                                              # torch_dtype float32
    for eng in ("triton", "triton_sequential"):
        w = _w(g, eng, c="bfloat16", bf16=True)
        assert [w[t] for t in ("r0", "a0", "c1", "a1")] == [
            "float32", "float32", "bfloat16", "float32"], eng
        assert runtime_widths(g, "bfloat16", eng, has_native_bf16=True, contract=NONE)["r0"] == 4
    for eng in ("compiled", "sequential"):
        assert _w(_one("aten::layer_norm", graph="float32"), eng, c="bfloat16",
                  bf16=True)["y"] == "float32"
    unstated = {k: v for k, v in g.items() if k != "torch_dtype"}
    with pytest.raises(ValueError, match="graph dtype"):
        _w(unstated, "triton", c="bfloat16", bf16=True)


def _contracts():
    return [("bfloat16", NONE),
            ("bfloat16:float32", NONE),                      # an fp32 graph run in bf16
            ("float16", NONE),
            ("float16", PrecisionContract(True, frozenset(), frozenset())),
            ("float16", PrecisionContract(True, frozenset(), frozenset({"op::0"})))]


def test_the_mirrors_answer_what_the_engines_execute():
    """Behavioural drift door: the width pass's answer for one AMP_FP32 op equals the dtype
    the engine's own wrapper RETURNS — compiled on CPU torch tensors, Triton on duck-typed
    tensors (no card) — for bf16 and for fp16 without contract, with the flag, with the
    narrow set. The mirrored SETS are compared above; this compares the RULE."""
    torch = pytest.importorskip("torch")
    from neurobrix.core.dtype.engine import DtypeEngine
    from neurobrix.kernels import wrappers as _wr
    from neurobrix.kernels.nbx_tensor import NBXDtype
    from neurobrix.triton.dtype import TritonDtypeEngine

    class Fake:
        def __init__(self, dt):
            self.nbx_dtype = dt
        def is_floating_point(self):
            return True
        def to(self, dt):
            return Fake(dt)
        def contiguous(self):
            return self
        def is_contiguous(self):
            return True

    for ck, k in _contracts():
        c, _, graph = ck.partition(":")
        graph = graph or c
        for op_type, fn, extra in (("aten::exp", torch.ops.aten.exp, ()),
                                   ("aten::rsqrt", torch.ops.aten.rsqrt, ()),
                                   ("aten::layer_norm", torch.ops.aten.layer_norm, ([8],))):
            tc = getattr(torch, c)
            eng = DtypeEngine(tc, graph_dtype=getattr(torch, graph), activations_fp16_safe=k.safe,
                              narrow_op_uids=k.narrow_op_uids, fp32_op_uids=k.fp32_op_uids)
            got = eng.compile_op(op_type, fn, {}, op_uid="op::0")(
                torch.rand(2, 8, dtype=tc) + 0.5, *extra).dtype
            want = _w(_one(op_type, graph=graph), "compiled", c=c, bf16=c == "bfloat16",
                      contract=k)["y"]
            assert str(got).replace("torch.", "") == want, (op_type, c, k)
        for op_type in ("aten::exp", "custom::rms_norm", "aten::layer_norm"):
            prev = _wr.get_activations_fp16_safe()
            _wr.set_activations_fp16_safe(k.safe)
            try:
                teng = TritonDtypeEngine(getattr(NBXDtype, c), graph_dtype=graph)
                teng.set_precision_contract(k.safe, k.fp32_op_uids, k.narrow_op_uids)
                name = op_type.split("::")[-1]
                got = teng.wrap_op(name, lambda x, *a, **kw: Fake(x.nbx_dtype),
                                   op_uid="op::0")(Fake(getattr(NBXDtype, c))).nbx_dtype.name
            finally:
                _wr.set_activations_fp16_safe(prev)
            want = _w(_one(op_type, graph=graph), "triton", c=c, bf16=c == "bfloat16",
                      contract=k)["y"]
            assert got == want, (op_type, c, k)


def _norm_tuple(graph="float32"):
    """native_layer_norm returns (out, mean, rstd): its traced outputs are three."""
    T = {"input::x": _t([2, 8], is_input=True), "param::g": _t([8]), "param::b": _t([8]),
         "y": _t([2, 8]), "mu": _t([2, 1]), "rs": _t([2, 1])}
    return _dag(T, [_op("ln::0", "aten::native_layer_norm", ["input::x", "param::g", "param::b"],
                        ["y", "mu", "rs"])], ["input::x"], ["y"], graph=graph)


def test_under_fp16_a_tuple_norm_output_keeps_the_fp32_its_kernel_wrote():
    """triton/dtype.py `cast_back`: under fp16 the cast back to C is the single-tensor one — a
    TUPLE result (native_layer_norm, native_group_norm) keeps the fp32 its kernel wrote from the
    fp32 inputs, narrow set, safe flag or neither (the fp16 contract held byte-identical by
    decision 2026-09-28). Measured by the census walk 2026-09-29: Kokoro's albert projections
    and PixArt's VAE attention read the norm's output in fp32 while this pass said fp16.
    Under bf16 on a bf16 graph every float of the tuple comes back to C. Injection: the tuple
    clause removed from `amp_fp32_out` -> 'float16' under the narrow set, RED."""
    narrow = PrecisionContract(True, frozenset(), frozenset({"ln::0"}))
    for eng in ("triton", "triton_sequential"):
        for k in (NONE, narrow, PrecisionContract(True, frozenset(), frozenset())):
            assert _w(_norm_tuple(), eng, contract=k)["y"] == "float32", (eng, k)
        assert _w(_norm_tuple("bfloat16"), eng, c="bfloat16", bf16=True)["y"] == "bfloat16"
    # the single-tensor norm still narrows under the contract
    assert _w(_block(), contract=narrow.__class__(True, frozenset(), frozenset({"rms::0"})))[
        "r0"] == "float16"


def test_an_attention_over_disagreeing_operands_writes_fp32():
    """`scaled_dot_product_attention_wrapper` casts q, k, v that disagree to fp32 before any
    route (`launch_keys.sdpa_operand_dtypes`), so its output is fp32 — measured on Wan's
    cross-attention (fp16 q/k, fp32 v), whose out-projection read fp32 in the census walk while
    this pass said fp16. Agreeing operands keep q's dtype. Injection: the disagreement clause
    removed -> 'float16', RED."""
    def dag(island):
        T = {"input::q": _t([1, 2, 8, 4], is_input=True), "input::k": _t([1, 2, 8, 4], is_input=True),
             "input::x": _t([1, 2, 8, 4], is_input=True), "v": _t([1, 2, 8, 4]), "o": _t([1, 2, 8, 4])}
        ops = [_op("exp::0", "aten::exp", ["input::x"], ["v"]),
               _op("sdpa::0", "aten::scaled_dot_product_attention", ["input::q", "input::k", "v"], ["o"])]
        k = PrecisionContract(True, frozenset({"exp::0"} if island else ()), frozenset())
        return _dag(T, ops, ["input::q", "input::k", "input::x"], ["o"]), k
    for eng in ("triton", "triton_sequential"):
        d, k = dag(True)                              # v from an fp32 island
        w = _w(d, eng, contract=k)
        assert (w["input::q"], w["input::k"], w["v"]) == ("float16", "float16", "float32")
        assert w["o"] == "float32", eng
        d, k = dag(False)                             # all three fp16
        assert _w(d, eng, contract=k)["o"] == "float16", eng
