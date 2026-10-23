"""What a temporal-stream chunk holds while it runs, priced as the engines hold it (mochi's VAE, 2026-10-17).

Three figures the activation profile missed, each measured on a card running mochi-1-preview's VAE chunks:

  * the chunk's INPUTS — its states and its slice of the clip. A graph input exists before the first op, so the
    profile never counted it; the run held all of them for the whole chunk. Priced 13.1 GB, measured 18.2 GB with
    5 GB of states. The runner now hands them to the run, which frees each at its last use; the profile counts them
    from before the first op to that use (`handed_inputs=True`).
  * the operand copies of the compiled engine's precision wrappers (`DtypeEngine.compile_op`): a conv's fp32 input
    cast to fp16 (1.4 GB), an fp16 matmul operand upcast to fp32 (1.2 GB, the op the run died at). They live while
    the op runs: `runtime_widths` returns them (`ExecutedWidths.casts`) and the profile adds them to the op's in-op
    figure. The Triton engine's wrappers copy alike (`TritonDtypeEngine.wrap_op`).
  * a state's width: it enters the next chunk as a SEAM, in the dtype its producer ran at (`seams=`), never at
    its traced dtype.

What each test would do if the code were wrong: `handed_inputs` ignored — the peak equals the no-input figure (red);
the casts not returned or not added — the two estimates are equal (red); the seam seed ignored — the width is the
traced 2 bytes (red).
"""
from __future__ import annotations

from neurobrix.core.prism.profiler import ActivationProfiler
from neurobrix.core.prism.runtime_widths import conservative_contract, runtime_widths

NONE = conservative_contract("test: no contract")


def _t(shape, dtype=None, **kw):
    """A traced tensor. A component INPUT is traced at the component's dtype (float16 here) unless the test says
    otherwise: it enters at the compute dtype (`runtime_widths`: an input traced wider enters at its width, the rule
    of 2026-10-18); a seam keeps its producer's traced dtype; anything else is traced float32 as these graphs were."""
    if dtype is None:
        dtype = "float16" if kw.get("is_input") and not kw.get("seam_alias_of") else "float32"
    return {"shape": list(shape), "dtype": dtype, **kw}


def _op(uid, op_type, ins, outs, **attrs):
    attrs.setdefault("args", [{"type": "tensor", "tensor_id": t} for t in ins])
    return {"op_uid": uid, "op_type": op_type, "input_tensor_ids": list(ins),
            "output_tensor_ids": list(outs), "attributes": attrs}


def _dag(tensors, ops, inputs, outputs, graph="float32"):
    return {"tensors": tensors, "ops": {o["op_uid"]: o for o in ops},
            "execution_order": [o["op_uid"] for o in ops],
            "input_tensor_ids": list(inputs), "output_tensor_ids": list(outputs), "torch_dtype": graph}


def _widths(dag, engine, **kw):
    return runtime_widths(dag, "float16", engine, attention_operands="narrowest", has_native_bf16=False,
                          stores_fp64=False, contract=NONE, **kw)


def _norm_then_conv():
    """An uncontracted group norm upcasts its fp16 operands (AMP_FP32) and keeps an fp32 output, which the next conv
    (AMP lower precision) casts to fp16 while it runs: 1024 elements, 2048 bytes; the conv's wide output makes it
    the peak."""
    T = {"input::z": _t([1, 16, 8, 8], is_input=True), "param::g": _t([16]), "param::b": _t([16]),
         "param::w": _t([64, 16, 1, 1]), "n": _t([1, 16, 8, 8]), "m": _t([1, 1]), "r": _t([1, 1]),
         "c": _t([1, 64, 8, 8])}
    O = [_op("gn::0", "aten::native_group_norm", ["input::z", "param::g", "param::b"], ["n", "m", "r"]),
         _op("conv::0", "aten::convolution", ["n", "param::w"], ["c"])]
    return _dag(T, O, ["input::z"], ["c"])


def test_the_compiled_engine_s_cast_copies_are_priced_while_the_op_runs():
    g = _norm_then_conv()
    w = _widths(g, "compiled")
    assert w["n"] == 4, "the uncontracted norm keeps fp32"
    assert w.casts.get("conv::0") == [("n", 2)], w.casts
    assert w.casts.get("gn::0") == [("input::z", 4), ("param::g", 4), ("param::b", 4)], w.casts
    with_casts = ActivationProfiler(g).estimate_peak_memory(dtype_bytes=2, widths=w)
    without = ActivationProfiler(g).estimate_peak_memory(dtype_bytes=2, widths=dict(w))
    assert with_casts.peak_op_uid == "conv::0"
    assert with_casts.peak_bytes - without.peak_bytes == 1024 * 2


def test_the_triton_engines_copy_as_their_wrappers_do():
    """The Triton dtype engine's wrappers copy too (`TritonDtypeEngine.wrap_op`: the fp32-internal wrap of the norm,
    the lower-precision wrap of the conv — mochi's Triton chunk ran 1.2 GB over a price that had none)."""
    g = _norm_then_conv()
    for engine in ("triton", "triton_sequential"):
        w = _widths(g, engine)
        assert w.casts.get("conv::0") == [("n", 2)], (engine, w.casts)
        # its weights are bound in fp32 once at load (`fp32_constant_names`): only the activation is copied
        assert w.casts.get("gn::0") == [("input::z", 4)], (engine, w.casts)


def test_a_handed_input_is_live_from_the_first_op_to_its_last_use():
    """x (4096 fp16 bytes) read by the first op only, then a small chain: handed, the peak holds x beside the first
    op's output; not handed (the caller's), it does not. (`neg` is in no AMP class: no cast copy.)"""
    T = {"input::x": _t([2048], is_input=True), "a": _t([2048]), "b": _t([16])}
    O = [_op("neg::0", "aten::neg", ["input::x"], ["a"]), _op("neg::1", "aten::neg", ["a"], ["b"])]
    g = _dag(T, O, ["input::x"], ["b"])
    w = _widths(g, "compiled")
    handed = ActivationProfiler(g).estimate_peak_memory(dtype_bytes=2, widths=w, handed_inputs=True)
    caller = ActivationProfiler(g).estimate_peak_memory(dtype_bytes=2, widths=w)
    assert handed.peak_op_uid == "neg::0"
    assert handed.peak_bytes == 2048 * w["input::x"] + 2048 * w["a"]
    assert caller.peak_bytes == 2048 * w["a"] + 16 * w["b"]           # x is not the run's: a and b only


def test_a_state_enters_as_the_seam_its_producer_made():
    T = {"input::s": _t([64], dtype="float16", is_input=True, seam_alias_of="stream.state_out::0::out_0"),
         "o": _t([64], dtype="float16")}
    g = _dag(T, [_op("neg::0", "aten::neg", ["input::s"], ["o"])], ["input::s"], ["o"], graph="float16")
    assert _widths(g, "compiled")["input::s"] == 2                       # traced, no producer known
    assert _widths(g, "compiled", seams={"input::s": "float32"})["input::s"] == 4
    assert _widths(g, "triton", seams={"input::s": "float32"})["input::s"] == 4


def test_a_conv3d_is_priced_with_the_workspace_its_vendor_states():
    """MPS's conv3d takes about its input's size beside input and output (the Mac's measurement, 2026-10-17: VACE's
    encoder chunk 19.3 GB on Metal against 10.9 GB priced); the vendor file states the fraction
    (`memory.compiled_conv3d_workspace_input_fraction`), the profile adds it on the input as the conv reads it (at its
    cast width). Red if the workspace is not added, or added on the wrong tensor or width."""
    T = {"input::x": _t([1, 4, 3, 8, 8], is_input=True, seam_alias_of="test"), "param::w": _t([4, 4, 1, 1, 1]),
         "c": _t([1, 4, 3, 8, 8])}
    g = _dag(T, [_op("conv::0", "aten::convolution", ["input::x", "param::w"], ["c"])], ["input::x"], ["c"])
    w = _widths(g, "compiled")
    assert w.casts.get("conv::0") == [("input::x", 2)], w.casts           # the fp32 seam cast to fp16 by its wrapper
    base = ActivationProfiler(g).estimate_peak_memory(dtype_bytes=2, widths=w)
    def one_input(x_shape, w_shape, attrs, x_width, out_width):          # the vendor rule at fraction 1.0
        n = 1
        for d in x_shape:
            n *= d
        return n * x_width
    ws = ActivationProfiler(g).estimate_peak_memory(dtype_bytes=2, widths=w, conv3d_workspace=one_input)
    assert ws.peak_bytes - base.peak_bytes == 1 * 4 * 3 * 8 * 8 * 2       # the input's numel at its cast width


def test_the_triton_conv3d_is_priced_with_its_folds():
    """The Triton engines run conv3d as kt conv2d launches over the folded frames, holding the input fold, the output
    fold and the accumulator at once (`tiling_sizes.conv3d_one_shot_peak`): 3 output-sized buffers measured beside a
    1,755 MB output on Wan2.1's VAE (2026-10-17). The same function decides the op's chunking; the profile prices from
    it. Red if the profile adds anything else than that function's figure."""
    from neurobrix.core.module.tiling_sizes import conv3d_one_shot_peak
    from neurobrix.core.prism import PrismSolver
    from types import SimpleNamespace
    T = {"input::x": _t([1, 4, 6, 8, 8], is_input=True, seam_alias_of="test"), "param::w": _t([4, 4, 3, 3, 3]),
         "c": _t([1, 4, 4, 6, 6])}
    op = _op("conv::0", "aten::convolution", ["input::x", "param::w"], ["c"], stride=[1, 1, 1], padding=[0, 0, 0],
             dilation=[1, 1, 1])
    g = _dag(T, [op], ["input::x"], ["c"])
    S = PrismSolver(); S._mode = "triton"
    prof = SimpleNamespace(devices=[SimpleNamespace(memory_mb=16384, architecture="volta", brand="nvidia", index=0)])
    hook = S._conv3d_workspace(prof)
    w = _widths(g, "triton")
    base = ActivationProfiler(g).estimate_peak_memory(dtype_bytes=2, widths=w)
    priced = ActivationProfiler(g).estimate_peak_memory(dtype_bytes=2, widths=w, conv3d_workspace=hook)
    x_width = dict(w.casts.get("conv::0", [])).get("input::x", w["input::x"])                 # the conv reads its cast copy's width
    expect = conv3d_one_shot_peak([1, 4, 6, 8, 8], [4, 4, 3, 3, 3], [1, 1, 1], [0, 0, 0], [1, 1, 1], x_width, w["c"])
    assert priced.peak_bytes - base.peak_bytes == expect
    assert expect > 0


def test_a_vendor_that_states_no_conv3d_workspace_is_refused_by_name():
    import pytest
    from neurobrix.core.prism.attention_scratch import vendor_memory_of
    from neurobrix.core.prism.op_workspace import conv3d_workspace_fraction
    assert conv3d_workspace_fraction(vendor_memory_of("nvidia", "volta")) == 0.0
    assert conv3d_workspace_fraction(vendor_memory_of("apple", "apple_m4_pro")) == 1.0
    with pytest.raises(KeyError, match="compiled_conv3d_workspace_input_fraction"):
        conv3d_workspace_fraction({})


def test_a_conv3d_workspace_stated_per_dtype_is_priced_at_the_conv_s_dtype(monkeypatch):
    """bf16 and fp16 share a width and not a workspace (V100's compiled conv3d at Open-Sora-v2's VAE op: 1.47 x its
    input in fp16, 24 x in bf16 — im2col; 2026-10-23): a vendor file may state the fraction per dtype, and the 2-byte
    conv is priced at its component's half dtype. Red if the table is read without the dtype (a bf16 VAE priced at the
    fp16 figure — Open-Sora's 9.28 GB where MPS held 17), or a dtype the table lacks is priced instead of refused."""
    import pytest
    from types import SimpleNamespace
    from neurobrix.core.prism import PrismSolver
    from neurobrix.core.prism import attention_scratch as AS
    from neurobrix.core.prism.op_workspace import conv3d_workspace_fraction, conv3d_dtype
    table = {"compiled_conv3d_workspace_input_fraction": {"float16": 1.5, "bfloat16": 24.0, "float32": 0.02}}
    assert conv3d_workspace_fraction(table, "bfloat16") == 24.0 and conv3d_workspace_fraction(table, "float16") == 1.5
    with pytest.raises(KeyError, match="float64"):
        conv3d_workspace_fraction(table, "float64")
    assert conv3d_dtype(2, "bfloat16") == "bfloat16" and conv3d_dtype(4, "bfloat16") == "float32"
    monkeypatch.setattr(AS, "vendor_memory", lambda dev: table)
    S = PrismSolver(); S._mode = "compiled"
    prof = SimpleNamespace(devices=[SimpleNamespace(memory_mb=16384, architecture="x", brand="y", index=0)])
    x = [1, 4, 6, 8, 8]
    n = 1 * 4 * 6 * 8 * 8
    assert S._conv3d_workspace(prof, "bfloat16")(x, [4, 4, 3, 3, 3], {}, 2, 2) == int(n * 2 * 24.0)
    assert S._conv3d_workspace(prof, "float16")(x, [4, 4, 3, 3, 3], {}, 2, 2) == int(n * 2 * 1.5)
    assert S._conv3d_workspace(prof, "bfloat16")(x, [4, 4, 3, 3, 3], {}, 4, 4) == int(n * 4 * 0.02)
