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


def _t(shape, dtype="float32", **kw):
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
    ws = ActivationProfiler(g).estimate_peak_memory(dtype_bytes=2, widths=w, conv3d_workspace=lambda: 1.0)
    assert ws.peak_bytes - base.peak_bytes == 1 * 4 * 3 * 8 * 8 * 2       # the input's numel at its cast width


def test_a_vendor_that_states_no_conv3d_workspace_is_refused_by_name():
    import pytest
    from neurobrix.core.prism.attention_scratch import vendor_memory_of
    from neurobrix.core.prism.op_workspace import conv3d_workspace_fraction
    assert conv3d_workspace_fraction(vendor_memory_of("nvidia", "volta")) == 0.0
    assert conv3d_workspace_fraction(vendor_memory_of("apple", "apple_m4_pro")) == 1.0
    with pytest.raises(KeyError, match="compiled_conv3d_workspace_input_fraction"):
        conv3d_workspace_fraction({})
