"""Under fp16, an op the vendor ran in fp32 inside a half forward keeps its fp32 output in Triton.

The vendor's timestep embedding computes its frequency table in fp32 inside an fp16 forward:
`exp(-log(10000) * arange(half) / half)`, multiplied by t (up to 999), then sin/cos. The graph
records those ops at float32 (`output_dtypes`). The Triton wrapper cast the result of every
fp32-internal op (the AMP_FP32 class and the fp16 `div`) back to fp16 whenever the component's
`activations_fp16_safe` flag was set, so the table was narrowed: CogVideoX-2b's
`transformer/aten.sin::0` departed from the PyTorch-sequential oracle by rel 0.21 (drift walk
2026-10-05, nbx/campaigns/2026_10_04_written/cogdrift/drift.txt), its input `aten.div::0`
read fp16 in the engine where the graph says float32. The compiled engine never narrows on the
flag (it passes safe=False and narrows per op from the record's narrow set), so the two engines
disagreed on the same container.

The rule now: an op traced float32 on a half graph is narrowed by the narrow set only, never by
the flag — in `amp_fp32_output_dtype` (both twins), in the Triton wrapper (which reads the op's
record), and in Prism's width pass. An op traced at the half dtype is unchanged.

What these tests do if the code is wrong: with the floor removed from the Triton twin,
test_the_rule_* and test_the_triton_wrapper_keeps_* are red (fp16 returned); with the wrapper
not handing the record's dtype to the rule, test_the_triton_wrapper_keeps_* is red.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest -q \
     tests/unit/runtime/test_a_vendor_fp32_island_is_not_narrowed_by_the_flag.py
"""
from __future__ import annotations

import pytest

from neurobrix.kernels.nbx_tensor import NBXDtype
from neurobrix.triton import dtype as T


def test_the_rule_keeps_a_traced_fp32_op_fp32_under_the_flag():
    for g in ("float16", "bfloat16"):
        assert T.amp_fp32_output_dtype("float16", g, True, False, "float32") == "float32"
        # the record's narrow set still narrows it, as the compiled engine does
        assert T.amp_fp32_output_dtype("float16", g, True, True, "float32") == "float16"
        assert T.amp_fp32_output_dtype("float16", g, False, True, "float32") == "float16"


def test_the_rule_is_unchanged_for_an_op_traced_at_the_half_dtype():
    for traced in ("float16", None):
        assert T.amp_fp32_output_dtype("float16", "float16", True, False, traced) == "float16"
        assert T.amp_fp32_output_dtype("float16", "float16", False, False, traced) == "float32"
    # an fp32 graph coerced to fp16 compute: every op is traced fp32; the flag rules as before
    assert T.amp_fp32_output_dtype("float16", "float32", True, False, "float32") == "float16"


def test_the_traced_dtype_is_read_from_the_record_s_top_level():
    assert T.traced_output_dtype_name({"output_dtypes": ["float32"]}) == "float32"
    assert T.traced_output_dtype_name({"output_dtypes": ["torch.float16"]}) == "float16"
    assert T.traced_output_dtype_name({"attributes": {"output_dtypes": ["float32"]}}) is None
    assert T.traced_output_dtype_name(None) is None


class _Fake:
    """The surface of an NBXTensor the AMP wrappers read — no device."""

    def __init__(self, dt):
        self.nbx_dtype = dt

    def is_floating_point(self):
        return True

    def to(self, dt):
        return _Fake(dt)

    def contiguous(self):
        return self

    def is_contiguous(self):
        return True


def _same(*args, **kwargs):
    return _Fake(args[0].nbx_dtype)


@pytest.fixture
def flag():
    from neurobrix.kernels import wrappers as _w
    prev = _w.get_activations_fp16_safe()
    _w.set_activations_fp16_safe(True)
    yield
    _w.set_activations_fp16_safe(prev)


def _out(op, traced, narrow=()):
    eng = T.TritonDtypeEngine(NBXDtype.float16, graph_dtype="float16", has_fp64=False)
    eng.set_precision_contract(True, (), narrow)
    rec = {"output_dtypes": [traced]}
    return eng.wrap_op(op, _same, op_uid="op::0", op_record=rec)(_Fake(NBXDtype.float32)).nbx_dtype


def test_the_triton_wrapper_keeps_a_traced_fp32_op_fp32(flag):
    # the timestep chain: div by a scalar, exp — and any AMP_FP32-class op traced fp32
    for op in ("div", "exp", "rsqrt", "layer_norm"):
        assert _out(op, "float32") == NBXDtype.float32, op


def test_the_triton_wrapper_still_narrows_what_the_vendor_ran_in_fp16(flag):
    for op in ("div", "exp", "rsqrt", "layer_norm"):
        assert _out(op, "float16") == NBXDtype.float16, op
    # the narrow set narrows a traced fp32 op, like the compiled engine
    assert _out("exp", "float32", narrow=("op::0",)) == NBXDtype.float16
