"""Under bf16 compute, an AMP_FP32 op computes fp32 inside and returns the compute dtype.

The rule (supervisor decision, 2026-09-28): an op of the AMP_FP32 class (norms, softmax,
pow/exp/rsqrt, sum ...) computes in fp32 and casts its OUTPUT back to the compute dtype C
under C = bf16, in BOTH engines, by one rule — an fp32 result always fits bf16's range
(bf16 has fp32's exponent) and the vendors' graphs return x.dtype (Sana's DC-AE RMSNorm).
Under C = fp16 the calibration contract is unchanged: the output is cast back only under
the contract (Triton: the `activations_fp16_safe` flag or the narrow set; compiled: the
narrow set, or a half-IO kernel), fp32 otherwise. A contract island keeps fp32 whatever C.

The decision is one pure function per engine — `amp_fp32_output_dtype(c, safe, narrowed)`
in triton/dtype.py (called by the Triton wrapper and by Prism's width pass) and its
torch-free twin in core/dtype/engine.py (called by `compile_op` and by the PyTorch-
sequential `amp_cast_result`); `test_the_two_engines_twins_are_one_rule` holds them equal.

What each test would do if the code were wrong — each wrong rule below was applied ALONE
and the named tests SEEN RED (2026-09-28, CUDA_VISIBLE_DEVICES= on the rack; the harness
and its output: nbx/campaigns/2026_09_28_bf16_castback/inject.py, injections.txt):
  * the unchanged tree (427c8c91 source, these tests): 15 red — every bf16 row (the decision
    function absent, the bf16 output fp32); green there, as they must be, the rows that pin
    what is UNCHANGED: the fp16 contract (both engines), `div` under bf16, the islands.
  * the bf16 clause removed from triton/dtype.py `amp_fp32_output_dtype` (bf16 answered like
    fp16): red — test_the_triton_rule_under_bf16_*, test_the_two_engines_twins_are_one_rule,
    test_the_triton_wrapper_casts_back_under_bf16, test_the_triton_wrapper_casts_a_native_
    norm_tuple_under_bf16 (+ the Prism width and drift tests).
  * the Triton wrapper not calling the rule (the old `force or flag` gate restored): red —
    test_the_triton_wrapper_casts_back_under_bf16, ..._native_norm_tuple_under_bf16.
  * the bf16 tuple cast dropped: red — test_the_triton_wrapper_casts_a_native_norm_tuple_under_bf16.
  * the fp16 cast-back extended to tuples (the fp16 contract no longer byte-identical): red —
    test_the_fp16_contract_is_unchanged_in_the_triton_wrapper.
  * the bf16 clause removed from core/dtype/engine.py's twin: red —
    test_the_two_engines_twins_are_one_rule and every test_compiled_returns_bf16_under_bf16 /
    test_compiled_casts_a_native_norm_tuple_under_bf16 row, compiled AND sequential.
  * `compile_op` (or `amp_cast_result`) passing the contract flag as `safe` — the flag alone
    narrowing on the compiled engine, a change of its fp16 behaviour: red —
    test_compiled_fp16_keeps_its_contract_unchanged[compiled] (resp. [sequential]).
  * the island branch removed (the op falls to the rule): red —
    test_an_island_stays_fp32_under_bf16_on_the_triton_engine (Triton);
    test_an_island_stays_fp32_under_bf16_on_the_compiled_engine[both] (compiled + sequential).

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest -q \
     tests/unit/runtime/test_under_bf16_an_fp32_internal_op_returns_the_compute_dtype.py
"""
from __future__ import annotations

import itertools

import pytest

from neurobrix.kernels.nbx_tensor import NBXDtype
from neurobrix.triton import dtype as T

_HALF = ("bfloat16", "float16")


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

def test_the_triton_rule_under_bf16_is_the_compute_dtype_whatever_the_contract():
    for safe, narrowed in itertools.product((False, True), repeat=2):
        assert T.amp_fp32_output_dtype("bfloat16", safe, narrowed) == "bfloat16"


def test_the_triton_rule_under_fp16_is_the_contract_s():
    assert T.amp_fp32_output_dtype("float16", False, False) == "float32"
    assert T.amp_fp32_output_dtype("float16", True, False) == "float16"
    assert T.amp_fp32_output_dtype("float16", False, True) == "float16"
    assert T.amp_fp32_output_dtype("float16", True, True) == "float16"


def test_the_triton_rule_refuses_a_compute_dtype_that_is_not_half():
    for c in ("float32", "float64", "NBXDtype.bfloat16", "1"):
        with pytest.raises(ValueError, match="not a half"):
            T.amp_fp32_output_dtype(c, False, False)


def test_the_two_engines_twins_are_one_rule():
    """The compiled engine's module imports torch and the Triton one imports triton: neither
    may import the other, so the rule has a twin. This door holds them equal everywhere."""
    pytest.importorskip("torch")
    from neurobrix.core.dtype import engine as E
    for c, safe, narrowed in itertools.product(_HALF, (False, True), (False, True)):
        assert E.amp_fp32_output_dtype(c, safe, narrowed) == T.amp_fp32_output_dtype(c, safe, narrowed)
    for c in ("float32", "float64"):
        for f in (E.amp_fp32_output_dtype, T.amp_fp32_output_dtype):
            with pytest.raises(ValueError):
                f(c, False, False)


# ---------------------------------------------------------------------------
# The Triton wrapper — card-free: `_is_float_tensor` / `_get_nbx_dtype` duck-type
# ---------------------------------------------------------------------------

class _Fake:
    """The surface of an NBXTensor the AMP wrappers read — no device."""

    def __init__(self, dt: NBXDtype):
        self.nbx_dtype = dt

    def is_floating_point(self):
        return True

    def to(self, dt):
        return _Fake(dt)

    def contiguous(self):
        return self

    def is_contiguous(self):
        return True


def _norm(*args, **kwargs):
    """An fp32-internal op: writes its output at its (upcast) input's dtype."""
    return _Fake(args[0].nbx_dtype)


def _native_norm(*args, **kwargs):
    """`native_layer_norm`: (out, mean, rstd), the statistics fp32 (wrappers.py)."""
    return (_Fake(args[0].nbx_dtype), _Fake(NBXDtype.float32), _Fake(NBXDtype.float32))


def _widening(*args, out_dtype=None, **kwargs):
    """A kernel that widens its loads: stores what it is asked (`_nbx_widens_on_load`)."""
    return _Fake(out_dtype)


_widening._nbx_widens_on_load = True


@pytest.fixture
def flag():
    """Set `_NBX_ACTIVATIONS_FP16_SAFE` for one test and restore it."""
    from neurobrix.kernels import wrappers as _w
    prev = _w.get_activations_fp16_safe()
    yield _w.set_activations_fp16_safe
    _w.set_activations_fp16_safe(prev)


def _out(c, op, func, x_dtype=None, **contract):
    eng = T.TritonDtypeEngine(c)
    if contract:
        eng.set_precision_contract(**contract)
    return eng.wrap_op(op, func, op_uid="op::0")(_Fake(x_dtype or c))


def test_the_triton_wrapper_casts_back_under_bf16(flag):
    flag(False)
    for op in ("rms_norm", "layer_norm", "group_norm", "_softmax", "sum", "rsqrt"):
        assert _out(NBXDtype.bfloat16, op, _norm).nbx_dtype == NBXDtype.bfloat16, op
    # an fp32 input (a vendor fp32 island upstream) is still returned at C
    assert _out(NBXDtype.bfloat16, "rms_norm", _norm, NBXDtype.float32).nbx_dtype == NBXDtype.bfloat16
    # the kernel that widens its loads stores fp32, and the rule casts it back
    assert _out(NBXDtype.bfloat16, "layer_norm", _widening).nbx_dtype == NBXDtype.bfloat16
    # the flag says nothing under bf16
    flag(True)
    assert _out(NBXDtype.bfloat16, "rms_norm", _norm).nbx_dtype == NBXDtype.bfloat16


def test_the_triton_wrapper_casts_a_native_norm_tuple_under_bf16(flag):
    flag(False)
    out = _out(NBXDtype.bfloat16, "native_layer_norm", _native_norm)
    assert isinstance(out, tuple) and [o.nbx_dtype for o in out] == [NBXDtype.bfloat16] * 3


def test_the_fp16_contract_is_unchanged_in_the_triton_wrapper(flag):
    flag(False)                                  # no contract: fp32 kept
    assert _out(NBXDtype.float16, "rms_norm", _norm).nbx_dtype == NBXDtype.float32
    assert _out(NBXDtype.float16, "layer_norm", _widening).nbx_dtype == NBXDtype.float32
    assert _out(NBXDtype.float16, "div", _norm).nbx_dtype == NBXDtype.float32
    flag(True)                                   # the contract's safe flag: fp16
    assert _out(NBXDtype.float16, "rms_norm", _norm).nbx_dtype == NBXDtype.float16
    assert _out(NBXDtype.float16, "layer_norm", _widening).nbx_dtype == NBXDtype.float16
    assert _out(NBXDtype.float16, "div", _norm).nbx_dtype == NBXDtype.float16
    # ... and its single-tensor cast, as it always was: a tuple keeps its kernel's dtypes
    out = _out(NBXDtype.float16, "native_layer_norm", _native_norm)
    assert [o.nbx_dtype for o in out] == [NBXDtype.float32] * 3
    flag(False)                                  # the narrow set alone: fp16
    assert _out(NBXDtype.float16, "rms_norm", _norm,
                safe=False, narrow_op_uids={"op::0"}).nbx_dtype == NBXDtype.float16


def test_div_under_bf16_is_a_plain_amp_fp16_op(flag):
    """`div` reaches the fp32 wrapper under fp16 only; under bf16 its inputs are cast to
    bf16 and it writes bf16 — nothing of the new rule concerns it."""
    flag(False)
    seen = []

    def div(a, *rest):
        seen.append(a.nbx_dtype)
        return _Fake(a.nbx_dtype)

    assert _out(NBXDtype.bfloat16, "div", div, NBXDtype.float32).nbx_dtype == NBXDtype.bfloat16
    assert seen == [NBXDtype.bfloat16]


def test_an_island_stays_fp32_under_bf16_on_the_triton_engine(flag):
    """Islands are decided before the rule (`_wrap_fp32`): unchanged. (Under bf16 the
    contract resolves to no island; an engine handed one still honours it.)"""
    flag(False)
    out = _out(NBXDtype.bfloat16, "rms_norm", _norm, safe=False, fp32_op_uids={"op::0"})
    assert out.nbx_dtype == NBXDtype.float32


# ---------------------------------------------------------------------------
# The compiled engine (torch on CPU) — compile_op and the sequential mirror
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def _engine(c, safe=False, narrow=(), islands=()):
    from neurobrix.core.dtype.engine import DtypeEngine
    return DtypeEngine(c, activations_fp16_safe=safe, narrow_op_uids=frozenset(narrow),
                       fp32_op_uids=frozenset(islands))


def _x(c):
    return torch.randn(2, 8, dtype=c)


def _compiled(eng, op_type, func, *args):
    return eng.compile_op(op_type, func, {}, op_uid="op::0")(*args)


def _sequential(eng, op_type, func, *args):
    """GraphExecutor's --sequential path: amp_cast_inputs, the native op, amp_cast_result."""
    a = eng.amp_cast_inputs(op_type, list(args), op_uid="op::0")
    return eng.amp_cast_result(op_type, func(*a), op_uid="op::0")


_OPS = [
    ("aten::exp", lambda c: (torch.ops.aten.exp, _x(c))),
    ("aten::layer_norm", lambda c: (torch.ops.aten.layer_norm, _x(c), [8])),
    ("aten::_softmax", lambda c: (torch.ops.aten._softmax, _x(c), -1, False)),
]


@pytest.mark.parametrize("run", [_compiled, _sequential], ids=["compiled", "sequential"])
@pytest.mark.parametrize("op_type,make", _OPS, ids=[o for o, _ in _OPS])
def test_compiled_returns_bf16_under_bf16(run, op_type, make):
    func, *args = make(torch.bfloat16)
    assert run(_engine(torch.bfloat16), op_type, func, *args).dtype == torch.bfloat16


@pytest.mark.parametrize("run", [_compiled, _sequential], ids=["compiled", "sequential"])
def test_compiled_casts_a_native_norm_tuple_under_bf16(run):
    out = run(_engine(torch.bfloat16), "aten::native_layer_norm",
              torch.ops.aten.native_layer_norm, _x(torch.bfloat16), [8], None, None, 1e-5)
    assert [o.dtype for o in out] == [torch.bfloat16] * 3


@pytest.mark.parametrize("run", [_compiled, _sequential], ids=["compiled", "sequential"])
def test_compiled_fp16_keeps_its_contract_unchanged(run):
    exp = torch.ops.aten.exp
    fp16 = torch.float16
    assert run(_engine(fp16), "aten::exp", exp, _x(fp16)).dtype == torch.float32
    # the flag alone never narrows an fp32-class output on this engine
    assert run(_engine(fp16, safe=True), "aten::exp", exp, _x(fp16)).dtype == torch.float32
    # the contract's narrow set does
    assert run(_engine(fp16, safe=True, narrow={"op::0"}), "aten::exp", exp,
               _x(fp16)).dtype == torch.float16
    # the narrow set without the contract does not
    assert run(_engine(fp16, narrow={"op::0"}), "aten::exp", exp, _x(fp16)).dtype == torch.float32


@pytest.mark.parametrize("run", [_compiled, _sequential], ids=["compiled", "sequential"])
def test_an_island_stays_fp32_under_bf16_on_the_compiled_engine(run):
    out = run(_engine(torch.bfloat16, islands={"op::0"}), "aten::exp", torch.ops.aten.exp,
              _x(torch.bfloat16))
    assert out.dtype == torch.float32
