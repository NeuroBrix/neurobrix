"""Under bf16 compute on a bf16 GRAPH, an AMP_FP32 op computes fp32 inside and returns bf16.

The rule (supervisor decision, 2026-09-28; refined the same day on the Mac's Apple gate):
an op of the AMP_FP32 class (norms, softmax, pow/exp/rsqrt, sum ...) computes in fp32 and,
under C = bf16, casts its OUTPUT back to bf16 when the component's GRAPH dtype — the
container's traced `torch_dtype` — is bf16, in every mode: an fp32 result always fits bf16's
range, and the vendor's graph returns x.dtype, bf16 there (Sana's DC-AE RMSNorm). On a graph
of another dtype that the profile's preferred dtype coerced to bf16 compute, x.dtype was fp32
in the vendor's forward and the output stays fp32, as before the cast-back (Kokoro's fp32
graph moved FARTHER from its fp32 reference with it: log-spectrogram corr 0.960->0.932
native, 0.948->0.891 triton). A graph that states no dtype is refused by name under bf16.
Under C = fp16 the calibration contract is unchanged: cast back only under the contract
(Triton: the `activations_fp16_safe` flag or the narrow set; compiled: the narrow set, or a
half-IO kernel), fp32 otherwise. A contract island keeps fp32 whatever C.

The decision is one pure function per engine — `amp_fp32_output_dtype(c, graph, safe,
narrowed)` in triton/dtype.py (called by the Triton wrapper and by Prism's width pass) and
its torch-free twin in core/dtype/engine.py (called by `compile_op` and by the PyTorch-
sequential `amp_cast_result`); `test_the_two_engines_twins_are_one_rule` holds them equal.
The graph dtype each engine reads is the DAG's `torch_dtype`:
`test_every_engine_reads_the_graph_dtype_from_the_container_s_torch_dtype`.

What each test would do if the code were wrong — each wrong rule below was applied ALONE
and the named tests SEEN RED (2026-09-28, CUDA_VISIBLE_DEVICES= on the rack; harness and
output: nbx/campaigns/2026_09_28_bf16_castback/inject2.py, injections2.txt; the first
round, before the graph refinement: inject.py, injections.txt):
  * the 4398f9d2 source (the cast-back without the graph dtype), these tests: 23 red; the
    semantic ones are test_compiled_keeps_fp32_for_an_fp32_graph_under_bf16 (all six rows:
    bf16 returned where the fp32 graph kept fp32) and the Prism
    test_an_fp32_graph_coerced_to_bf16_keeps_its_fp32_norm_output; the Triton rows are red
    there on the missing `graph_dtype` argument, so the semantic Triton red is the next one.
  * the rule ignoring the graph (bf16 always — 4398f9d2's rule) in triton/dtype.py: red —
    test_the_triton_rule_under_bf16_follows_the_graph_*, test_the_triton_wrapper_keeps_
    fp32_for_an_fp32_graph_under_bf16, test_the_two_engines_twins_are_one_rule (+ Prism);
    in core/dtype/engine.py: red — every test_compiled_keeps_fp32_for_an_fp32_graph row
    (compiled AND sequential) and the twins test.
  * a wrapper / compile_op / amp_cast_result reading the COMPUTE dtype as the graph's: red —
    the fp32-graph row of that engine and mode (+ the refusal-at-wrap/compile tests).
  * TritonSequence or TritonSequentialDispatcher not handing the DAG's `torch_dtype` to its
    dtype engine: red — test_every_engine_reads_the_graph_dtype_*.
  * the unchanged tree 427c8c91 (no cast-back at all): every bf16-graph row red.
  * the bf16 clause removed (bf16 answered like fp16): red — test_the_triton_rule_*,
    test_the_triton_wrapper_casts_back_under_bf16, ..._native_norm_tuple_under_bf16, the
    twins test; in the compiled twin every test_compiled_returns_bf16_under_bf16 /
    test_compiled_casts_a_native_norm_tuple_under_bf16 row.
  * the Triton wrapper's old `force or flag` gate restored: red — ..._casts_back_under_bf16,
    ..._native_norm_tuple_under_bf16, ..._keeps_fp32_for_an_fp32_graph_under_bf16.
  * the bf16 tuple cast dropped: red — test_the_triton_wrapper_casts_a_native_norm_tuple_under_bf16.
  * the fp16 cast-back extended to tuples: red — test_the_fp16_contract_is_unchanged_in_the_triton_wrapper.
  * `compile_op` (or `amp_cast_result`) passing the contract flag as `safe`: red —
    test_compiled_fp16_keeps_its_contract_unchanged[compiled] (resp. [sequential]).
  * the island branch removed: red — test_an_island_stays_fp32_under_bf16_on_the_triton_engine;
    test_an_island_stays_fp32_under_bf16_on_the_compiled_engine[both].

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

_GRAPHS = ("bfloat16", "float16", "float32", "float64")


def test_the_triton_rule_under_bf16_follows_the_graph_whatever_the_contract():
    for safe, narrowed in itertools.product((False, True), repeat=2):
        assert T.amp_fp32_output_dtype("bfloat16", "bfloat16", safe, narrowed) == "bfloat16"
        for g in ("float32", "float16", "float64"):     # a graph the profile coerced to bf16
            assert T.amp_fp32_output_dtype("bfloat16", g, safe, narrowed) == "float32", g


def test_the_triton_rule_refuses_an_unstated_graph_dtype_under_bf16():
    for g in (None, "", "int64", "torch.bfloat16"):
        with pytest.raises(ValueError, match="graph dtype"):
            T.amp_fp32_output_dtype("bfloat16", g, False, False)


def test_the_triton_rule_under_fp16_is_the_contract_s_whatever_the_graph():
    for g in _GRAPHS + (None,):
        assert T.amp_fp32_output_dtype("float16", g, False, False) == "float32"
        assert T.amp_fp32_output_dtype("float16", g, True, False) == "float16"
        assert T.amp_fp32_output_dtype("float16", g, False, True) == "float16"
        assert T.amp_fp32_output_dtype("float16", g, True, True) == "float16"


def test_the_triton_rule_refuses_a_compute_dtype_that_is_not_half():
    for c in ("float32", "float64", "NBXDtype.bfloat16", "1"):
        with pytest.raises(ValueError, match="not a half"):
            T.amp_fp32_output_dtype(c, "bfloat16", False, False)


def test_the_graph_dtype_is_read_by_name_from_the_container():
    assert T.graph_dtype_name("torch.bfloat16") == "bfloat16"
    assert T.graph_dtype_name("float32") == "float32"
    assert T.graph_dtype_name(NBXDtype.bfloat16) == "bfloat16"
    assert T.graph_dtype_name("") is None and T.graph_dtype_name(None) is None


def test_the_two_engines_twins_are_one_rule():
    """The compiled engine's module imports torch and the Triton one imports triton: neither
    may import the other, so the rule has a twin. This door holds them equal everywhere."""
    pytest.importorskip("torch")
    from neurobrix.core.dtype import engine as E
    for c, g, safe, narrowed, traced in itertools.product(
            _HALF, _GRAPHS, (False, True), (False, True), (None,) + _GRAPHS):
        assert (E.amp_fp32_output_dtype(c, g, safe, narrowed, traced)
                == T.amp_fp32_output_dtype(c, g, safe, narrowed, traced)), (c, g, safe, narrowed, traced)
    for f in (E.amp_fp32_output_dtype, T.amp_fp32_output_dtype):
        assert f("float16", None, True, False) == "float16"
        for c, g in (("float32", "float32"), ("float64", "bfloat16"), ("bfloat16", None),
                     ("bfloat16", "")):
            with pytest.raises(ValueError):
                f(c, g, False, False)


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


def _out(c, op, func, x_dtype=None, graph="same", **contract):
    """`graph`: the component's graph dtype — by default the compute dtype's (a bf16
    graph run in bf16, an fp16 one in fp16)."""
    eng = T.TritonDtypeEngine(c, graph_dtype=c.name if graph == "same" else graph, has_fp64=False)
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


def test_the_triton_wrapper_keeps_fp32_for_an_fp32_graph_under_bf16(flag):
    """An fp32 graph the profile coerced to bf16 compute: x.dtype was fp32 in the vendor's
    forward — the output keeps fp32, as before the bf16 cast-back (Kokoro, Apple gate)."""
    for safe in (False, True):
        flag(safe)
        for op in ("rms_norm", "layer_norm", "_softmax"):
            assert _out(NBXDtype.bfloat16, op, _norm, graph="float32").nbx_dtype == NBXDtype.float32
        assert _out(NBXDtype.bfloat16, "layer_norm", _widening,
                    graph="torch.float32").nbx_dtype == NBXDtype.float32
        out = _out(NBXDtype.bfloat16, "native_layer_norm", _native_norm, graph="float32")
        assert [o.nbx_dtype for o in out] == [NBXDtype.float32] * 3


def test_the_triton_wrapper_refuses_an_unstated_graph_under_bf16_at_wrap_time():
    eng = T.TritonDtypeEngine(NBXDtype.bfloat16, has_fp64=False)
    with pytest.raises(ValueError, match="graph dtype"):
        eng.wrap_op("rms_norm", _norm, op_uid="op::0")
    # an op the rule does not concern still wraps (the Triton engine is built before any op)
    eng.wrap_op("mm", _norm, op_uid="op::1")


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


def _engine(c, safe=False, narrow=(), islands=(), graph="same"):
    """`graph`: the component's graph dtype (DtypeEngine's `graph_dtype`, the container's
    torch_dtype parsed) — by default the compute dtype's."""
    from neurobrix.core.dtype.engine import DtypeEngine
    return DtypeEngine(c, graph_dtype=c if graph == "same" else graph,
                       activations_fp16_safe=safe, narrow_op_uids=frozenset(narrow),
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


@pytest.mark.parametrize("run", [_compiled, _sequential], ids=["compiled", "sequential"])
@pytest.mark.parametrize("op_type,make", _OPS, ids=[o for o, _ in _OPS])
def test_compiled_keeps_fp32_for_an_fp32_graph_under_bf16(run, op_type, make):
    """An fp32 graph the profile coerced to bf16 compute keeps the fp32 output it had."""
    func, *args = make(torch.bfloat16)
    assert run(_engine(torch.bfloat16, graph=torch.float32), op_type, func,
               *args).dtype == torch.float32


def test_compiled_refuses_an_unstated_graph_under_bf16_at_compile_time():
    with pytest.raises(ValueError, match="graph dtype"):
        _engine(torch.bfloat16, graph=None).compile_op("aten::exp", torch.ops.aten.exp, {},
                                                       op_uid="op::0")


def test_every_engine_reads_the_graph_dtype_from_the_container_s_torch_dtype():
    """The call sites: each engine's dtype engine holds the DAG's own `torch_dtype` — never a
    model name, never the compute dtype. A segment or sub-graph carries the key (layer_partition
    keeps the top-level keys; the const-fold sub-DAGs copy it)."""
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence
    from neurobrix.triton.sequence import TritonSequence
    from neurobrix.triton.sequential import TritonSequentialDispatcher
    for g in ("float32", "bfloat16"):
        dag = {"torch_dtype": g, "ops": {}, "tensors": {}, "execution_order": []}
        assert TritonSequence(dag, 0, NBXDtype.bfloat16, has_fp64=False)._dtype_engine.graph_dtype == g
        assert TritonSequentialDispatcher(0, NBXDtype.bfloat16,
                                          graph_dtype=g, has_fp64=False)._dtype_engine.graph_dtype == g
        cs = CompiledSequence(dag, torch.device("cpu"), torch.bfloat16)
        assert cs.op_resolver.dtype_engine.graph_dtype == getattr(torch, g)
