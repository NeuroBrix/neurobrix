"""The symbolic vocabulary carries true division and square root — one evaluator, every reader.

mochi-1-preview's rotary table (diffusers MochiRoPE, `transformer_mochi.py`, `_get_positions`)
spans its centres over `+-height * scale / 2`, `scale = (target_area / (height * width)) ** 0.5`,
height and width the post-patch extents. With integers only (`add sub mul floordiv mod neg`), Forge
could record those bounds as the trace's floats alone, and the table's values froze at the trace
grid (9.5 rad of rotary error at 160x416 from a 10x60x106 trace). Forge's extent provenance now
records them as expressions with `truediv` and `sqrt`; this file proves that every runtime reader
evaluates them through the ONE evaluator (`core/runtime/symexpr.py`) and agrees:

* the ATen shape resolver (`resolve_scalar`), the Triton symbol resolver (`resolve_scalar`), the
  compiled sequence's and the Triton sequence's expression arguments, the Triton-sequential
  argument and keyword paths, and the ATen-sequential argument path — seven readers, one value,
  equal to the vendor's eager arithmetic at the trace grid and at two others, one far;
* a real in a SHAPE slot (an extent, a size-list element) refuses by name in every reader;
* Prism's profiler declines a real dim, and the census derivation binds its tiles through the
  same function (an injection that breaks the shared evaluator breaks the derivation).

The expressions below are what Forge emitted for the vendor's construction (Forge branch
the-symbolic-vocabulary-carries-true-division-and-sqrt, `tests/test_a_real_scalar_derived_from_
symbolic_extents_is_an_expression.py`, `_MochiPositions` captured at hidden_states [2, 4, 7, 26,
38]); s2 / s3 are the latent height / width.

What each test would do if the code were wrong: a reader that kept its own integer evaluator
raises "unknown type truediv" or returns the dict to the op (seen red per reader before the
consolidation: the compiled/Triton `_compile_arg` returned the raw dict; the resolvers raised);
a shape slot that took the float would return it instead of refusing (injection: the slot check
removed -> the refusal cells red); `** 0.5` vs math.sqrt differ by one ulp at most in float64
(measured: 286 of 358 801 grids), so the eager comparison is to one ulp and the float32 table exact.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest


def _sym(i, trace):
    return {"type": "symbol", "id": f"s{i}", "trace": trace}


def _hh():
    return {"type": "floordiv", "left": _sym(2, 26), "right": 2, "trace": 13}


def _ww():
    return {"type": "floordiv", "left": _sym(3, 38), "right": 2, "trace": 19}


def _scale():
    return {"type": "sqrt", "operand": {"type": "truediv", "left": 36864,
                                        "right": {"type": "mul", "left": _hh(), "right": _ww(), "trace": 247},
                                        "trace": 149.24696356275302}, "trace": 12.216667449134933}


# Forge-emitted (verbatim structure): aten.linspace::0 args[0:2] and aten.linspace::1 args[0:2]
H_START = {"type": "truediv", "left": {"type": "mul", "left": {"type": "neg", "operand": _hh(), "trace": -13},
                                       "right": _scale(), "trace": -158.81667683875412},
           "right": 2, "trace": -79.40833841937706}
H_STOP = {"type": "truediv", "left": {"type": "mul", "left": _hh(), "right": _scale(), "trace": 158.81667683875412},
          "right": 2, "trace": 79.40833841937706}
W_START = {"type": "truediv", "left": {"type": "mul", "left": {"type": "neg", "operand": _ww(), "trace": -19},
                                       "right": _scale(), "trace": -232.11668153356374},
           "right": 2, "trace": -116.05834076678187}
W_STOP = {"type": "truediv", "left": {"type": "mul", "left": _ww(), "right": _scale(), "trace": 232.11668153356374},
          "right": 2, "trace": 116.05834076678187}
H_STEPS = {"type": "add", "left": {"type": "add", "left": {"type": "floordiv", "left": {
    "type": "add", "left": _sym(2, 26), "right": -2, "trace": 24}, "right": 2, "trace": 12},
    "right": 1, "trace": 13}, "right": 1, "trace": 14}
BOUNDS = {"h_start": H_START, "h_stop": H_STOP, "w_start": W_START, "w_stop": W_STOP}

CTX = {"symbols": {
    "s0": {"name": "batch", "trace_value": 2, "source": "input::hidden_states::dim_0", "constraints": {"min": 1}},
    "s1": {"name": "time", "trace_value": 7, "source": "input::hidden_states::dim_2", "constraints": {"min": 1}},
    "s2": {"name": "height", "trace_value": 26, "source": "input::hidden_states::dim_3", "constraints": {"min": 1}},
    "s3": {"name": "width", "trace_value": 38, "source": "input::hidden_states::dim_4", "constraints": {"min": 1}}}}

TRACE = (7, 26, 38)
GRIDS = [TRACE, (3, 18, 30), (9, 90, 250)]          # the trace, a second grid, a far one


def _env(grid):
    t, h, w = grid
    return {"s0": 2, "s1": t, "s2": h, "s3": w}


def _eager(grid):
    """The vendor's arithmetic, verbatim (MochiRoPE._get_positions on the post-patch extents)."""
    hh, ww = grid[1] // 2, grid[2] // 2
    scale = (36864 / (hh * ww)) ** 0.5
    return {"h_start": -hh * scale / 2, "h_stop": hh * scale / 2,
            "w_start": -ww * scale / 2, "w_stop": ww * scale / 2}


# ───────────────────────── the seven readers ─────────────────────────

def _aten(env):
    from neurobrix.core.runtime.shape_resolver import SymbolicShapeResolver
    r = SymbolicShapeResolver(CTX)
    for sid, v in env.items():
        r._bind_symbol(sid, v, CTX["symbols"][sid])
    return r


def _triton(env):
    from neurobrix.triton.symbols import SymbolResolver
    r = SymbolResolver(CTX)
    for sid, v in env.items():
        r._bind(sid, v)
    return r


def read_aten_resolver(node, env):
    return _aten(env).resolve_scalar(node)


def read_triton_resolver(node, env):
    return _triton(env).resolve_scalar(node)


def read_compiled_sequence(node, env, in_list=False):
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence, ExprArg, ListArg
    cs = CompiledSequence.__new__(CompiledSequence)
    cs._shape_resolver = _aten(env)
    arg = cs._compile_arg(node, {})
    assert isinstance(arg, ExprArg), f"the compiled sequence did not compile {node['type']!r} as an expression"
    if in_list:
        return cs._make_args_resolver((ListArg((arg,)),))(None)[0][0]
    return cs._make_args_resolver((arg,))(None)[0]


def read_triton_sequence(node, env, in_list=False):
    from neurobrix.triton.sequence import TritonSequence, ExprArg, ListArg
    ts = TritonSequence.__new__(TritonSequence)
    ts._symbol_resolver = _triton(env)
    arg = ts._compile_arg(node, {})
    assert isinstance(arg, ExprArg), f"the Triton sequence did not compile {node['type']!r} as an expression"
    if in_list:
        return ts._make_args_resolver((ListArg((arg,)),))(None)[0][0]
    return ts._make_args_resolver((arg,))(None)[0]


def read_triton_sequential(node, env, in_list=False):
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    ge = GraphExecutor.__new__(GraphExecutor)
    arg = {"type": "list", "value": [node]} if in_list else node
    out = ge._resolve_sequential_arg(arg, {}, _triton(env), None)
    return out[0] if in_list else out


def read_triton_sequential_keyword(node, env):
    from neurobrix.triton.sequential import TritonSequentialDispatcher
    d = TritonSequentialDispatcher.__new__(TritonSequentialDispatcher)
    d.symbol_resolver = _triton(env)
    return d.resolve_kwargs({"kwargs": {"start": node}})["start"]


def read_aten_sequential(node, env, in_list=False):
    pytest.importorskip("torch")
    from neurobrix.core.runtime.graph.tensor_resolver import TensorResolver
    ctx = SimpleNamespace(symbolic_shapes_enabled=True, shape_resolver=_aten(env))
    tr = TensorResolver(ctx)
    if in_list:
        return tr._resolve_arg_info("u", "aten::zeros", {"type": "list", "value": [node]}, None)[0]
    return tr._resolve_arg_info("u", "aten::linspace", node, None)


SCALAR_READERS = [read_aten_resolver, read_triton_resolver, read_compiled_sequence, read_triton_sequence,
                  read_triton_sequential, read_triton_sequential_keyword, read_aten_sequential]
LIST_READERS = [read_compiled_sequence, read_triton_sequence, read_triton_sequential, read_aten_sequential]


@pytest.mark.parametrize("grid", GRIDS, ids=["trace", "second", "far"])
@pytest.mark.parametrize("reader", SCALAR_READERS, ids=lambda f: f.__name__)
def test_every_reader_evaluates_the_rotary_bounds_as_the_vendor_computes_them(reader, grid):
    env, want = _env(grid), _eager(grid)
    for name, node in BOUNDS.items():
        got = reader(node, env)
        assert isinstance(got, float), (name, got)
        assert abs(got - want[name]) <= math.ulp(want[name]), (name, got, want[name])
    assert reader(H_STEPS, env) == grid[1] // 2 + 1


@pytest.mark.parametrize("grid", GRIDS, ids=["trace", "second", "far"])
def test_the_readers_agree_to_the_bit(grid):
    env = _env(grid)
    for node in BOUNDS.values():
        values = {f.__name__: f(node, env) for f in SCALAR_READERS}
        assert len(set(values.values())) == 1, values


@pytest.mark.parametrize("grid", GRIDS[1:], ids=["second", "far"])
def test_the_float32_edges_are_the_vendors(grid):
    torch = pytest.importorskip("torch")
    env, want = _env(grid), _eager(grid)
    hh = grid[1] // 2
    got = torch.linspace(read_triton_resolver(H_START, env), read_triton_resolver(H_STOP, env),
                         read_triton_resolver(H_STEPS, env), dtype=torch.float32)
    ref = torch.linspace(want["h_start"], want["h_stop"], hh + 1, dtype=torch.float32)
    assert torch.equal(got, ref)


def test_the_recorded_trace_is_what_the_tree_gives_at_the_trace():
    env = _env(TRACE)
    for node in BOUNDS.values():
        assert read_triton_resolver(node, env) == node["trace"]


# ───────────────────────── a real never becomes an extent ─────────────────────────

REAL_SIZE = {"type": "truediv", "left": _sym(2, 26), "right": 2, "trace": 13.0}
SQRT_SIZE = {"type": "sqrt", "operand": {"type": "mul", "left": _sym(2, 26), "right": _sym(2, 26)}, "trace": 26.0}


@pytest.mark.parametrize("node", [REAL_SIZE, SQRT_SIZE, H_START], ids=["truediv", "sqrt", "rotary-bound"])
def test_a_real_in_a_shape_slot_refuses_by_name_in_the_resolvers(node):
    from neurobrix.core.runtime.symexpr import RealInShapeSlot
    env = _env(GRIDS[1])
    with pytest.raises(RealInShapeSlot, match="shape slot"):
        _aten(env).resolve([4, node])                         # a tensor's extents
    with pytest.raises(RealInShapeSlot, match="shape slot"):
        _triton(env).resolve(node)


@pytest.mark.parametrize("reader", LIST_READERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("node", [REAL_SIZE, SQRT_SIZE], ids=["truediv", "sqrt"])
def test_a_real_in_a_size_list_refuses_by_name_in_every_sequence(reader, node):
    from neurobrix.core.runtime.symexpr import RealInShapeSlot
    with pytest.raises(RealInShapeSlot, match=node["type"]):
        reader(node, _env(GRIDS[1]), in_list=True)


@pytest.mark.parametrize("reader", LIST_READERS, ids=lambda f: f.__name__)
def test_control_an_integer_size_in_a_list_still_resolves(reader):
    assert reader(H_STEPS, _env(GRIDS[2]), in_list=True) == GRIDS[2][1] // 2 + 1


def test_an_exact_quotient_is_still_a_real():
    """`truediv` is Python's `/`: 26 / 2 is 13.0, a float — an extent written with `/` is a
    defect of the emitter, never rounded into an int by the reader."""
    from neurobrix.core.runtime.symexpr import RealInShapeSlot, evaluate, SCALAR, SHAPE
    assert evaluate(REAL_SIZE, lambda s, n: 26, SCALAR) == 13.0
    with pytest.raises(RealInShapeSlot):
        evaluate(REAL_SIZE, lambda s, n: 26, SHAPE)


def test_a_node_outside_the_vocabulary_refuses_by_name():
    from neurobrix.core.runtime.symexpr import UnknownExpression
    with pytest.raises(UnknownExpression, match="'pow'"):
        _triton(_env(TRACE)).resolve_scalar({"type": "pow", "left": _sym(2, 26), "right": 0.5})
    with pytest.raises(UnknownExpression, match="'pow'"):
        _aten(_env(TRACE)).resolve({"type": "pow", "left": _sym(2, 26), "right": 2})


def test_an_unbound_symbol_inside_a_real_refuses_by_name():
    from neurobrix.core.runtime.shape_resolver import ShapeResolutionError
    from neurobrix.triton.symbols import UnboundSymbolError
    env = {"s0": 2, "s1": 7, "s2": 26}                          # width never bound
    with pytest.raises(UnboundSymbolError, match="s3"):
        _triton(env).resolve_scalar(H_START)
    with pytest.raises(ShapeResolutionError, match="s3"):
        _aten(env).resolve_scalar(H_START)


# ───────────────────────── Prism and the census derivation ─────────────────────────

def test_prisms_profiler_declines_a_real_dim():
    from neurobrix.core.prism.profiler import _eval_symbolic_expr, ActivationProfiler
    assert _eval_symbolic_expr(REAL_SIZE, "s2", 26) is None
    assert _eval_symbolic_expr(H_STEPS, "s2", 90) == 46
    p = ActivationProfiler.__new__(ActivationProfiler)
    assert p._eval_dim_expr(H_STEPS, {"s2": 90}) == 46
    with pytest.raises(ValueError, match="shape slot"):
        p._eval_dim_expr(REAL_SIZE, {"s2": 90})


def _tile_graph(dim3):
    return {"symbolic_context": CTX, "input_tensor_ids": ["input::hidden_states"],
            "tensors": {"input::hidden_states": {"shape": [2, 4, 7, 26, 38], "symbolic_shape": {
                "dims": [_sym(0, 2), 4, _sym(1, 7), dim3, _sym(3, 38)], "concrete": [2, 4, 7, 26, 38]}}}}


def _derived_census():
    import importlib
    import sys
    from pathlib import Path
    tools = str(Path(__file__).resolve().parents[3] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("derived_census")


def test_the_census_derivation_binds_through_the_one_evaluator(monkeypatch):
    D = _derived_census()
    from neurobrix.core.runtime import symexpr
    env = _env(GRIDS[2])
    spec = {"tile_size": 64}
    hh_plus = {"type": "add", "left": _hh(), "right": _hh(), "trace": 26}    # an integer extent tree
    out = D.tile_bindings(_tile_graph(hh_plus), env, spec)
    assert out[0]["s3"] == 64                                               # 250 tiled to 64
    with pytest.raises(symexpr.RealInShapeSlot):
        D.tile_bindings(_tile_graph(REAL_SIZE), env, spec)
    # the derivation reads extents through symexpr.evaluate itself: break it and the derivation breaks
    def broken(*a, **k):
        raise RuntimeError("the shared evaluator was called")
    monkeypatch.setattr(symexpr, "evaluate", broken)
    with pytest.raises(RuntimeError, match="shared evaluator was called"):
        D.tile_bindings(_tile_graph(hh_plus), env, spec)


# ───────────────────────── the product, through the same evaluator ─────────────────────────

PRODUCT = {"type": "product", "factors": ["s2", {"type": "floordiv", "left": _sym(3, 38), "right": 2, "trace": 19}],
           "trace_value": 26 * 19}


@pytest.mark.parametrize("reader", LIST_READERS, ids=lambda f: f.__name__)
def test_a_product_with_an_expression_factor_is_evaluated_not_frozen(reader):
    """The Triton promotion's `product` node used to compile through each sequence's own
    ProductArg, which froze a factor that is an expression to its trace value (19 here); it is
    now one node of the shared vocabulary. Injection: the sequences' product branch restored ->
    the compiled and Triton sequence cells read 90 * 19, RED."""
    grid = GRIDS[2]
    assert reader(PRODUCT, _env(grid), in_list=True) == grid[1] * (grid[2] // 2)
