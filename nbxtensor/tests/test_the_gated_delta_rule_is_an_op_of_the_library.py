"""The gated delta rule is an op of the library: `nbxtensor.ops.gated_delta_rule.gated_delta_rule_wrapper` (q, k, v, g,
beta[, state]) -> (o, state), registered under the graph op `gated_delta_rule`, its one launch answered by
`gated_delta_rule_launches`, and every operand the op does not take refused BY NAME before anything reaches a device.

Before this lot the library had no such op: the import fails (red). What this test does if the code is wrong: a
compute dtype other than float32 accepted, a state or a decay narrowed below float32 accepted, value heads that are not
a whole multiple of the key heads accepted, a length the release does not run accepted, a shape that disagrees with q
and v accepted, the op missing from the dispatch table, a launch key that is not one key of the scan kernel - red.
No device needed: the refusals happen on the operands' shapes and dtypes."""
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import gated_delta_rule as G
from nbxtensor.ops import launch_keys as LK
from nbxtensor.tensor import NBXTensor

F16, BF16, F32 = NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32
ATTRS = dict(qk_l2norm=True, eps=1e-6, scale=0.125)


def _t(shape, dt=F16):
    return NBXTensor.empty_cpu(tuple(shape), dtype=dt)


def _ops(B=2, L=1, Hk=2, Hv=6, dk=16, dv=8, dt=F16, g=F32, state=F32):
    return [_t((B, L, Hk, dk), dt), _t((B, L, Hk, dk), dt), _t((B, L, Hv, dv), dt), _t((B, L, Hv), g),
            _t((B, L, Hv), dt), None if state is None else _t((B, Hv, dk, dv), state)]


def test_the_op_is_in_the_dispatch_table_as_its_wrapper():
    from nbxtensor.ops import dispatch
    assert dispatch._build_op_map()["gated_delta_rule"] is G.gated_delta_rule_wrapper
    assert G.gated_delta_rule_wrapper.self_manages_dtype is True


def test_its_operands_shapes_name_the_problem():
    assert G.gated_delta_rule_shapes(*_ops()) == (2, 1, 2, 6, 16, 8)
    assert G.gated_delta_rule_shapes(*_ops(Hk=6, state=None)) == (2, 1, 6, 6, 16, 8)


def test_the_launch_is_one_key_of_the_scan_kernel_at_any_length():
    keys = LK.gated_delta_rule_launches(128, 128, F16, F16, F16, F32, F16)
    assert keys == [(LK.GATED_DELTA_RULE, (128, 128, "fp16", "fp16", "fp16", "fp32", "fp16", "fp32", "fp16", "fp32"))]
    assert G.gated_delta_rule_launches is LK.gated_delta_rule_launches
    assert LK.GATED_DELTA_RULE == f"{G.__name__}.{G.gated_delta_rule_scan_kernel.fn.__name__}"


@pytest.mark.parametrize("compute", [F16, BF16, "float16", NBXDtype.float64])
def test_a_compute_dtype_other_than_float32_is_refused_by_name(compute):
    with pytest.raises(ValueError, match="compute_dtype .* is refused"):
        G.gated_delta_rule_wrapper(*_ops(), compute_dtype=compute, **ATTRS)


def test_a_compute_dtype_that_is_no_dtype_is_refused_by_name():
    with pytest.raises(ValueError, match="compute_dtype 'fp-thirty-two' is not a dtype"):
        G.gated_delta_rule_wrapper(*_ops(), compute_dtype="fp-thirty-two", **ATTRS)


@pytest.mark.parametrize("which,kw,match", [
    ("g", dict(g=F16), "g is float16; it must be float32"),
    ("state", dict(state=F16), "state is float16; it must be float32"),
    ("state", dict(state=BF16), "state is bfloat16; it must be float32"),
    ("inputs", dict(dt=NBXDtype.float64), "q is float64; it must be one of"),
])
def test_an_operand_at_a_dtype_the_op_does_not_take_is_refused_by_name(which, kw, match):
    with pytest.raises(ValueError, match=match):
        G.gated_delta_rule_wrapper(*_ops(**kw), compute_dtype=F32, **ATTRS)


def test_value_heads_not_a_multiple_of_the_key_heads_are_refused_by_name():
    with pytest.raises(ValueError, match="5 value heads are not a whole multiple of 2 key heads"):
        G.gated_delta_rule_wrapper(*_ops(Hk=2, Hv=5), compute_dtype=F32, **ATTRS)


@pytest.mark.parametrize("L", [0, 2, 23])
def test_a_length_this_release_does_not_run_is_refused_by_name(L):
    with pytest.raises(ValueError, match=f"a sequence of length {L}: this release runs the delta rule at the lengths"):
        G.gated_delta_rule_wrapper(*_ops(L=L), compute_dtype=F32, **ATTRS)


@pytest.mark.parametrize("at,shape,match", [
    (0, (2, 1, 16), "q has shape .*; it must have 4 dimensions"),
    (1, (2, 1, 3, 16), "k has shape \\(2, 1, 3, 16\\); q .* and v .* ask \\(2, 1, 2, 16\\)"),
    (2, (2, 1, 6), "v has shape .*; it must have 4 dimensions"),
    (3, (2, 1, 5), "g has shape \\(2, 1, 5\\); .* ask \\(2, 1, 6\\)"),
    (4, (2, 2, 6), "beta has shape \\(2, 2, 6\\); .* ask \\(2, 1, 6\\)"),
    (5, (2, 6, 8, 16), "state has shape \\(2, 6, 8, 16\\); .* ask \\(2, 6, 16, 8\\)"),
])
def test_an_operand_whose_shape_disagrees_is_refused_by_name(at, shape, match):
    ops = _ops()
    ops[at] = _t(shape, ops[at].nbx_dtype)
    with pytest.raises(ValueError, match=match):
        G.gated_delta_rule_wrapper(*ops, compute_dtype=F32, **ATTRS)


def test_the_attributes_have_no_default():
    for name in ("qk_l2norm", "eps", "scale", "compute_dtype"):
        kw = dict(ATTRS, compute_dtype=F32)
        del kw[name]
        with pytest.raises(TypeError, match=name):
            G.gated_delta_rule_wrapper(*_ops(), **kw)
