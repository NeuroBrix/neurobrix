"""The routing of a mixture of experts is an op of the library: `nbxtensor.ops.moe_route.moe_route` (hidden,
router_weight, choice_bias) -> (indices int64, weights float32), its one launch answered by `moe_route_launches`, and
every operand or attribute the op does not take refused BY NAME before anything reaches a device.

Before this lot the library had no such op: the import fails (red). What this test does if the code is wrong: a bias
narrowed below float32 accepted, more experts than one program holds accepted, a token count the release does not run
accepted, groups that do not cut the experts accepted, a top_k larger than the kept groups hold accepted (a dropped
expert would be routed to), a score function or group score the op does not compute accepted, a fill of NaN or +inf
accepted, an attribute with a silent default, a launch key that is not one key of the routing kernel - red.
No device needed: the refusals happen on the operands' shapes and dtypes and on the attributes."""
import math

import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import launch_keys as LK
from nbxtensor.ops import moe_route as R
from nbxtensor.tensor import NBXTensor

F16, BF16, F32 = NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32
ATTRS = dict(score_fn="sigmoid", top_k=8, n_group=8, topk_group=4, group_score="top2_sum", fill=-math.inf,
             renorm_eps=1e-20, scale=2.5)


def _t(shape, dt=BF16):
    return NBXTensor.empty_cpu(tuple(shape), dtype=dt)


def _ops(T=1, E=256, H=64, dt=BF16, wdt=None, bias=F32):
    return _t((T, H), dt), _t((E, H), wdt or dt), None if bias is None else _t((E,), bias)


def _route(ops, **over):
    hidden, weight, bias = ops
    return R.moe_route(hidden, weight, choice_bias=bias, **dict(ATTRS, **over))


def test_its_operands_shapes_name_the_problem():
    assert R.moe_route_shapes(*_ops()) == (1, 64, 256)
    assert R.moe_route_shapes(*_ops(E=64, bias=None)) == (1, 64, 64)


def test_the_launch_is_one_key_of_the_routing_kernel():
    keys = LK.moe_route_launches(256, 7168, 8, BF16, BF16)
    assert keys == [(LK.MOE_ROUTE, (256, 7168, 8, "bf16", "bf16", "fp32", "int64", "fp32"))]
    assert R.moe_route_launches is LK.moe_route_launches
    assert LK.MOE_ROUTE == f"{R.__name__}.{R.moe_route_kernel.fn.__name__}"
    assert R.moe_route.self_manages_dtype is True


@pytest.mark.parametrize("kw,match", [
    (dict(bias=F16), "choice_bias is float16; it must be float32"),
    (dict(bias=BF16), "choice_bias is bfloat16; it must be float32"),
    (dict(dt=NBXDtype.float64), "hidden is float64; it must be one of"),
    (dict(wdt=NBXDtype.int8), "router_weight is int8; it must be one of"),
])
def test_an_operand_at_a_dtype_the_op_does_not_take_is_refused_by_name(kw, match):
    with pytest.raises(ValueError, match=match):
        _route(_ops(**kw))


@pytest.mark.parametrize("at,shape,match", [
    (0, (64,), "hidden has shape \\(64,\\); it must have 2 dimension"),
    (1, (256, 32), "router_weight has shape \\(256, 32\\); hidden \\(1, 64\\) asks \\[E, 64\\]"),
    (2, (255,), "choice_bias has shape \\(255,\\); router_weight \\(256, 64\\) asks \\(256,\\)"),
])
def test_an_operand_whose_shape_disagrees_is_refused_by_name(at, shape, match):
    ops = list(_ops())
    ops[at] = _t(shape, ops[at].nbx_dtype)
    with pytest.raises(ValueError, match=match):
        _route(ops)


def test_more_experts_than_one_program_holds_are_refused_by_name():
    with pytest.raises(ValueError, match="512 experts: one program holds 1 to 256 experts"):
        _route(_ops(E=512))


@pytest.mark.parametrize("T", [0, 2, 17])
def test_a_token_count_this_release_does_not_run_is_refused_by_name(T):
    with pytest.raises(ValueError, match=f"{T} tokens: this release routes the token counts \\(1,\\) only"):
        _route(_ops(T=T))


@pytest.mark.parametrize("over,match", [
    (dict(score_fn="tanh"), "score_fn 'tanh' is not one of"),
    (dict(group_score="mean"), "group_score 'mean' is not one of"),
    (dict(n_group=7), "n_group 7 does not cut 256 experts into equal groups"),
    (dict(n_group=0), "n_group 0 does not cut 256 experts"),
    (dict(n_group=256, topk_group=4), "group_score 'top2_sum' over groups of 1 expert"),
    (dict(topk_group=9), "topk_group 9 is not between 1 and n_group 8"),
    (dict(n_group=1, topk_group=2), "topk_group 2 is not between 1 and n_group 1"),
    (dict(topk_group=1, top_k=33), "top_k 33 is not between 1 and the 32 experts of the 1 kept group"),
    (dict(top_k=0), "top_k 0 is not between 1"),
    (dict(fill=math.nan), "fill nan would rank"),
    (dict(fill=math.inf), "fill inf would rank"),
    (dict(renorm_eps=-1e-20), "renorm_eps -1e-20 must be None"),
    (dict(renorm_eps=math.inf), "renorm_eps inf must be None"),
    (dict(scale=math.nan), "scale nan is not a finite number"),
])
def test_an_attribute_the_op_does_not_take_is_refused_by_name(over, match):
    with pytest.raises(ValueError, match=match):
        _route(_ops(), **over)


def test_the_attributes_have_no_default():
    hidden, weight, bias = _ops()
    for name in ("choice_bias",) + tuple(ATTRS):
        kw = dict(ATTRS, choice_bias=bias)
        del kw[name]
        with pytest.raises(TypeError, match=name):
            R.moe_route(hidden, weight, **kw)
