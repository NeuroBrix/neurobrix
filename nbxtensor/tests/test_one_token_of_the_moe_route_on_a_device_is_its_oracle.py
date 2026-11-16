"""`moe_route` on a device at one token: its expert sets and weights against the float64 oracle (`moe_route_oracle`),
32 tokens per case, one launch per token, at the hidden row's float32, float16 and bfloat16 (bfloat16 where the device
has it natively; the router weight at the same dtype, the choice bias float32).

The cases: E 64 / n_group 1 / k 4 with a sigmoid and a bias; E 256 / n_group 8 / topk_group 4 / k 8 with a sigmoid and
a bias, at a fill of -inf AND of 0; a softmax with the group maximum, no bias, no renormalisation; and a draw whose
negative bias makes a fill of 0 outrank kept experts (the test checks the two fills route differently there). H 1000
is not a multiple of any hidden tile. n_group 1 gives the same bytes as every group kept.

The comparison is `_moe_route_compare.compare` (its docstring is the rule; a host test proves it goes red): an exact
tie (the fill lanes of a dropped group) requires the reference set, a gap above 0 and within the band accepts a
different set, and the weights are compared on the device's own set either way. Two cases are built so that tokens
route to filled experts at fill 0 (a negative bias mean); the test asserts from the reference that they do.

A card test: on a host with no device it skips by name. It runs on a V100 card job and on the Mac (Metal), the same file.

THE BOUNDS (stated as they are: the weights deviation is relative to the token's largest weight, not per weight; the
device bounds are four times a HOST emulation, not a device measure, and no bound changes without a device measurement),
stated from what was measured (2026-11-16, this lot's proof folder, `measure_tolerances.out`): a float32
emulation of the kernel's order of operations against the oracle, worst over 8 draws of 32 tokens per case: weights
1.01e-06 relative to the token's largest weight, choice values 2.53e-07 absolute. The device's bounds are four times
that (weights 4e-06, choice values 1e-06: the card's fused multiply-adds, its own exp and its tile order round
differently from numpy). The near-tie band is two choice bounds (both sides of the gap move), four for a group score
of two summed choice values.

What this test does if the code is wrong: a hidden tile's edge dropped, the bias added to the weight, a dropped group's
expert chosen under -inf, the fill ignored, the group score wrong, the renormalisation or the scale missing, an index
outside the experts or chosen twice - red."""
import math

import numpy as np
import pytest

from _moe_route_compare import compare
from _moe_route_draw import draw
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops.oracles import fp64 as O

WEIGHTS, CHOICE = 4e-6, 1e-6
TOKENS = 32
SIGMOID = dict(score_fn="sigmoid", group_score="top2_sum", renorm_eps=1e-20)
CASES = {
    "E64-g1-k4": (64, 2048, 0.0, dict(SIGMOID, top_k=4, n_group=1, topk_group=1, fill=-math.inf, scale=1.8)),
    "E256-g8-tg4-k8-inf": (256, 7168, 0.0, dict(SIGMOID, top_k=8, n_group=8, topk_group=4, fill=-math.inf, scale=2.5)),
    "E256-g8-tg4-k8-zero": (256, 7168, 0.0, dict(SIGMOID, top_k=8, n_group=8, topk_group=4, fill=0.0, scale=2.5)),
    "E160-softmax-max": (160, 5120, None, dict(score_fn="softmax", group_score="max", renorm_eps=None, top_k=6,
                                               n_group=8, topk_group=3, fill=0.0, scale=16.0)),
    "E64-g8-tg2-k8-zero-H1000": (64, 1000, -0.6, dict(SIGMOID, top_k=8, n_group=8, topk_group=2, fill=0.0, scale=1.0)),
    # mean -1.0: most kept choice values fall below 0, so many tokens route to filled experts, some in part
    "E64-g8-tg2-k8-zero-H1000-mostly-filled": (64, 1000, -1.0, dict(SIGMOID, top_k=8, n_group=8, topk_group=2,
                                                                    fill=0.0, scale=1.0)),
    # mean -1.5: every kept choice value is below 0, every token routes to filled experts only (an exact tie on 0)
    "E64-g8-tg2-k8-zero-H1000-all-filled": (64, 1000, -1.5, dict(SIGMOID, top_k=8, n_group=8, topk_group=2,
                                                                 fill=0.0, scale=1.0)),
    "E64-g8-tg2-k8-inf-H1000": (64, 1000, -0.6, dict(SIGMOID, top_k=8, n_group=8, topk_group=2, fill=-math.inf,
                                                     scale=1.0)),
}
DTYPES = [NBXDtype.float32, NBXDtype.float16, NBXDtype.bfloat16]


@pytest.fixture(autouse=True)
def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def _to_device(a, dt):
    from nbxtensor.tensor import NBXTensor
    return NBXTensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device="cuda:0").to(dt)


def _host(t):
    return O.host_values(t.to(NBXDtype.float32)).astype(np.float64)


def _skip_bf16(dt):
    from nbxtensor.ops.wrappers import has_native_bf16
    if dt == NBXDtype.bfloat16 and not has_native_bf16():
        pytest.skip("bfloat16: this device has no native bfloat16")


def _route_on_device(x, w, b, dt, a):
    """Each token row routed by its own launch: (indices [T, k] int64, weights [T, k] float64)."""
    from nbxtensor.ops.moe_route import moe_route
    dw, db = _to_device(w, dt), None if b is None else _to_device(b, NBXDtype.float32)
    dx = _to_device(x, dt)
    idx, wt = [], []
    for t in range(x.shape[0]):
        i, v = moe_route(dx[t:t + 1], dw, choice_bias=db, **a)
        assert i.nbx_dtype == NBXDtype.int64 and v.nbx_dtype == NBXDtype.float32
        assert tuple(i.shape) == tuple(v.shape) == (1, a["top_k"])
        idx.append(O.host_values(i)[0].astype(np.int64))
        wt.append(_host(v)[0])
    return np.stack(idx), np.stack(wt)


@pytest.mark.parametrize("dt", DTYPES, ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("name", list(CASES))
def test_one_token_is_the_oracle(dt, name):
    _skip_bf16(dt)
    E, H, bias_mean, a = CASES[name]
    x, w, b = draw(TOKENS, E, H, 41, bias_mean)
    xr, wr = _host(_to_device(x, dt)), _host(_to_device(w, dt))       # the numbers the kernel reads
    idx, wt = _route_on_device(x, w, b, dt, a)
    ref = O.moe_route_oracle(xr, wr, b, **a)()
    filled = int((ref[2][np.arange(TOKENS)[:, None], ref[0]] == a["fill"]).sum()) if a["fill"] == 0.0 else 0
    if "filled" in name:                       # the case is built for it: the draw must really route to filled experts
        assert filled > 0, f"{name}: the reference routes no token to a filled expert, the case tests nothing"
    worst, ties = compare(idx, wt, ref, a, name, xr, wr, WEIGHTS, CHOICE)
    print(f"{name} {dt.name}: weights {worst:.2e} (bound {WEIGHTS:.0e}); near-ties {len(ties)} of {TOKENS} {ties}; filled-expert slots {filled}")
    assert worst <= WEIGHTS


def test_the_two_fills_route_the_negative_draw_differently():
    E, H, bias_mean, a = CASES["E64-g8-tg2-k8-zero-H1000"]
    x, w, b = draw(TOKENS, E, H, 41, bias_mean)
    zero, _ = _route_on_device(x, w, b, NBXDtype.float32, a)
    inf, _ = _route_on_device(x, w, b, NBXDtype.float32, dict(a, fill=-math.inf))
    differ = sum(set(zero[t].tolist()) != set(inf[t].tolist()) for t in range(TOKENS))
    print(f"fill 0 against -inf: {differ} of {TOKENS} tokens routed differently")
    assert differ > 0


@pytest.mark.parametrize("dt", DTYPES, ids=["fp32", "fp16", "bf16"])
def test_one_group_is_every_group_kept_byte_for_byte(dt):
    _skip_bf16(dt)
    x, w, b = draw(TOKENS, 256, 7168, 43, 0.0)
    a = dict(SIGMOID, top_k=8, fill=-math.inf, scale=2.5)
    i1, w1 = _route_on_device(x, w, b, dt, dict(a, n_group=1, topk_group=1))
    i8, w8 = _route_on_device(x, w, b, dt, dict(a, n_group=8, topk_group=8))
    assert np.array_equal(i1, i8) and np.array_equal(w1, w8)
