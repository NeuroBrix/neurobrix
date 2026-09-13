"""`adaptive_avg_pool2d` and `conv_depthwise2d` against an fp64 oracle.

Why these two, and why now
--------------------------
2026-09-08 closed a silent wrong in the Metal lowerer: a `tl.arange` whose
only route to its `[:, None]` broadcast ran through `//` or `%` lost its axis
and took the COLUMN index, so every thread of a row addressed the same
element. `conv2d_forward_kernel` returned an array of the right shape and
plausible magnitude that was 99% wrong. Nothing refused, nothing warned.

A sweep of `kernels/ops/` for that same address shape — a range reaching a
broadcast through integer division — found six kernels:

    adaptive_avg_pool2d   conv1d   conv2d
    depthwise_conv2d      dequant_gemv   moe_decode_vec

`conv2d` is fixed and swept per config. `conv1d` delegates to `conv2d`.
`dequant_gemv` and `moe_decode_vec` already have oracle tests. **The other
two had no numerical test on any vendor** — `adaptive_avg_pool2d` had no test
at all, `depthwise_conv2d` only an import guard. They are exactly the shape
of defect that was just found: an output nothing was looking at.

These are vendor-agnostic: nothing here names a device, a prefix or a
backend, so the same assertions are the verdict wherever the wrapper
dispatches.

The oracles are computed in float64 in numpy and compared at the tolerance
the output dtype allows — not against another implementation of the same
idea, which would share its mistakes.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels.nbx_tensor import NBXTensor
from neurobrix.kernels.wrappers import (
    adaptive_avg_pool2d_wrapper,
    conv_depthwise2d_wrapper,
)


def _fp32_tol(ref: np.ndarray) -> float:
    """Absolute tolerance for an fp32 result: a few steps at the peak.

    A relative-per-element check would explode where the reference crosses
    zero and report a catastrophe on a correctly rounded answer.
    """
    return 8.0 * float(np.spacing(np.abs(ref).max() or 1.0, dtype=np.float32))


@pytest.mark.parametrize(
    "shape,out_size",
    [
        ((1, 3, 8, 8), 4),        # exact division: each cell is a 2x2 block
        ((1, 3, 7, 5), 3),        # ragged: cell bounds are floor/ceil, not uniform
        ((2, 4, 16, 16), 1),      # global average
        ((1, 2, 5, 5), 5),        # identity: every cell is one element
        ((1, 3, 12, 8), (3, 4)),  # non-square output: OH != OW
        ((1, 2, 9, 14), (2, 7)),  # non-square AND ragged in the h axis
        ((1, 2, 20, 6), (1, 3)),  # a 20-tall window, well past the old cap of 8
    ],
)
def test_adaptive_avg_pool2d_matches_an_fp64_oracle(shape, out_size):
    """The ragged case is the one that matters.

    Adaptive pooling's cell bounds are `floor(i*IH/OH)` to `ceil((i+1)*IH/OH)`,
    so for a non-dividing size the cells have DIFFERENT extents and each one's
    divisor differs. A collapsed index averages the wrong window and still
    returns finite, plausible numbers.
    """
    N, C, IH, IW = shape
    OH, OW = (out_size, out_size) if isinstance(out_size, int) else out_size
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)

    ref = np.zeros((N, C, OH, OW), dtype=np.float64)
    xd = x.astype(np.float64)
    for i in range(OH):
        h0, h1 = (i * IH) // OH, -(-((i + 1) * IH) // OH)
        for j in range(OW):
            w0, w1 = (j * IW) // OW, -(-((j + 1) * IW) // OW)
            ref[:, :, i, j] = xd[:, :, h0:h1, w0:w1].mean(axis=(2, 3))

    got = adaptive_avg_pool2d_wrapper(NBXTensor.from_numpy(x), out_size).numpy()
    # A separate divisor per axis is the point of the non-square cases: OH and
    # OW were collapsed to one `out_size` here, and a kernel that used the
    # wrong axis's size would still return the right SHAPE.
    assert got.shape == ref.shape, f"{got.shape} != {ref.shape}"
    assert np.isfinite(got).all(), "non-finite values in the pooled output"
    np.testing.assert_allclose(got.astype(np.float64), ref, atol=_fp32_tol(ref), rtol=0)


@pytest.mark.parametrize(
    "N,C,IH,IW,KH,KW,stride,padding",
    [
        (1, 4, 8, 8, 3, 3, 1, 1),   # padded: the border must read zeros, not neighbours
        (1, 3, 9, 7, 3, 3, 2, 1),   # strided and non-square
        (2, 5, 6, 6, 1, 1, 1, 0),   # 1x1: a pure per-channel scale
        (1, 2, 5, 5, 3, 3, 1, 0),   # unpadded: output smaller than input
    ],
)
def test_conv_depthwise2d_matches_an_fp64_oracle(N, C, IH, IW, KH, KW, stride, padding):
    """Padding is the discriminator.

    A depthwise convolution with `padding=1` must read ZERO outside the
    image. An address that collapses reads a real neighbour there instead,
    which changes the border and leaves the interior intact — a difference
    that averages away in any summary statistic and shows up only per element.
    """
    rng = np.random.default_rng(1)
    x = rng.standard_normal((N, C, IH, IW)).astype(np.float32)
    w = (rng.standard_normal((C, 1, KH, KW)) * 0.5).astype(np.float32)

    OH = (IH + 2 * padding - (KH - 1) - 1) // stride + 1
    OW = (IW + 2 * padding - (KW - 1) - 1) // stride + 1

    xp = np.pad(x.astype(np.float64),
                ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ref = np.zeros((N, C, OH, OW), dtype=np.float64)
    for oh in range(OH):
        for ow in range(OW):
            win = xp[:, :, oh * stride:oh * stride + KH, ow * stride:ow * stride + KW]
            ref[:, :, oh, ow] = (win * w[:, 0][None, :, :, :]).sum(axis=(2, 3))

    got = conv_depthwise2d_wrapper(
        NBXTensor.from_numpy(x), NBXTensor.from_numpy(w),
        stride=stride, padding=padding).numpy()

    assert got.shape == ref.shape, f"{got.shape} != {ref.shape}"
    assert np.isfinite(got).all(), "non-finite values in the depthwise output"
    np.testing.assert_allclose(got.astype(np.float64), ref, atol=_fp32_tol(ref), rtol=0)


def test_depthwise_border_is_not_the_interior():
    """A named guard against the one failure mode a shape check cannot see.

    With `padding=1` the output's border row is computed from a window that
    is two thirds zeros, so it must differ from the interior in a way a
    collapsed address would erase. If the border equals what an unpadded
    read would give, the padding was not applied.
    """
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, 2, 6, 6)).astype(np.float32) + 3.0   # keep it away from 0
    w = np.full((2, 1, 3, 3), 1.0 / 9.0, dtype=np.float32)           # a box filter

    got = conv_depthwise2d_wrapper(
        NBXTensor.from_numpy(x), NBXTensor.from_numpy(w),
        stride=1, padding=1).numpy()

    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    interior_mean = float(got[:, :, 1:-1, 1:-1].mean())
    border_mean = float(got[:, :, 0, :].mean())
    # x is centred on +3, so a box filter over a zero-padded border averages
    # roughly two thirds of that. The border MUST come out visibly lower.
    assert border_mean < interior_mean * 0.85, (
        f"border {border_mean:.4f} against interior {interior_mean:.4f}: "
        "the zero padding is not being read")
    expect_corner = float(xp[0, 0, 0:3, 0:3].sum() / 9.0)
    assert abs(float(got[0, 0, 0, 0]) - expect_corner) < 1e-5, (
        f"corner {got[0, 0, 0, 0]:.6f} against {expect_corner:.6f}")
