"""A biased convolution allocates ONE output, not two.

The malloc trace of real-esrgan-x8 @1024 (2026-09-21) showed the tile peak
at 8008 MB with three full maps live at once — and one of the three was the
conv wrapper's bias epilogue: `add(output, bias.view(1,-1,1,1))`, an
out-of-place broadcast born at `_prepare_binary_strided`, 2450 MB beside
the 2450 MB conv output it was about to replace. The input's last use IS
that add, the output is the wrapper's own fresh allocation, so the bias now
rides in place (`_conv_bias_inplace`), and this test holds both halves:

RED before the fix: two output-sized allocations per biased conv.
GREEN after: exactly one — and bit-agreement with the out-of-place law.
"""
from __future__ import annotations

import numpy as np
import pytest

triton = pytest.importorskip("triton")

from neurobrix.kernels.nbx_tensor import (  # noqa: E402
    DeviceAllocator, NBXDtype, NBXTensor,
)
from neurobrix.kernels import wrappers as w  # noqa: E402


def _gpu_or_skip():
    if DeviceAllocator.device_count() <= 0:
        pytest.skip("no device answers; the counting proof is a device's")


def _conv_args(rng, N=1, C=8, H=48, W=48, OC=8, K=3):
    x = NBXTensor.from_numpy(
        rng.standard_normal((N, C, H, W)).astype(np.float32))
    wt = NBXTensor.from_numpy(
        (rng.standard_normal((OC, C, K, K)) * 0.1).astype(np.float32))
    b = NBXTensor.from_numpy(rng.standard_normal(OC).astype(np.float32))
    return x, wt, b


def test_a_biased_conv_allocates_one_output_not_two():
    _gpu_or_skip()
    rng = np.random.default_rng(17)
    x, wt, b = _conv_args(rng)

    out_nbytes = 8 * 48 * 48 * 4          # the conv output's size

    # Warm the shape first: an uncertified conv sweeps at first touch and
    # the sweep's candidate buffers are output-sized — the counter below
    # must see the EPILOGUE's behavior, not autotune's.
    w.conv2d_wrapper(x, wt, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1)

    big = []
    orig = DeviceAllocator.malloc_cuda

    def counting(cls_or_size, *a, **k):
        # classmethod signature tolerance: first arg may be size
        size = cls_or_size if isinstance(cls_or_size, int) else a[0]
        if size >= out_nbytes:
            big.append(size)
        return orig(cls_or_size, *a, **k)

    DeviceAllocator.malloc_cuda = counting          # type: ignore[assignment]
    try:
        out = w.conv2d_wrapper(x, wt, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1)
    finally:
        DeviceAllocator.malloc_cuda = orig          # type: ignore[assignment]

    assert len(big) == 1, (
        f"a biased conv allocated {len(big)} output-sized blocks "
        f"({[s >> 10 for s in big]} KiB) — the bias epilogue is allocating "
        f"a second full map again")

    # the law is unchanged: in-place bias equals conv-then-broadcast-add
    ref = w.conv2d_wrapper(x, wt, None, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1)
    ref = w.add(ref, b.view(1, -1, 1, 1))
    got, want = out.numpy(), ref.numpy()
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5), (
        f"in-place bias diverges from the out-of-place law: "
        f"max abs {np.abs(got - want).max():.3e}")


def test_the_bias_is_cast_down_never_the_map_up():
    """bf16 conv + fp32 bias: the epilogue casts 256 bytes of bias, not
    the full map to fp32 (the standard add's wider-dtype rule would)."""
    _gpu_or_skip()
    rng = np.random.default_rng(23)
    x = NBXTensor.from_numpy(
        (((rng.standard_normal((1, 8, 32, 32)).astype(np.float32)
           .view(np.uint32) + 0x7FFF) >> 16).astype(np.uint16)),
        dtype=NBXDtype.bfloat16)
    wt = NBXTensor.from_numpy(
        (((rng.standard_normal((8, 8, 3, 3)).astype(np.float32) * 0.1)
          .view(np.uint32) + 0x7FFF) >> 16).astype(np.uint16),
        dtype=NBXDtype.bfloat16)
    b = NBXTensor.from_numpy(rng.standard_normal(8).astype(np.float32))
    out = w.conv2d_wrapper(x, wt, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1)
    assert out._dtype == NBXDtype.bfloat16, (
        f"the biased conv's output widened to {out._dtype} — the epilogue "
        f"upcast the map instead of casting the bias")
