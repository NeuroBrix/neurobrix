"""A convolution reads its whole input past 2^31 elements.

The 8x upscaler's last conv (64 -> 3 channels) at a 6344x6344 tile has an input
of 2 576 000 000 elements. Its OUTPUT is small (3 channels), so the wrapper does
not band-stream it, and the kernel's input-channel offset term
`input_in_feat_stride * inf_offset` was an int32 product: channel 54 of 64 sits
past 2^31 elements and the launch faulted with an illegal address (CUDA 700,
2026-09-20, `nbx/campaigns/2026_09_20_ladder/PIN_card2_700.md`, four of four runs
at a 9.6 GB tile budget). Register 58's class: every offset the kernel derives
from a program id was already 64-bit; the loop-derived channel offset was not.

Shape, and why: one image, 64 channels of 6000x6000 fp16 (36 000 000 elements a
channel, 2 304 000 000 in all, 4.6 GB); 2^31 / 36 000 000 = 59.65, so channels 60
to 63 lie entirely past the boundary. They hold 2.0 and the others 1.0; a 3x3
kernel of 1/(64*9) makes every interior output pixel (60*1 + 4*2)/64 = 1.0625.
Read wrongly, those four channels are garbage or a fault (or 1.0 if the read lands
on a plane of ones); read right, 1.0625. (First written as 1.125 — an arithmetic
slip caught by the green run's own value, which is the point of a value gate.)
Seen RED before the widening (the launch faulted), GREEN after.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src pytest tests/unit/kernels/test_a_conv_input_beyond_two_billion_elements.py -p no:cacheprovider
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import NBXTensor

C_IN, H = 64, 6000                      # 2 304 000 000 input elements; the boundary falls inside channel 59
C_PAST = 60                             # first channel entirely past 2^31 (2^31 // (H*H) = 59, so channel 60 starts at 2 160 000 000)
BOUNDARY = 2 ** 31
NEEDED_BYTES = C_IN * H * H * 2 + 3 * H * H * 4 + (1 << 30)     # fp16 input, fp32 output, 1 GB headroom


def _cuda_free_bytes():
    try:
        import ctypes
        cuda = ctypes.CDLL("libcudart.so")
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        return free.value if cuda.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0 else 0
    except OSError:
        return 0


def test_the_channels_past_the_boundary_are_read():
    assert C_PAST * H * H >= BOUNDARY > (C_PAST - 1) * H * H, "the boundary must fall inside the input, before the marked channels"
    try:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        DeviceAllocator.empty_cache_pool()
    except Exception:
        pass
    free = _cuda_free_bytes()
    if free < NEEDED_BYTES:
        pytest.skip(f"needs {NEEDED_BYTES / 2**30:.1f} GB free on the card, {free / 2**30:.1f} GB free")

    x = NBXTensor.ones((1, C_IN, H, H), dtype="float16", device="cuda")
    x[:, C_PAST:].fill_(2.0)                 # one contiguous tail block: channels 60..63, all past 2^31
    weight = NBXTensor.from_numpy(np.full((3, C_IN, 3, 3), 1.0 / (C_IN * 9), dtype=np.float16)).to("cuda")

    y = W.conv2d_wrapper(x, weight, None, stride=1, padding=1, dilation=1, groups=1)
    assert tuple(y.shape) == (1, 3, H, H)
    # interior pixels far from the borders, in every output channel, sampled across the image
    for (r, col) in ((100, 100), (H // 2, H // 2), (H - 100, H - 100), (37, H - 37)):
        v = y[:, :, r:r + 1, col:col + 1].contiguous().numpy().reshape(3)
        assert np.allclose(v, 1.0625, atol=2e-3), (r, col, v)
