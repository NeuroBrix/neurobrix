"""The in-place conv bias reaches every element of an output past 2^31 elements.

`conv2d_bias_inplace_kernel` read its program id in int32
(`tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)`), the form register 58 names:
the product wraps past 2^31 elements, so a biased convolution whose OUTPUT is
larger than two billion elements added its bias to the wrong addresses past the
boundary. The source gate (`test_flat_offsets_are_64_bit`) was RED on main for
this one site (kernel_units.log, 2026-09-25); this test holds the arithmetic on
a card, which a source gate cannot.

Shape: one map of 64 channels of 6000x6000 fp16 — 2 304 000 000 elements,
4.6 GB. 2^31 / 36 000 000 = 59.65, so channels 60 to 63 lie past the boundary.
The output holds 1.0, the bias of channel c is c/64 (exact in fp16); after the
epilogue channel c holds 1 + c/64 at its first and last element, before and past
the boundary alike. Seen RED with the int32 program id restored (2026-09-26,
card 2), GREEN with the widened one. Skipped, and said, without room for the
map.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src pytest tests/unit/kernels/test_the_conv_bias_reaches_past_two_billion_elements.py -p no:cacheprovider
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import NBXTensor

C, H, W_ = 64, 6000, 6000               # 2 304 000 000 elements; channel 60 starts past 2^31
BOUNDARY = 2 ** 31
NEEDED_BYTES = C * H * W_ * 2 + (1 << 30)   # one fp16 map + 1 GB headroom


def _cuda_free_bytes():
    try:
        import ctypes
        cuda = ctypes.CDLL("libcudart.so")
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        return free.value if cuda.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0 else 0
    except OSError:
        return 0


def _require_room():
    try:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        DeviceAllocator.empty_cache_pool()
    except Exception:
        pass
    free = _cuda_free_bytes()
    if free < NEEDED_BYTES:
        pytest.skip(f"needs {NEEDED_BYTES / 2 ** 30:.1f} GB free on one card; {free / 2 ** 30:.1f} GB free")


def test_the_bias_lands_on_every_channel_past_the_boundary():
    _require_room()
    assert (C - 1) * H * W_ >= BOUNDARY > 59 * H * W_, "the shape must straddle 2^31"
    out = NBXTensor.ones((1, C, H, W_), dtype="float16")
    bias = NBXTensor.from_numpy(np.arange(C, dtype=np.float16) / np.float16(64))
    out = W._conv_bias_inplace(out, bias)
    for c in (0, 59, 60, 61, 63):
        plane = out[0, c:c + 1].contiguous()
        first = float(plane[0, 0:1, 0:1].contiguous().numpy().reshape(-1)[0])
        last = float(plane[0, H - 1:H, W_ - 1:W_].contiguous().numpy().reshape(-1)[0])
        want = 1.0 + c / 64
        assert first == want and last == want, f"channel {c}: first {first}, last {last}, wanted {want}"
