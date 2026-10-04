"""A Metal launch wraps the extent its tensor can address, not the rest of the allocation it lives in.

`_buffer_for` wrapped an interior pointer from its address to the END of its allocation (`metal_native` binds every
buffer at offset 0, so a view has to start its own wrap). Every weight is such an interior pointer: it lives inside
its component's arena, one multi-GB block. Measured on the Mac (M4 Pro, 2026-10-04): the same 3072x3072 bf16 gemv
costs 0.49 ms on its own buffer, 10.9 ms as a view at the start of a 1 GiB block and 46.7 ms at the start of a 4 GiB
block, ~11.5 ms per GiB of wrap. In orpheus-3b's decode that was 848 launches per token at 7.4 ms each, 6.3 s per
token, 99 % of it inside launch(); the gate cell timed out at 2 400 s.

The launcher now hands a driver that asks for them (`wants_extents`) the byte extent each tensor argument can address
(from its shape and strides), and the Metal driver wraps that, rounded up to the page, never past the allocation.
Before this branch the timing case fails (the view at the block's start costs ~20x its own buffer); the values must
match either way.
"""
from __future__ import annotations

import time

import numpy as np
import pytest


def _metal_or_skip():
    try:
        from neurobrix.kernels.nbx_tensor import _detect_gpu_backend
        if _detect_gpu_backend() != "metal":
            pytest.skip("the wrap of an interior pointer is the Metal driver's; other drivers bind their own way")
        from neurobrix.triton import triton_ext_driver as drv
        return drv
    except Exception:
        pytest.skip("no Metal backend the engine can resolve")


N = K = 3072


def _bf16(rng, shape):
    from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
    a = rng.standard_normal(shape, dtype=np.float32) * 0.1
    return NBXTensor.from_numpy((a.view(np.uint32) >> 16).astype(np.uint16), dtype=NBXDtype.bfloat16)


def _ms_per_launch(fn, n=60):
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    for _ in range(5):
        fn()
    DeviceAllocator.sync_device()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    DeviceAllocator.sync_device()
    return (time.perf_counter() - t0) / n * 1e3


@pytest.fixture(scope="module")
def placed():
    _metal_or_skip()
    from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
    rng = np.random.default_rng(0)
    own, vec = _bf16(rng, (N, K)), _bf16(rng, (K,))
    block = NBXTensor.empty((1 << 30) // 2, device=own.device, dtype=NBXDtype.bfloat16)   # 1 GiB
    view = block.narrow(0, 0, N * K).view(N, K)                                           # at the block's START
    view.copy_(own)
    yield own, view, vec, block


def test_a_view_at_the_start_of_a_large_block_launches_as_cheaply_as_its_own_buffer(placed):
    from neurobrix.kernels import wrappers as W
    own, view, vec, _ = placed
    t_own = _ms_per_launch(lambda: W.mv_wrapper(own, vec))
    t_view = _ms_per_launch(lambda: W.mv_wrapper(view, vec))
    assert t_view < 3 * t_own + 0.5, (
        f"a gemv on a view at the start of a 1 GiB block costs {t_view:.2f} ms per launch against {t_own:.2f} ms on "
        f"its own buffer: the launch wraps the rest of the allocation, not the tensor")


def test_the_view_computes_what_its_own_buffer_computes(placed):
    from neurobrix.kernels import wrappers as W
    own, view, vec, _ = placed
    a = W.mv_wrapper(own, vec).numpy()
    b = W.mv_wrapper(view, vec).numpy()
    assert np.array_equal(a, b)
