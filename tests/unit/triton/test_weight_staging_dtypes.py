"""CPU-staged weights must be converted before they are copied, not after.

`_load_to_pinned_cpu` sizes its destination from the TARGET dtype and its copy
from the SOURCE array. Until 2026-09-03 it converted only the two bf16 pairs
its docstring named, so any other mismatched pair reached the copy at the
source width and wrote past the end of a `cudaMallocHost` region.

Kokoro-82M's decoder is exactly that case — a `(256,)` float32 tensor with a
float16 target, 1024 bytes into a 512-byte buffer. The full-zoo release gate
recorded it as a **240 s timeout**, not as an overflow, because the run never
got far enough to fail; the timeout was then read as machine contention. A
timeout is not a diagnosis.

It was caught at all only because pinned memory is registered with the driver,
which rejected the write. The identical defect on ordinary host memory
corrupts the heap in silence — which is why the guard here refuses on the size
mismatch rather than trusting the driver to notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXDtype
from neurobrix.triton.weight_loader import _load_to_pinned_cpu


def _has_gpu() -> bool:
    try:
        return DeviceAllocator.device_count() > 0
    except Exception:                                  # pragma: no cover
        return False


pytestmark = pytest.mark.skipif(
    not _has_gpu(), reason="pinned host allocation needs a CUDA runtime")


@pytest.mark.parametrize("source_np,source_dtype,target_dtype", [
    (np.float32, NBXDtype.float32, NBXDtype.float16),   # Kokoro-82M's decoder
    (np.float16, NBXDtype.float16, NBXDtype.float32),   # the widening direction
    (np.float32, NBXDtype.float32, NBXDtype.float32),   # no conversion at all
    (np.int32, NBXDtype.int32, NBXDtype.int64),
])
def test_the_staged_buffer_is_exactly_the_target_width(
        source_np, source_dtype, target_dtype):
    values = (np.arange(256, dtype=np.float64) * 0.5).astype(source_np)
    staged = _load_to_pinned_cpu(values.tobytes(), (256,),
                                 source_dtype, target_dtype)
    from neurobrix.kernels.nbx_tensor import dtype_size
    assert staged.nbytes() == 256 * dtype_size(target_dtype)


def test_the_values_survive_the_conversion():
    """A conversion that resizes correctly but scrambles the numbers would
    pass the size check and produce a silently wrong model."""
    values = (np.arange(256, dtype=np.float32) * 0.25)
    staged = _load_to_pinned_cpu(values.tobytes(), (256,),
                                 NBXDtype.float32, NBXDtype.float16)
    got = np.frombuffer(
        (np.ctypeslib.as_array(
            __import__("ctypes").cast(
                staged.data_ptr(),
                __import__("ctypes").POINTER(__import__("ctypes").c_uint16)),
            shape=(256,))).tobytes(), dtype=np.float16)
    np.testing.assert_allclose(got.astype(np.float32), values, rtol=1e-3)


def test_bf16_to_fp16_still_takes_its_bit_exact_path():
    """The bf16 paths are bit-level and predate this fix. The generic cast
    must not have displaced them — numpy cannot represent bf16 at all."""
    raw_u16 = np.array([0x3F80, 0x4000, 0xBF80], dtype=np.uint16)  # 1.0, 2.0, -1.0
    staged = _load_to_pinned_cpu(raw_u16.tobytes(), (3,),
                                 NBXDtype.bfloat16, NBXDtype.float16)
    assert staged.nbytes() == 6
    import ctypes
    got = np.ctypeslib.as_array(
        ctypes.cast(staged.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
        shape=(3,)).view(np.float16)
    np.testing.assert_allclose(got.astype(np.float32), [1.0, 2.0, -1.0])


def test_an_unconvertible_pair_refuses_instead_of_overflowing(monkeypatch):
    """The guard is the point: a dtype pair nobody handled must fail HERE,
    naming itself, rather than write past a buffer and be caught — or not —
    somewhere downstream."""
    import neurobrix.triton.weight_loader as wl

    real = wl.np.ascontiguousarray

    def sabotage(arr, *a, **k):
        # Simulate a future pair that escapes conversion: hand back an array
        # wider than the target buffer.
        out = real(arr, *a, **k)
        return out.astype(np.float64) if out.dtype == np.float16 else out

    monkeypatch.setattr(wl.np, "ascontiguousarray", sabotage)
    with pytest.raises(RuntimeError, match="would overflow"):
        _load_to_pinned_cpu(np.zeros(256, np.float16).tobytes(), (256,),
                            NBXDtype.float16, NBXDtype.float16)
