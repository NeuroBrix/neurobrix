"""A first-frame upsample reads its input through the input's strides.

The causal 3-D VAEs of CogVideoX and Open-Sora up-sample their first latent frame on its own branch:
`x[:, :, 0]` of a (B, C, T, H, W) activation goes through `aten::upsample_nearest2d`, the other
frames through the 3-D op. That select is a VIEW whose channel stride spans every frame. The Triton
kernel indexed its input as if it were contiguous, so frame 0 was assembled from other channels'
frames: Triton's frame 0 decoded flat grey (Open-Sora-v2, std 2.1 against 71.7) or blocky
(CogVideoX-2b, 31.7 dB against 45-51 for the other frames) from the same latent as the oracle.

What this test does if the code is wrong: the strided cases read the wrong bytes and fail; the
contiguous case keeps the old behaviour honest.

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_a_first_frame_upsample_reads_its_strides.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_a_first_frame_upsample_reads_its_strides.py
"""
from __future__ import annotations

try:
    import pytest
except ModuleNotFoundError:  # script-mode under the pytest-less GPU venv
    class _NoPytest:
        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore

import numpy as np


def _gpu_available() -> bool:
    """Whether the ENGINE has a GPU, asked of the engine (not of a vendor tool)."""
    import pytest

    nbx = pytest.importorskip("neurobrix.kernels.nbx_tensor")
    detect = nbx._detect_gpu_backend
    try:
        return detect() is not None
    except Exception:
        return False


def _download(t) -> np.ndarray:
    """Host copy of a CONTIGUOUS NBXTensor, as float16."""
    import ctypes
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    buf = (ctypes.c_char * t._nbytes)()
    DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(), t._nbytes, 2)
    return np.frombuffer(bytes(buf), dtype=np.float16).reshape(tuple(t.shape))


def _nearest2d_reference(x: np.ndarray, oh: int, ow: int) -> np.ndarray:
    """ATen nearest (legacy) indexing: src = floor(dst * in / out), clamped."""
    n, c, ih, iw = x.shape
    rows = np.minimum((np.arange(oh) * (ih / oh)).astype(np.int64), ih - 1)
    cols = np.minimum((np.arange(ow) * (iw / ow)).astype(np.int64), iw - 1)
    return x[:, :, rows][:, :, :, cols]


def _check(view_np: np.ndarray, view) -> None:
    from neurobrix.kernels.wrappers import upsample_nearest2d_wrapper
    n, c, ih, iw = view_np.shape
    out = upsample_nearest2d_wrapper(view, [2 * ih, 2 * iw], scales_h=2.0, scales_w=2.0)
    got = _download(out)
    want = _nearest2d_reference(np.ascontiguousarray(view_np), 2 * ih, 2 * iw)
    assert got.shape == want.shape, (got.shape, want.shape)
    assert np.array_equal(got, want), "upsample_nearest2d departs from the reference"


def test_a_selected_frame_is_upsampled_from_its_own_bytes() -> None:
    if not _gpu_available():
        pytest.skip("no GPU")
    from neurobrix.kernels.nbx_tensor import NBXTensor

    rng = np.random.default_rng(11)
    # (B, C, T, H, W): two batches so the batch stride is exercised too.
    x_np = rng.standard_normal((2, 5, 3, 6, 7)).astype(np.float16)
    x = NBXTensor.from_numpy(x_np).to("cuda:0")
    for frame in (0, 2):   # the first frame (offset 0) and a frame at a storage offset
        view = x.select(2, frame)
        assert not view.is_contiguous()
        _check(x_np[:, :, frame], view)


def test_a_transposed_plane_is_upsampled_from_its_own_bytes() -> None:
    if not _gpu_available():
        pytest.skip("no GPU")
    from neurobrix.kernels.nbx_tensor import NBXTensor

    rng = np.random.default_rng(12)
    x_np = rng.standard_normal((1, 4, 6, 9)).astype(np.float16)
    x = NBXTensor.from_numpy(x_np).to("cuda:0")
    view = x.transpose(2, 3)   # H and W strides swapped
    assert not view.is_contiguous()
    _check(np.swapaxes(x_np, 2, 3), view)


def test_a_contiguous_input_is_unchanged() -> None:
    if not _gpu_available():
        pytest.skip("no GPU")
    from neurobrix.kernels.nbx_tensor import NBXTensor

    rng = np.random.default_rng(13)
    x_np = rng.standard_normal((2, 3, 5, 4)).astype(np.float16)
    _check(x_np, NBXTensor.from_numpy(x_np).to("cuda:0"))


if __name__ == "__main__":
    test_a_selected_frame_is_upsampled_from_its_own_bytes()
    test_a_transposed_plane_is_upsampled_from_its_own_bytes()
    test_a_contiguous_input_is_unchanged()
    print("ok")
