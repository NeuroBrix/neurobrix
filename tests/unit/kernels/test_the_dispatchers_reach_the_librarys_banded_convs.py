"""The engine's band-streamed conv dispatchers (kernels/ops/fused_upsample_conv_torch) hand an NBXTensor to the
library's `tiled_conv2d` / `tiled_upsample_conv2d` (nbxtensor.ops.fused_upsample_conv): the same bytes. Torch-free; any
GPU backend nbxtensor drives."""
from __future__ import annotations

import numpy as np
import pytest

from nbxtensor.ops.fused_upsample_conv import FusionUpsampleProxy, tiled_conv2d, tiled_upsample_conv2d
from nbxtensor.tensor import NBXTensor


def _gpu_or_skip():
    try:
        from nbxtensor.backend.runtime import gpu_backend
        if gpu_backend() in ("cuda", "metal"):
            return
    except Exception:
        pass
    pytest.skip("no GPU backend nbxtensor drives here")


def _nbx(a):
    return NBXTensor.from_numpy(np.ascontiguousarray(a.astype(np.float32)))


def test_the_engines_dispatchers_hand_nbx_inputs_to_the_librarys_banded_convs():
    """kernels/ops/fused_upsample_conv_torch's `tiled_conv2d_spatial` / `fused_upsample_conv2d` route an NBXTensor to
    the library's `tiled_conv2d` / `tiled_upsample_conv2d`: the same bytes. Injection (seen red, 2026-10-30): the
    library's functions imported under the dispatchers' own names, which the dispatchers' definitions shadow — each
    called itself, RecursionError."""
    _gpu_or_skip()
    from neurobrix.kernels.ops.fused_upsample_conv_torch import fused_upsample_conv2d, tiled_conv2d_spatial
    rng = np.random.default_rng(6)
    x = rng.standard_normal((1, 4, 13, 9)).astype(np.float32)
    w = (rng.standard_normal((6, 4, 3, 3)) * 0.2).astype(np.float32)
    a = tiled_conv2d_spatial(_nbx(x), _nbx(w), None, 1, 1, 1, False, 0, 1, 3).numpy()
    b = tiled_conv2d(_nbx(x), _nbx(w), None, 1, 1, 1, 1, 1, 1, 1, 3).numpy()
    assert a.tobytes() == b.tobytes()
    proxy = FusionUpsampleProxy(_nbx(x), 2.0, 2.0, [1, 4, 26, 18])
    a = fused_upsample_conv2d(proxy, _nbx(w), None, 1, 1, 1, False, 0, 1, 4).numpy()
    b = tiled_upsample_conv2d(FusionUpsampleProxy(_nbx(x), 2.0, 2.0, [1, 4, 26, 18]), _nbx(w), None, 1, 1, 1,
                              False, 0, 1, 4).numpy()
    assert a.tobytes() == b.tobytes()
