"""Prism's op-level tiled conv and fused upsample->conv, on the device (the NBX paths), against the
whole computation: every band of `launch_keys.tiled_conv2d_bands` — the halo rounded to whole
strides — lands where the untiled conv puts it, at stride 1 AND 2. Before 2026-09-29 the NBX paths
skipped the read-side halo as OUTPUT rows, so a stride-2 conv's internal bands were misaligned by half
an output row (the CPU twins are tested in tests/unit/census/). Injection: the halo left unrounded in
launch_keys.tiled_conv2d_bands -> the stride-2 cases, RED. Needs a CUDA device; skipped without one."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from neurobrix.kernels.nbx_tensor import NBXTensor, nbx_to_torch  # noqa: E402
from neurobrix.kernels.ops.fused_upsample_conv import (  # noqa: E402
    FusionUpsampleProxy, _fused_upsample_conv2d_nbx, _tiled_conv2d_spatial_nbx)


def _nbx(a):
    return NBXTensor.from_numpy(np.ascontiguousarray(a.astype(np.float32)))


@pytest.mark.parametrize("sh,kh,tf", [(1, 3, 3), (2, 3, 3), (2, 5, 4), (1, 1, 2)])
def test_the_tiled_conv_equals_the_whole_conv_on_the_device(sh, kh, tf):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 4, 37, 21)).astype(np.float32)
    w = (rng.standard_normal((6, 4, kh, 3)) * 0.2).astype(np.float32)
    ph = kh // 2
    ref = torch.nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(w),
                                     stride=(sh, 1), padding=(ph, 1)).numpy()
    got = nbx_to_torch(_tiled_conv2d_spatial_nbx(_nbx(x), _nbx(w), None, sh, 1, ph, 1, 1, 1, 1, tf)
                       ).float().cpu().numpy()
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-3, np.abs(got - ref).max()


@pytest.mark.parametrize("sh", [1, 2])
def test_the_fused_upsample_conv_equals_upsample_then_conv_on_the_device(sh):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((1, 4, 13, 9)).astype(np.float32)
    w = (rng.standard_normal((6, 4, 3, 3)) * 0.2).astype(np.float32)
    up = torch.nn.functional.interpolate(torch.from_numpy(x), scale_factor=2, mode="nearest")
    ref = torch.nn.functional.conv2d(up, torch.from_numpy(w), stride=(sh, 1), padding=(1, 1)).numpy()
    proxy = FusionUpsampleProxy(_nbx(x), 2.0, 2.0, list(up.shape))
    got = nbx_to_torch(_fused_upsample_conv2d_nbx(proxy, _nbx(w), None, (sh, 1), (1, 1), (1, 1),
                                                  False, (0, 0), 1, 4)).float().cpu().numpy()
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-3, np.abs(got - ref).max()
