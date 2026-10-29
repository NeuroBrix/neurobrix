"""Both orientations of the conv2d tile (`SPATIAL_FAST`: [OUTF, BHW] with the spatial axis fast, or today's
[BHW, OUTF]) compute the convolution's definition: each launched explicitly at one tile against an fp64 reference,
over stride, padding, dilation and groups. The certifier picks the orientation per key per hardware (Metal: spatial
fast, 2x on CogVideoX's VAE; V100: today's on five shapes of six), so both must be exact, not only the one a
directory happens to serve."""
import numpy as np
import pytest
import triton

from nbxtensor.tensor import DeviceAllocator, NBXTensor
from nbxtensor.ops.oracles.conv2d_fp64 import conv2d_reference
from nbxtensor.ops.conv2d import conv2d_forward_kernel


def _gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


@pytest.mark.parametrize("spatial_fast", [False, True])
@pytest.mark.parametrize("N,C,H,Wd,O,k,stride,pad,dil,groups", [
    (1, 8, 16, 16, 16, 3, 1, 1, 1, 1),
    (2, 32, 19, 23, 24, 3, 1, 0, 1, 1),     # odd extents, a partial tile on every axis
    (1, 64, 16, 16, 32, 3, 2, 1, 1, 1),     # strided
    (1, 16, 20, 20, 16, 3, 1, 2, 2, 1),     # dilated
    (1, 32, 12, 12, 32, 3, 1, 1, 1, 4),     # grouped
    (3, 16, 9, 11, 40, 1, 1, 0, 1, 1),      # 1x1, a batch
])
def test_the_tile_orientation_computes_the_definition(spatial_fast, N, C, H, Wd, O, k, stride, pad, dil, groups):
    if not _gpu():
        pytest.skip("no device")
    rng = np.random.default_rng(7)
    x = rng.standard_normal((N, C, H, Wd)).astype(np.float32)
    w = rng.standard_normal((O, C // groups, k, k)).astype(np.float32)
    ref = conv2d_reference(x, w, stride=(stride, stride), padding=(pad, pad), dilation=(dil, dil), groups=groups)
    OH, OW = ref.shape[2], ref.shape[3]
    xt, wt = NBXTensor.from_numpy(x), NBXTensor.from_numpy(w)
    y = NBXTensor.empty((N, O, OH, OW), dtype=xt._dtype, device="cuda")
    bhw, outf, inf = 32, 32, 16
    grid = (triton.cdiv(N * OH * OW, bhw), triton.cdiv(O // groups, outf), groups)
    conv2d_forward_kernel.fn[grid](
        xt, wt, y, N, C, H, Wd, O, OH, OW, Wd, OW, *xt.stride(), *wt.stride(), *y.stride(),
        kernel_height=k, kernel_width=k, stride_height=stride, stride_width=stride,
        padding_height=pad, padding_width=pad, groups=groups, fp16=False,
        dilation_height=dil, dilation_width=dil,
        BLOCK_SIZE_BHW=bhw, BLOCK_SIZE_OUTF=outf, BLOCK_SIZE_INF=inf, SPATIAL_FAST=spatial_fast,
        num_warps=4, num_stages=2)
    got = y.numpy().astype(np.float64)
    err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-12)
    assert err < 1e-2, f"orientation spatial_fast={spatial_fast}: relative error {err:.3g}"
