"""The 2-D reduction tile is clamped to what the backend stages correctly.

Measured 2026-09-13 from the IR swin2SR actually produces: `l2_norm_kernel`
forms an [8, 1024] = 8192-element tile and reduces it along axis 1. The Metal
lowering stages one element per thread, so 8192 > the 1024-thread threadgroup
and it refuses -- fatally, because the norm wrapper calls the Triton kernel
directly with no per-op ATen fallback (unlike conv, whose refusals fall back).

`_RED_BM, _RED_BN = 8, 1024` is a CUDA-tuned constant. The fix is a backend
capability -- Metal 1024, cuda/hip far higher -- clamping BLOCK_N so the tile
fits, which adds loop iterations and changes no result. One helper for
std/var/norm, not a copy in each.

Runnable: PYTHONPATH=src python3 -m pytest \
    tests/unit/kernels/test_reduction_tile_is_a_capability.py -v
"""
from __future__ import annotations

import pytest


def test_every_known_backend_has_a_row():
    from neurobrix.kernels.nbx_tensor import (
        _BACKEND_REDUCE_TILE_MAX, _BACKEND_TRAPS_ON_DEVICE_ASSERT)

    missing = set(_BACKEND_TRAPS_ON_DEVICE_ASSERT) - set(_BACKEND_REDUCE_TILE_MAX)
    assert not missing, (
        f"a backend with a row elsewhere and none here would inherit no cap: "
        f"{sorted(missing)}")


def test_cuda_takes_the_full_8192_tile_unchanged():
    """A shared-engine change is numerically inert on CUDA unless it fixes a
    bug, and this fixes one only on Metal: cuda/hip must still take 8x1024."""
    from neurobrix.kernels.nbx_tensor import _BACKEND_REDUCE_TILE_MAX

    assert _BACKEND_REDUCE_TILE_MAX["cuda"] >= 8192
    assert _BACKEND_REDUCE_TILE_MAX["hip"] >= 8192


def test_metal_caps_at_the_threadgroup():
    from neurobrix.kernels.nbx_tensor import _BACKEND_REDUCE_TILE_MAX

    assert _BACKEND_REDUCE_TILE_MAX["metal"] == 1024


def test_the_clamp_shrinks_only_block_n_and_keeps_the_product_legal(monkeypatch):
    """The helper's arithmetic, both regimes, without a GPU.

    Over the cap: BLOCK_M kept (it sets the row grid), BLOCK_N shrunk so the
    product fits. Under it: untouched. The product must never exceed the cap.
    """
    import neurobrix.kernels.wrappers as W


    monkeypatch.setattr(W, "_reduce_tile_max", lambda: 1024)
    _, bm, bn = W._reduction_tile(4096)
    assert bm == W._RED_BM, "BLOCK_M sets the grid and must be kept"
    assert bm * bn <= 1024, f"tile {bm}x{bn} still exceeds the cap"
    assert bn < W._RED_BN, "over the cap, BLOCK_N must shrink"

    monkeypatch.setattr(W, "_reduce_tile_max", lambda: 1 << 20)
    _, bm2, bn2 = W._reduction_tile(4096)
    assert (bm2, bn2) == (W._RED_BM, W._RED_BN), (
        "under the cap nothing moves -- the CUDA path")


def test_a_backend_without_a_row_refuses():
    from neurobrix.kernels.nbx_tensor import _backend_capability

    with pytest.raises(RuntimeError):
        _backend_capability({"cuda": 1}, "_T", "its reduction tile cap")


# ── the numerical verdict: the clamped tile computes the right norm ─────────

import sys                                                          # noqa: E402


@pytest.mark.skipif(sys.platform != "darwin", reason="the Metal clamp path")
def test_l2_norm_with_the_clamped_tile_matches_aten():
    """Shrinking BLOCK_N must change the numbers by nothing but float order.

    On the swin2SR shape (a row of 8192) the clamp takes BLOCK_N 1024 -> 128;
    the L2 norm over that row must still match torch. Measured through the
    real wrapper, against an independent implementation.
    """
    import numpy as np
    torch = pytest.importorskip("torch")
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.nbx_tensor import NBXTensor

    rng = np.random.default_rng(20260913)
    x = (rng.standard_normal((4, 8192)) * 0.1).astype(np.float32)

    got = np.asarray(
        W.vector_norm_wrapper(NBXTensor.from_numpy(x), ord=2.0, dim=1)
        .to_cpu().numpy(), dtype=np.float64).ravel()
    want = torch.linalg.vector_norm(
        torch.from_numpy(x), ord=2, dim=1).to(torch.float64).numpy().ravel()

    scale = float(np.abs(want).max()) or 1.0
    dev = float(np.abs(got - want).max() / scale)
    assert dev <= 1e-4, (
        f"the clamped-tile L2 norm differs from ATen by {dev:.3e}; a tile that "
        f"lowers but computes the wrong norm is the outcome this guards")


@pytest.mark.skipif(sys.platform != "darwin", reason="the Metal clamp path")
def test_weight_norm_with_the_clamped_tile_matches_aten():
    """weight_norm's tile is 64x256 = 16384 (Kokoro's wall), a different
    kernel and constant from l2_norm's 8192 -- same clamp helper. The norm it
    produces must still match torch after BLOCK_N is shrunk 256 -> 16.
    """
    import numpy as np
    torch = pytest.importorskip("torch")
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.nbx_tensor import NBXTensor

    rng = np.random.default_rng(20260913)
    v = (rng.standard_normal((32, 16384)) * 0.1).astype(np.float32)   # dim=0, N=16384
    g = (rng.standard_normal((32, 1)) * 0.1 + 1.0).astype(np.float32)

    out, _ = W.weight_norm_interface_wrapper(
        NBXTensor.from_numpy(v), NBXTensor.from_numpy(g), dim=0)
    got = np.asarray(out.to_cpu().numpy(), dtype=np.float64)

    tv, tg = torch.from_numpy(v), torch.from_numpy(g)
    tnorm = torch.linalg.vector_norm(tv, dim=1, keepdim=True)
    want = (tv * (tg / tnorm)).to(torch.float64).numpy()

    scale = float(np.abs(want).max()) or 1.0
    dev = float(np.abs(got - want).max() / scale)
    assert dev <= 1e-3, (
        f"clamped-tile weight_norm differs from torch by {dev:.3e}")
