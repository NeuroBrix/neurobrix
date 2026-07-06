"""Unit test for the Triton max_pool3d_with_indices kernel.

Guard for the aten::max_pool3d_with_indices dispatch gap: video
conditioning masks (Allegro-TI2V state_video_mask downsampled to the
latent grid via F.max_pool3d) trace to this op, which had no triton
kernel — `[triton] Missing op: aten::max_pool3d_with_indices` in the
transformer loop. Kernel: kernels/ops/max_pool3d.py (temporal extension
of the FlagGems-derived 2D pool kernel).

Values must be BIT-EXACT vs torch (max selection, no arithmetic).
Indices must match exactly (first-max-in-window tie rule, flattened
(T, H, W) offsets per (N, C) plane).

Runnable two ways:
  - pytest:  PYTHONPATH=src /usr/bin/python3 -m pytest tests/unit/kernels/test_max_pool3d.py -v
  - script:  PYTHONPATH=src <gpu-venv>/bin/python tests/unit/kernels/test_max_pool3d.py
"""
from __future__ import annotations

try:
    import pytest
except ModuleNotFoundError:  # script-mode under the pytest-less GPU runtime venv
    class _NoPytest:
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore

import numpy as np
import torch

from neurobrix.kernels.dispatch import dispatch
from neurobrix.kernels.nbx_tensor import NBXTensor, nbx_to_torch


def _nbx(arr):
    return NBXTensor.from_numpy(np.ascontiguousarray(arr))


# (config-id, x.shape, kernel, stride, padding, dilation, ceil_mode)
_CONFIGS = [
    # Allegro-TI2V mask pooling: spatial-only window over a 5D mask.
    ("allegro_mask", (1, 1, 5, 14, 22), (1, 2, 2), (1, 2, 2), 0, 1, False),
    ("cube_222", (2, 3, 8, 10, 12), (2, 2, 2), (2, 2, 2), 0, 1, False),
    ("k3_s1_p1", (1, 2, 6, 9, 11), (3, 3, 3), (1, 1, 1), (1, 1, 1), 1, False),
    ("k3_s2_p1", (2, 2, 7, 12, 13), (3, 3, 3), (2, 2, 2), (1, 1, 1), 1, False),
    ("temporal_k31", (1, 4, 9, 6, 7), (3, 1, 1), (2, 1, 1), (1, 0, 0), 1, False),
    ("ceil_mode", (1, 2, 7, 9, 10), (2, 2, 2), (2, 2, 2), 0, 1, True),
]


@pytest.mark.parametrize("dt", ["float32", "float16"])
@pytest.mark.parametrize("cid,xs,k,s,p,d,cm", _CONFIGS)
def test_max_pool3d_matches_torch(cid, xs, k, s, p, d, cm, dt):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    npdt = np.float32 if dt == "float32" else np.float16
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal(xs).astype(npdt)

    ref_v, ref_i = torch.nn.functional.max_pool3d(
        torch.from_numpy(x_np).cuda(), k, s, p, d,
        ceil_mode=cm, return_indices=True)

    fn = dispatch("aten::max_pool3d_with_indices")
    assert fn is not None, "max_pool3d_with_indices missing from dispatch"
    got_v, got_i = fn(_nbx(x_np), list(k), list(s),
                      list(p) if isinstance(p, tuple) else p,
                      list(d) if isinstance(d, tuple) else d, cm)
    got_v = nbx_to_torch(got_v)
    got_i = nbx_to_torch(got_i)

    assert tuple(got_v.shape) == tuple(ref_v.shape), (
        f"{cid}/{dt}: shape {tuple(got_v.shape)} != {tuple(ref_v.shape)}")
    assert torch.equal(got_v.cpu(), ref_v.cpu()), (
        f"{cid}/{dt}: values not bit-exact, "
        f"max|d|={(got_v.float() - ref_v.float()).abs().max().item()}")
    assert torch.equal(got_i.cpu(), ref_i.cpu()), (
        f"{cid}/{dt}: indices differ "
        f"(#diff={int((got_i.cpu() != ref_i.cpu()).sum())})")


def test_max_pool3d_values_only_alias():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    fn = dispatch("aten::max_pool3d")
    assert fn is not None
    x = np.arange(2 * 4 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4, 4)
    got = nbx_to_torch(fn(_nbx(x), [2, 2, 2], [2, 2, 2], 0, 1, False)).cpu()
    ref = torch.nn.functional.max_pool3d(torch.from_numpy(x), 2, 2)
    assert torch.equal(got, ref)


if __name__ == "__main__":  # script mode
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    for dt in ("float32", "float16"):
        for cfg in _CONFIGS:
            test_max_pool3d_matches_torch(*cfg, dt)
            print(f"[OK] {cfg[0]} {dt}")
    test_max_pool3d_values_only_alias()
    print("[OK] values-only alias")
