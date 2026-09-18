"""An 8-D permute, cloned and flattened, must land where NumPy lands.

Wan2.1's DiT puts its patches back exactly the way HuggingFace's
`WanTransformer3DModel` does:

    view   [B, seq, C*p]          -> [B, T, Hp, Wp, p_t, p_h, p_w, -1]   (8-D)
    permute(0, 7, 1, 4, 2, 5, 3, 6)                                      (8-D)
    clone
    flatten(6,7) -> flatten(4,5) -> flatten(2,3)

Read from the container, the graph's chain is exactly that, so the STRUCTURE is
right and only its execution can be wrong. The rank is the reason to doubt it: a
prior 5-D-hardcoded strided-copy kernel silently dropped dimensions past the
fifth (`[1] * (5 - ndim)` returns `[]` for ndim > 5), corrupting 6-D PixArt
patchify clones and never reaching production on 8-D SANA-Video ones. The N-D
kernel replaced it. Nothing pins that 8-D case.

WHY IT IS WORTH PINNING. The `--triton` render of Wan2.1 at 20 steps is a field
with a period of exactly 16.00 pixels, measured by autocorrelation (r = 0.957)
and by a 2-D FFT whose dominant term is kx = 52 across an 832-pixel frame -- and
52 is exactly the number of patch columns. The VAE compresses space by 8 and the
patch is 2, so 8 x 2 = 16: the repeating unit is ONE DiT PATCH projected into
pixel space. A wrong permutation moves values without changing any of them, which
is why every statistical exclusion passed and why both execution modes show it.

This test does not assume the permute is the fault. It ASKS.

Run: PYTHONPATH=src python -m pytest tests/unit/kernels/test_the_unpatchify_permute_survives_eight_dimensions.py
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator

_TOTAL = DeviceAllocator.device_count()
needs_a_card = pytest.mark.skipif(_TOTAL == 0, reason="needs a CUDA device")

# Wan2.1-T2V-1.3B at 480x832x8: the shapes read out of the container's graph,
# not invented. seq = T * Hp * Wp = 10 * 30 * 52 = 15600, and C*p = 16*1*2*2 = 64.
B, T, HP, WP, PT, PH, PW, C = 1, 10, 30, 52, 1, 2, 2, 16
PERM = (0, 7, 1, 4, 2, 5, 3, 6)


def _reference(flat):
    """What the chain must produce, in NumPy, which knows nothing of this engine."""
    x = flat.reshape(B, T, HP, WP, PT, PH, PW, C)
    x = np.transpose(x, PERM)                      # [B, C, T, p_t, Hp, p_h, Wp, p_w]
    x = np.ascontiguousarray(x)
    x = x.reshape(B, C, T, PT, HP, PH, WP * PW)
    x = x.reshape(B, C, T, PT, HP * PH, WP * PW)
    return x.reshape(B, C, T, HP * PH, WP * PW)


@needs_a_card
def test_the_eight_dimensional_unpatchify_matches_numpy():
    """Values are an ARANGE, so a misplacement is visible as a wrong index.

    Random values would make a wrong permutation look like plausible noise --
    which is exactly the disguise the bug under investigation wears. With
    `arange`, every element carries its own source address.
    """
    n = B * T * HP * WP * PT * PH * PW * C
    flat = np.arange(n, dtype=np.float32)

    t = NBXTensor.from_numpy(flat.reshape(B, T, HP, WP, PT, PH, PW, C).copy())
    t = t.permute(*PERM)
    t = t.contiguous()
    DeviceAllocator.sync_device()
    got = t.numpy().reshape(B, C, T, PT, HP, PH, WP * PW)
    got = got.reshape(B, C, T, PT, HP * PH, WP * PW).reshape(B, C, T, HP * PH, WP * PW)

    expected = _reference(flat)
    assert got.shape == expected.shape, f"{got.shape} != {expected.shape}"
    np.testing.assert_array_equal(
        got, expected,
        err_msg="the 8-D unpatchify does not land where NumPy lands")


@needs_a_card
def test_a_permuted_clone_is_really_materialised():
    """`clone`/`contiguous` after an 8-D permute must COMPACT, not relabel.

    If it returns something still carrying the permuted strides, the
    `_unsafe_view` that follows reads the original memory order -- a per-patch
    scramble, with values all correct and all in the wrong place.
    """
    n = B * T * HP * WP * PT * PH * PW * C
    flat = np.arange(n, dtype=np.float32)
    t = NBXTensor.from_numpy(flat.reshape(B, T, HP, WP, PT, PH, PW, C).copy())
    p = t.permute(*PERM)
    c = p.contiguous()
    DeviceAllocator.sync_device()

    assert c.is_contiguous(), "contiguous() returned a non-contiguous tensor"
    # A compacted tensor read FLAT must equal the permutation read flat.
    np.testing.assert_array_equal(
        c.numpy().reshape(-1),
        np.ascontiguousarray(np.transpose(
            flat.reshape(B, T, HP, WP, PT, PH, PW, C), PERM)).reshape(-1),
        err_msg="the clone did not materialise the permuted layout")


@needs_a_card
@pytest.mark.parametrize("ndim", [4, 5, 6, 7, 8])
def test_a_reversing_permute_round_trips_at_every_rank(ndim):
    """The rank sweep crosses 5, where the old kernel silently truncated.

    Each rank uses distinct extents -- a permute among equal dimensions is a
    no-op whatever the strides do, so equal extents would make every rank pass.
    """
    shape = tuple(range(2, 2 + ndim))              # 2,3,4,... all different
    n = int(np.prod(shape))
    flat = np.arange(n, dtype=np.float32)
    perm = tuple(reversed(range(ndim)))

    t = NBXTensor.from_numpy(flat.reshape(shape).copy()).permute(*perm).contiguous()
    DeviceAllocator.sync_device()
    np.testing.assert_array_equal(
        t.numpy(), np.ascontiguousarray(np.transpose(flat.reshape(shape), perm)),
        err_msg=f"rank {ndim} permute+contiguous diverges from NumPy")
