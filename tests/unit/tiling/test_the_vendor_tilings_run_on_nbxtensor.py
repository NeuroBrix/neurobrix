"""The vendor tilings (`core/module/vendor_tiling.py`) run on NBXTensor, on the device, with no torch: the Triton
engine's path through `ramp_crop` (the AutoencoderKL family's `tiled_decode`) and `mask_sum` (Allegro's). With a tile
function that hands its tile back, each vendor blend is a partition of unity — the ramps of two overlapping tiles sum
to one, the crop keeps each tile's own stride — so the output IS the input; the vendor equivalence itself is
`tests/unit/module/test_a_vendor_tiling_is_the_vendors.py`'s (on the vendor's own models, torch side). Run in the
torch-free environment by the 3.21 proof (supervisor, 2026-11-04 13:40: executed, not only imported).

What this test would do if the code were wrong: a ramp reversed or on the wrong end, a blend over the wrong rows, a
tile placed at the wrong stride, the output not zeroed where a tile adds — red at the comparison; a torch import on
the NBXTensor path — red in the torch-free environment (and a NameError-free pass here proves nothing of that, which
is why the proof runs it there). Needs a device."""
from __future__ import annotations

import numpy as np
import pytest


def _card():
    try:
        from nbxtensor.memory.allocator import DeviceAllocator
        return DeviceAllocator.device_count() >= 1
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _card(), reason="needs a device")


def _nbx(a):
    from nbxtensor.tensor import NBXTensor
    return NBXTensor.from_numpy(np.ascontiguousarray(a)).to("cuda")


def test_ramp_crop_hands_back_its_input():
    from neurobrix.core.module.vendor_tiling import ramp_crop
    z = np.random.default_rng(0).standard_normal((1, 4, 3, 40, 56)).astype(np.float32)
    decl = {"tile": [16, 16], "stride": [12, 12], "blend": [4, 4], "crop": [12, 12],
            "in_axes": [3, 4], "out_axes": [3, 4]}
    out = ramp_crop(_nbx(z), decl, lambda t: t).numpy()
    np.testing.assert_allclose(out, z, rtol=0, atol=1e-6)


def test_mask_sum_hands_back_its_input():
    from neurobrix.core.module.vendor_tiling import mask_sum
    x = np.random.default_rng(1).standard_normal((1, 4, 12, 40, 40)).astype(np.float32)
    decl = {"tile_in": [8, 16, 16], "stride_in": [4, 12, 12], "tile_out": [8, 16, 16], "stride_out": [4, 12, 12],
            "in_axes": [2, 3, 4], "out_axes": [2, 3, 4], "out_scale": [[1, 1], [1, 1], [1, 1]]}
    out = mask_sum(_nbx(x), decl, lambda t: t).numpy()
    np.testing.assert_allclose(out, x, rtol=0, atol=1e-5)
