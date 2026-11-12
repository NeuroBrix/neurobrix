"""`ops.tiles`: a tensor computed tile by tile (the engine's TilingEngine, whose tile blend and tile read moved here
2026-11-12). A tile is read as a contiguous window, its edge replicated where it runs off the tensor; results are
crossfaded across their overlaps — weights that sum to 1 at every overlapped element — and divided once. What this
would do if wrong: a strided interior tile (a grid of wrong tile interiors, Sana 4Kpx conv::55), a box average (a line
at every tile stride, CogVideoX-2b / Wan2.1-T2V, 2026-09-27)."""
import numpy as np
import pytest

from nbxtensor.backend.runtime import gpu_backend
from nbxtensor.ops.tiles import TileBlend, crossfade_ramp, tile_window
from nbxtensor.tensor import NBXTensor

needs_device = pytest.mark.skipif(gpu_backend() is None, reason="no device")


def test_two_ramps_sum_to_one_across_their_overlap():
    head, tail = crossfade_ramp(8, 0, 4), crossfade_ramp(8, 4, 0)
    assert np.array_equal(head[:4], np.ones(4, np.float32))
    assert np.allclose(head[4:] + tail[:4], 1.0, atol=0)


@needs_device
@pytest.mark.parametrize("y, x", [(0, 0), (16, 16), (48, 40)])
def test_a_window_is_the_slice_padded_by_its_edge_and_contiguous(y, x):
    a = np.random.default_rng(0).standard_normal((1, 2, 64, 56)).astype(np.float32)
    got = tile_window(NBXTensor.from_numpy(a, device=0), (y, x), (32, 32))
    want = np.pad(a[:, :, y:y + 32, x:x + 32], ((0, 0), (0, 0), (0, max(0, y + 32 - 64)), (0, max(0, x + 32 - 56))),
                  mode="edge")
    assert got.is_contiguous() and np.array_equal(got.numpy(), want)


@needs_device
def test_tiles_of_a_constant_each_blend_to_a_continuous_whole():
    """Each tile answers its own constant on a 1x16 output, tiles of 8 at stride 4: the whole has no step larger than
    one ramp step, keeps the first tile's value where only it reaches, and equals the numpy crossfade."""
    starts, length = [0, 4, 8], 8
    like = NBXTensor.from_numpy(np.zeros((1, 1, 1, length), np.float32), device=0)
    blend = TileBlend(like, [1, 1, 1, 16], [1, 1, 1, 16], [(-1, starts, length)])
    want_num, want_den = np.zeros(16, np.float32), np.zeros(16, np.float32)
    for i, o in enumerate(starts):
        value = 10.0 * (i + 1)
        blend.add(NBXTensor.from_numpy(np.full((1, 1, 1, length), value, np.float32), device=0), (i,))
        before = starts[i - 1] + length - o if i else 0
        after = o + length - starts[i + 1] if i + 1 < len(starts) else 0
        w = crossfade_ramp(length, before, after)
        want_num[o:o + length] += value * w
        want_den[o:o + length] += w
    got = blend.result().numpy()[0, 0, 0]
    assert np.allclose(got, want_num / want_den, rtol=0, atol=1e-5)
    assert got[0] == 10.0 and float(np.abs(np.diff(got)).max()) <= 10.0 / 4 + 1e-4
