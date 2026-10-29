"""NBXTensor slicing reads Python's bounds: a stop past an axis's extent is clamped to it, as torch and numpy do.

Found 2026-10-17 by the first judged run of a vendor's tiled decode in the Triton engine (CogVideoX-2b, 480x720):
the last tile row `z[..., 50:80, :]` over 60 latent rows came back 30 rows tall — `narrow` past the storage — and the
video was a grid of misplaced tiles at 600x864. What the test would do if the code were wrong: the shape or the
values differ from numpy's for the same key: red.
"""
import numpy as np
import pytest

from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.tensor import NBXTensor


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="needs a device")


@pytest.mark.parametrize("key", [
    (slice(None), slice(None), slice(None), slice(50, 80), slice(72, 117)),     # both past the end
    (slice(None), slice(None), slice(-2, None)),                                # from the end
    (slice(None), slice(None), slice(0, 1)),
    slice(0, 5),                                                                 # a lone slice past dim 0
    (slice(None), slice(None), slice(None), slice(70, 90)),                      # wholly past: empty
])
def test_a_slice_reads_numpys_bounds(key):
    a = np.arange(1 * 2 * 3 * 60 * 90, dtype=np.float32).reshape(1, 2, 3, 60, 90)
    t = NBXTensor.from_numpy(a).to_cuda()
    got = t[key]
    want = a[key]
    assert tuple(got.shape) == want.shape, (tuple(got.shape), want.shape)
    if want.size:
        assert np.array_equal(got.contiguous().numpy(), want)


def test_a_step_is_refused_by_name():
    t = NBXTensor.from_numpy(np.zeros((4, 4), dtype=np.float32)).to_cuda()
    with pytest.raises(RuntimeError, match="step 2"):
        t[0:4:2]
