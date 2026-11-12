"""`ops.axis_slices`: an axis of extent P * total * inner read and written a slice of its `total` rows at a time, on
NBXTensor — the stitch of a computation run in slices (the engine's chunked region, moved here 2026-11-12). Each
result is compared with numpy's own slicing of the same array; a cyclic read that runs off the end continues from the
start; partials summed in the accumulator dtype and stored once in the output's."""
import numpy as np
import pytest

from nbxtensor.backend.runtime import gpu_backend
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import axis_slices as AX
from nbxtensor.tensor import NBXTensor

pytestmark = pytest.mark.skipif(gpu_backend() is None, reason="no device")

P, TOTAL, INNER = 2, 6, 3


def _whole():
    a = np.arange(4 * P * TOTAL * INNER * 5, dtype=np.float32).reshape(4, P * TOTAL * INNER, 5)
    return a, NBXTensor.from_numpy(a, device=0)


def _rows(a, start, length):
    """numpy's reference: rows [start, start+length) (modulo TOTAL) of the middle factor of axis 1."""
    v = a.reshape(4, P, TOTAL, INNER, 5)
    idx = [(start + i) % TOTAL for i in range(length)]
    return v[:, :, idx].reshape(4, P * length * INNER, 5)


@pytest.mark.parametrize("start, length", [(0, 2), (2, 3), (5, 1)])
def test_a_slice_is_numpys_slice_of_the_split_axis(start, length):
    a, x = _whole()
    assert np.array_equal(AX.take_slice(x, 1, TOTAL, INNER, start, length).numpy(), _rows(a, start, length))


def test_a_cyclic_read_past_the_end_continues_from_the_start():
    a, x = _whole()
    assert np.array_equal(AX.take_slice_cyclic(x, 1, TOTAL, INNER, 4, 4).numpy(), _rows(a, 4, 4))


def test_slices_put_back_rebuild_the_whole():
    a, x = _whole()
    full = None
    for start, length in [(0, 2), (2, 3), (5, 1)]:
        part = AX.take_slice(x, 1, TOTAL, INNER, start, length)
        if full is None:
            full = part.new_empty(AX.whole_shape(part.shape, 1, TOTAL, length))
        AX.put_slice(full, part, 1, TOTAL, INNER, start, length)
    assert np.array_equal(full.numpy(), a)


def test_a_broadcast_input_is_its_first_extent_at_every_slice():
    b = np.arange(4 * 1 * 5, dtype=np.float32).reshape(4, 1, 5)
    x = NBXTensor.from_numpy(np.repeat(b, TOTAL, axis=1), device=0)
    assert np.array_equal(AX.take_slice(x, 1, TOTAL, None, 3, 2).numpy(), np.repeat(b, 2, axis=1))


def test_partials_sum_in_the_accumulator_dtype_and_store_once_in_the_output_s():
    parts = [NBXTensor.from_numpy_as(np.full((3,), v, np.float32), NBXDtype.float16, device=0) for v in (0.5, 1.25)]
    acc = None
    for p in parts:
        acc = AX.accumulate(acc, p, NBXDtype.float32)
    assert acc.nbx_dtype == NBXDtype.float32
    out = AX.store(acc, NBXDtype.float16)
    assert out.nbx_dtype == NBXDtype.float16
    assert np.array_equal(out.to(NBXDtype.float32).numpy(), np.full((3,), 1.75, np.float32))
