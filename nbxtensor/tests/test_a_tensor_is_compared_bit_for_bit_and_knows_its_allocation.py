"""`NBXTensor.bits`, `same_bits`, `storage_root` and `shares_storage`: what the engine's invariant state asks before it
reuses a kept result (the same inputs, bit for bit) and before it keeps one (does the kept value view an input's
memory). They lived in the engine (triton/sequence.py `_nbx_bits` / `_nbx_same_bits` / `_nbx_shares_storage`) until
2026-11-12.

What this test would do if the code were wrong: `same_bits` true for +0.0 against -0.0, for another shape, dtype or a
single flipped bit — a kept result reused for other inputs — RED; false for one NaN against the same NaN, or for a
strided copy of equal bits — a reuse lost; `shares_storage` false for a view of a narrowed root, or true for two
allocations of equal values."""
import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.tensor import NBXTensor


def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def test_the_bits_are_the_same_bytes_at_the_element_width():
    _accelerator()
    x = np.array([[1.0, -0.0, np.nan], [np.inf, 2.5, -3.0]], dtype=np.float32)
    t = NBXTensor.from_numpy(x)
    assert t.bits().nbx_dtype == NBXDtype.int32 and t.bits().numpy().tobytes() == x.tobytes()
    assert t.t().bits().numpy().tobytes() == np.ascontiguousarray(x.T).tobytes()      # a strided view: packed first
    for dt, width in ((NBXDtype.float16, NBXDtype.int16), (NBXDtype.bfloat16, NBXDtype.int16),
                      (NBXDtype.int8, NBXDtype.uint8), (NBXDtype.int64, NBXDtype.int64)):
        assert t.to(dt).bits().nbx_dtype == width
    mask = NBXTensor.from_numpy(x > 0)
    assert mask.bits() is mask


@pytest.mark.parametrize("dt", [NBXDtype.float32, NBXDtype.bfloat16, NBXDtype.int64, NBXDtype.bool_])
def test_equal_bits_are_the_same_and_one_flipped_bit_is_not(dt):
    _accelerator()
    x = np.array([[0.5, np.nan, -1.25, 7.0], [3.0, 0.0, 1.0, -2.0]], dtype=np.float32)
    if dt == NBXDtype.bool_:
        x = x > 0
    a = NBXTensor.from_numpy(x).to(dt) if dt != NBXDtype.bool_ else NBXTensor.from_numpy(x)
    assert a.same_bits(a.clone())
    assert a.same_bits(a.t().contiguous().t())                     # equal bits through a strided view
    y = np.array(x)
    y[1, 3] = (not y[1, 3]) if dt == NBXDtype.bool_ else 5.0
    b = NBXTensor.from_numpy(y).to(dt) if dt != NBXDtype.bool_ else NBXTensor.from_numpy(y)
    assert not a.same_bits(b)
    assert not a.same_bits(a.reshape(4, 2))
    if dt != NBXDtype.bool_:
        assert not a.same_bits(a.to(NBXDtype.float16))


def test_plus_zero_and_minus_zero_are_not_the_same_bits():
    _accelerator()
    assert not NBXTensor.from_numpy(np.array([0.0], np.float32)).same_bits(
        NBXTensor.from_numpy(np.array([-0.0], np.float32)))
    assert NBXTensor.from_numpy(np.zeros((0, 3), np.float32)).same_bits(NBXTensor.from_numpy(np.zeros((0, 3), np.float32)))


def test_views_of_one_allocation_share_it_and_two_allocations_do_not():
    _accelerator()
    root = NBXTensor.from_numpy(np.arange(24, dtype=np.float32).reshape(4, 6))
    inner = root.narrow(0, 1, 2)
    assert inner.storage_root() is root and root.storage_root() is root
    assert inner.storage_address() == root.storage_address() and inner.data_ptr() != root.data_ptr()
    assert inner.shares_storage(root.t()) and root.shares_storage(inner.narrow(1, 2, 3))
    assert not root.shares_storage(root.clone())
    host = NBXTensor.empty_cpu((2, 2), NBXDtype.float32)       # its base is the numpy array: no device allocation
    assert host.storage_root() is host
