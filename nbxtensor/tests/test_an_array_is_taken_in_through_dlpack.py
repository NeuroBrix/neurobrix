"""`nbxtensor.interop.from_dlpack`: another library's array taken in through DLPack, read with ctypes, no producer
imported (numpy here; torch in the engine's own test).

Before (audit 2026-11-12): the engine converted foreign tensors in four places, three of which ignored strides, so a
transposed or reversed view arrived in its buffer's order instead of the order it shows.

What this test does if the code is wrong: a transposed, a reversed or a sliced view reads back in the wrong order; a
dtype arrives as another; an empty array fails; a capsule taken twice is not refused — red.

Run: python -m pytest nbxtensor/tests/test_an_array_is_taken_in_through_dlpack.py
"""
import numpy as np
import pytest


def _device():
    from nbxtensor.memory.allocator import DeviceAllocator
    try:
        DeviceAllocator.set_device(0)
    except Exception as e:                                      # pragma: no cover
        pytest.skip(f"no device: {e}")


@pytest.mark.parametrize("make", [
    lambda a: a,                                                # C order
    lambda a: a.T,                                              # transposed: strides swapped
    lambda a: a[::-1, ::2],                                     # reversed rows, every other column: negative stride
    lambda a: a[1:3, 2:5],                                      # an interior slice: a byte offset into the buffer
], ids=["c-order", "transposed", "reversed-stepped", "slice"])
def test_the_values_shown_arrive_in_the_order_shown(make):
    from nbxtensor.interop import from_dlpack
    _device()
    view = make(np.arange(24, dtype=np.float32).reshape(4, 6))
    got = from_dlpack(view)
    assert tuple(got.shape) == view.shape
    assert np.array_equal(got.numpy(), view)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8",
                                   "bool"])
def test_each_dtype_arrives_as_itself(dtype):
    from nbxtensor.interop import from_dlpack
    _device()
    a = (np.arange(12) % 3 == 0) if dtype == "bool" else np.arange(12).astype(dtype)
    got = from_dlpack(a.reshape(3, 4))
    assert str(got.nbx_dtype.name).rstrip("_") == dtype
    assert np.array_equal(got.numpy(), a.reshape(3, 4))


def test_an_empty_array_arrives_empty():
    from nbxtensor.interop import from_dlpack
    _device()
    got = from_dlpack(np.zeros((0, 5), dtype=np.float32))
    assert tuple(got.shape) == (0, 5)


def test_a_capsule_is_taken_once():
    from nbxtensor.interop import dlpack
    a = np.ones(4, dtype=np.float32)

    class Once:
        capsule = a.__dlpack__()

        def __dlpack__(self):
            return self.capsule
    dlpack._taken(Once())[0].release()
    with pytest.raises(Exception):                              # the capsule is marked used: no second taker
        dlpack._taken(Once())


def test_the_producer_is_let_go_once_its_values_are_copied():
    """numpy's capsule holds the array it exports until the capsule's deleter runs: the array is collected after the
    copy only if `from_dlpack` called the deleter (a crossing that never did would keep every producer alive). The
    control keeps one capsule untaken-for-release, and its array stays."""
    import gc
    import weakref
    from nbxtensor.interop import dlpack, from_dlpack
    _device()
    a = np.ones(1024, dtype=np.float32)
    gone = weakref.ref(a)
    from_dlpack(a)
    del a
    gc.collect()
    assert gone() is None, "the producer is still held after its values were copied: the deleter never ran"
    b = np.ones(1024, dtype=np.float32)
    kept = weakref.ref(b)
    owner, _ = dlpack._taken(b)
    del b
    gc.collect()
    assert kept() is not None                                   # the instrument sees a capsule not let go
    owner.release()
    del owner                                                   # it also holds the producer, for the in-place path
    gc.collect()
    assert kept() is None
