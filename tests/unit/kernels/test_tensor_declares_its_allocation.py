"""A tensor may not declare more bytes than were allocated for it.

`_elem_size` and `_nbytes` are derived from the dtype ONCE, in the
constructor. A tensor built from a uint16 container and tagged float32 —
which is what the old `from_numpy` fallback did to every bf16 constant —
owns `numel * 2` bytes and declares `numel * 4`; a view with a non-zero
offset then lands on row 2*i instead of row i, and any copy sized from
`_nbytes` runs into whichever allocation follows. Measured 2026-09-07 on
TinyLlama: 66 rotary tensors, each declaring 524288 bytes over a
262144-byte buffer.
"""
import numpy as np
import pytest

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, dtype_size


def test_a_declared_dtype_sizes_the_tensor_by_its_container():
    bits = np.zeros((8, 4), dtype=np.uint16)          # bf16 bits
    t = NBXTensor.from_numpy(bits, dtype=NBXDtype.bfloat16)
    assert t.nbx_dtype == NBXDtype.bfloat16
    assert t._elem_size == 2
    assert t._nbytes == bits.nbytes == 64


def test_a_declared_dtype_that_contradicts_the_container_is_refused():
    bits = np.zeros((8, 4), dtype=np.uint16)
    with pytest.raises(ValueError, match="bytes per element"):
        NBXTensor.from_numpy(bits, dtype=NBXDtype.float32)


def test_an_unmapped_numpy_dtype_is_refused_rather_than_guessed():
    """Silently calling an unknown container float32 is what produced the
    two-to-one mismatch; the refusal names the dtype."""
    with pytest.raises(TypeError, match="uint16"):
        NBXTensor.from_numpy(np.zeros(4, dtype=np.uint16))


@pytest.mark.parametrize("np_dt,nbx_dt", [
    (np.float32, NBXDtype.float32), (np.float16, NBXDtype.float16),
    (np.int32, NBXDtype.int32), (np.int64, NBXDtype.int64),
    (np.uint8, NBXDtype.uint8), (np.bool_, NBXDtype.bool_),
])
def test_every_mapped_container_agrees_with_its_element_size(np_dt, nbx_dt):
    arr = np.zeros((3, 5), dtype=np_dt)
    t = NBXTensor.from_numpy(arr)
    assert t.nbx_dtype == nbx_dt
    assert t._elem_size == dtype_size(nbx_dt) == arr.dtype.itemsize
    assert t._nbytes == arr.nbytes


def test_offsets_are_bytes_of_the_real_element_size():
    """The doubled element size moved every offset view one row too far:
    row `i` of a bf16 table was read at byte `i * 64 * 4` instead of
    `i * 64 * 2`, so the model rotated by twice the angle."""
    bits = np.arange(8 * 4, dtype=np.uint16).reshape(8, 4)
    t = NBXTensor.from_numpy(bits, dtype=NBXDtype.bfloat16)
    row3 = t[3]
    assert row3.data_ptr() - t.data_ptr() == 3 * 4 * 2
