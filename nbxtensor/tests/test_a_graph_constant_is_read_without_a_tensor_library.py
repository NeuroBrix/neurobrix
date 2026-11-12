"""`io.constants`: a graph constant embedded as the archive `torch.save` writes is read without any tensor library —
its first storage out of the ZIP, then its bytes as the array of the dtype the engine binds it in
(`constant_load_dtype`). bfloat16 is widened exactly or narrowed to float16, or kept as its bits DECLARED bfloat16;
never handed over as uint16 under another name (TinyLlama's rotary table, 2026-09-07: two bf16 ones read as one float,
1.0019378). Moved from the engine's `_load_constant_triton` (2026-11-12)."""
import io
import zipfile

import numpy as np
import pytest

from nbxtensor.dtype.convert import f32_to_bf16_bits
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.io.constants import archived_storage, constant_array


def _archive(payload: bytes, with_data: bool = True) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", b"pickle")
        if with_data:
            zf.writestr("archive/data/0", payload)
    return buf.getvalue()


def test_the_first_storage_comes_out_of_the_archive_and_none_without_one():
    assert archived_storage(_archive(b"\x01\x02")) == b"\x01\x02"
    assert archived_storage(_archive(b"", with_data=False)) is None


VALUES = np.array([[1.0, -2.5], [0.15625, 3.0e4]], dtype=np.float32)


@pytest.mark.parametrize("compute, want_dtype, declared", [("float32", np.float32, None), ("float16", np.float16, None),
                                                            ("bfloat16", np.uint16, NBXDtype.bfloat16)])
def test_bfloat16_is_decoded_to_what_the_compute_dtype_binds(compute, want_dtype, declared):
    bits = f32_to_bf16_bits(VALUES)
    arr, bits_are = constant_array(bits.tobytes(), "bfloat16", (2, 2), compute, stores_fp64=False)
    assert arr.dtype == want_dtype and arr.shape == (2, 2) and bits_are == declared
    if declared is None:
        widened = (bits.astype(np.uint32) << 16).view(np.float32)
        assert np.array_equal(arr, widened.astype(want_dtype))
    else:
        assert np.array_equal(arr, bits)


def test_float64_is_held_as_float32_where_the_store_has_no_float64_and_kept_where_it_has():
    raw = np.arange(4, dtype=np.float64).tobytes()
    assert constant_array(raw, "float64", (4,), "float32", stores_fp64=False)[0].dtype == np.float32
    assert constant_array(raw, "float64", (4,), "float32", stores_fp64=True)[0].dtype == np.float64


def test_an_integer_constant_keeps_its_dtype():
    arr, bits_are = constant_array(np.arange(3, dtype=np.int64).tobytes(), "int64", (3,), "bfloat16", False)
    assert arr.dtype == np.int64 and list(arr) == [0, 1, 2] and bits_are is None
