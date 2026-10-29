"""Type promotion follows the Array API standard (Hocine 16:37, point 2): the tables of
https://data-apis.org/array-api/draft/API_specification/type_promotion.html, every pair of dtypes the library's table
holds, both operand orders. What the standard leaves unspecified (mixed kinds; float16 and bfloat16, which it does not
define) follows PyTorch, the engine's oracle — written in the README's Array API section."""
import itertools

import pytest

from nbxtensor.dtype.promotion import binary_result

_C = {"i1": "int8", "i2": "int16", "i4": "int32", "i8": "int64", "u1": "uint8", "u2": "uint16", "u4": "uint32",
      "u8": "uint64", "f4": "float32", "f8": "float64", "c8": "complex64", "c16": "complex128"}


def _table(rows, cols, cells):
    return {(_C[r], _C[c]): _C[v] for r, line in zip(rows, cells) for c, v in zip(cols, line.split())}


STANDARD = {
    **_table("i1 i2 i4 i8".split(), "i1 i2 i4 i8".split(), ["i1 i2 i4 i8", "i2 i2 i4 i8", "i4 i4 i4 i8", "i8 i8 i8 i8"]),
    **_table("u1 u2 u4 u8".split(), "u1 u2 u4 u8".split(), ["u1 u2 u4 u8", "u2 u2 u4 u8", "u4 u4 u4 u8", "u8 u8 u8 u8"]),
    **_table("i1 i2 i4 i8".split(), "u1 u2 u4".split(), ["i2 i4 i8", "i2 i4 i8", "i4 i4 i8", "i8 i8 i8"]),
    **_table("f4 f8 c8 c16".split(), "f4 f8 c8 c16".split(),
             ["f4 f8 c8 c16", "f8 f8 c16 c16", "c8 c16 c8 c16", "c16 c16 c16 c16"]),
}


@pytest.mark.parametrize("a,b", sorted(STANDARD))
def test_the_standards_tables_both_orders(a, b):
    assert binary_result(a, b) == STANDARD[(a, b)]
    assert binary_result(b, a) == STANDARD[(a, b)]


def test_what_the_standard_leaves_open_follows_pytorch():
    assert binary_result("float16", "bfloat16") == binary_result("bfloat16", "float16") == "float32"
    assert binary_result("int64", "float16") == "float16"          # an integer with a float: the float
    assert binary_result("bool", "int8") == "int8" and binary_result("uint8", "bool") == "uint8"
    with pytest.raises(ValueError, match="uint64"):                 # no signed integer holds uint64
        binary_result("int8", "uint64")


@pytest.mark.parametrize("a,b,c", list(itertools.combinations(["int8", "uint8", "int16", "float32", "complex64"], 3)))
def test_the_result_does_not_depend_on_the_order(a, b, c):
    r = {binary_result(binary_result(x, y), z) for x, y, z in itertools.permutations((a, b, c))}
    assert len(r) == 1, r


from nbxtensor.dtype.promotion import scalar_result  # noqa: E402


@pytest.mark.parametrize("dtype,scalar,want", [
    ("bool", True, "bool"), ("int8", 3, "int8"), ("uint8", 255, "uint8"), ("float16", 2, "float16"),
    ("float16", 0.5, "float16"), ("bfloat16", 1e30, "bfloat16"), ("complex64", 1, "complex64"),
    ("complex64", 2.5, "complex64"), ("complex128", 1j, "complex128"),
    ("float32", 1j, "complex64"), ("float64", 1j, "complex128"), ("float16", 1j, "complex64"),
])
def test_a_python_scalar_is_weak(dtype, scalar, want):
    assert scalar_result(dtype, scalar) == want


def test_what_the_standard_leaves_open_for_scalars_follows_nep50():
    assert scalar_result("int32", 1.5) == "float64"
    assert scalar_result("int8", 1j) == "complex128"
    assert scalar_result("bool", 2) == "int64"
    with pytest.raises(OverflowError, match="does not fit int8"):
        scalar_result("int8", 128)
    with pytest.raises(OverflowError, match="uint8"):
        scalar_result("uint8", -1)
    with pytest.raises(TypeError):
        scalar_result("float32", "1")
