"""nbxtensor.dtype.promotion: the one rule an op's floating inputs are brought to."""
import pytest

from nbxtensor.dtype import itemsize as I
from nbxtensor.dtype.promotion import float_math_input, promoted


def test_the_widest_float_wins_past_two_bytes():
    assert promoted(["float16", "float32"]) == "float32"
    assert promoted(["bfloat16", "float64", "float32"]) == "float64"
    assert promoted(["float32", "int64"]) == "float32"


def test_two_byte_floats_alone_are_not_promoted():
    """Which half dtype an op computes in is the caller's precision policy, never a promotion."""
    assert promoted(["float16", "bfloat16"]) is None
    assert promoted(["int64", "bool"]) is None
    assert promoted([]) is None


def test_a_float_math_op_reads_an_integer_as_float32():
    assert float_math_input("int64") == "float32" and float_math_input("bool") == "float32"
    assert float_math_input("float16") == "float16" and float_math_input("complex64") == "complex64"


def test_every_dtype_of_the_table_has_a_kind():
    for name in I.known_dtypes():
        assert I.kind(name) in ("float", "int", "uint", "bool", "complex")
    with pytest.raises(ValueError, match="unknown dtype"):
        I.kind("fp8_nonexistent")


def test_finite_maxima_come_from_the_table():
    assert I.finite_max("float16") == 65504.0
    assert I.finite_max("bfloat16") == pytest.approx(3.3895313892515355e38)
    assert I.finite_max("float8_e4m3fn") == 448.0
    with pytest.raises(ValueError, match="not a floating dtype"):
        I.finite_max("int32")


_OLD_PRIORITY = {"bool": 0, "uint8": 1, "int8": 2, "int16": 3, "int32": 4, "int64": 5, "float16": 6, "bfloat16": 6,
                 "float32": 7, "float64": 8, "complex64": 9, "complex128": 10}


def test_the_binary_result_is_the_wrappers_former_table_on_every_pair():
    """The table the Triton wrappers and Prism's width pass each kept (`_DTYPE_PRIORITY`, `_RANK`): the higher rank,
    the first operand on a tie — now one rule read from the dtype table's kind and width."""
    from nbxtensor.dtype.promotion import binary_result
    for a in _OLD_PRIORITY:
        for b in _OLD_PRIORITY:
            want = a if _OLD_PRIORITY[a] >= _OLD_PRIORITY[b] else b
            assert binary_result(a, b) == want, (a, b)
