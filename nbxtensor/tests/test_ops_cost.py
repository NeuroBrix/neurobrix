"""nbxtensor.ops._cost: what an op declares it allocates, moves and computes — card-less."""
import pytest

from nbxtensor.ops._cost import (Spec, elementwise_binary, elementwise_unary, native_layer_norm, reduction,
                                 rms_norm, softmax)


def _t(shape, dtype="float16"):
    return Spec.contiguous(shape, dtype)


def test_a_contiguous_unary_allocates_its_output_only():
    c = elementwise_unary(_t((4, 8)))
    assert [b.name for b in c.buffers] == ["output"] and c.peak_bytes == 4 * 8 * 2
    assert c.bytes_moved == 2 * 4 * 8 * 2 and c.flops == 32


def test_a_strided_unary_pays_its_contiguous_copy_until_it_returns():
    x = Spec((8, 4), (1, 8), "float32")                      # a transposed view
    c = elementwise_unary(x)
    assert [b.name for b in c.buffers] == ["contiguous input", "output"]
    assert c.peak_bytes == 2 * 32 * 4                         # both live at the return


def test_a_binary_casts_the_narrower_operand_and_broadcasts_by_views():
    a, b = _t((2, 1, 8), "float16"), _t((3, 8), "float32")
    c = elementwise_binary(a, b, common_dtype="float32", out_shape=(2, 3, 8))
    names = [x.name for x in c.buffers]
    assert names == ["a cast", "output"]                       # no expand copy: strided views
    assert c.outputs[0].nbytes == 2 * 3 * 8 * 4
    assert c.peak_bytes == 2 * 8 * 4 + 2 * 3 * 8 * 4


def test_softmax_over_a_non_last_dim_copies_the_rows():
    assert [b.name for b in softmax(_t((2, 5, 7)), -1).buffers] == ["output"]
    c = softmax(_t((2, 5, 7)), 1)
    assert [b.name for b in c.buffers] == ["rows made contiguous", "output"]
    assert c.flops == 5 * 70


def test_rms_norm_writes_into_its_contiguous_copy():
    strided = Spec((6, 4), (1, 6), "float16")
    c = rms_norm(strided, has_weight=True)
    assert [b.name for b in c.buffers] == ["output (in the contiguous copy)"] and c.peak_bytes == 6 * 4 * 2
    c2 = rms_norm(strided, has_weight=True, out_dtype="float32")    # another dtype: a second buffer
    assert c2.peak_bytes == 6 * 4 * 2 + 6 * 4 * 4


def test_layer_norm_returns_its_float32_row_statistics():
    c = native_layer_norm(_t((3, 10), "bfloat16"), (10,), True, True)
    assert [b.name for b in c.outputs] == ["output", "mean", "rstd"]
    assert c.peak_bytes == 3 * 10 * 2 + 2 * 3 * 4


@pytest.mark.parametrize("dim,out,want", [
    (-1, "float16", 6 * 4 + 6 * 2),                              # rows already last: accumulator + result
    (0, "float16", 2 * 3 * 4 * 2 + 12 * 4 + 12 * 2),             # dim 0 moved last: a copy first
    (-1, "float32", 6 * 4),                                       # a float32 result IS the accumulator
])
def test_a_reduction_accumulates_in_float32(dim, out, want):
    x = _t((2, 3, 4), "float16") if out != "float32" else _t((2, 3, 4), "float32")
    c = reduction(x, dim, False, out)
    assert c.peak_bytes == want and c.flops == 24
