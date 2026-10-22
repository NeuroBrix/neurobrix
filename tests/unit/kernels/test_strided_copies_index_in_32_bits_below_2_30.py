"""The strided copy kernels take 64-bit index arithmetic only when an extent needs it (`_index_wide`): the element
count or an operand's largest offset at 2^30 or more (a factor of two under 2^31 for a masked lane's overshoot).
64-bit everywhere made every copy's `%` and `//` emulated 64-bit on Apple (4.8 vs 48 GB/s, 2026-10-21)."""
from neurobrix.kernels.nbx_tensor import _index_wide, _max_linear_offset


def test_a_small_copy_indexes_in_32_bits():
    shape, strides = (1, 9, 128, 242, 362), (0, 87604, 788436, 362, 1)
    assert not _index_wide(1 * 9 * 128 * 242 * 362, _max_linear_offset(shape, strides))


def test_the_element_count_or_any_offset_at_2_30_takes_64_bits():
    assert _index_wide(1 << 30)
    assert _index_wide(1024, 1 << 30)
    assert _index_wide(1024, 5, -(1 << 30))
    assert not _index_wide((1 << 30) - 1, (1 << 30) - 1)
