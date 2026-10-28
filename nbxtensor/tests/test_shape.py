"""nbxtensor.shape: the layout rules, as pure functions on integers."""
import pytest

from nbxtensor import shape as S


def test_numel_and_the_scalar():
    assert S.numel((2, 3, 4)) == 24 and S.numel(()) == 1 and S.numel((5, 0)) == 0


def test_contiguous_strides():
    assert S.contiguous_strides((2, 3, 4)) == (12, 4, 1) and S.contiguous_strides(()) == ()


def test_an_extent_one_dim_carries_any_stride():
    # a transposed single-token head block (decode step) is contiguous
    assert S.is_contiguous((1, 32, 64, 1), (2048, 64, 1, 64))
    assert not S.is_contiguous((2, 3), (1, 2))
    assert S.is_contiguous((3, 2, 0), (7, 7, 7))


@pytest.mark.parametrize("old,strides,new,want", [
    ((2, 3, 4), (12, 4, 1), (6, 4), [4, 1]),                 # merge contiguous dims
    ((2, 3, 4), (12, 4, 1), (2, 12), [12, 1]),
    ((4, 3), (1, 4), (12,), None),                           # a transposed layout has no flat view
    ((4, 3), (1, 4), (2, 2, 3), [2, 1, 4]),                  # splitting the outer dim stays a view
    ((2, 1, 3), (3, 99, 1), (2, 3), [3, 1]),                 # an extent-1 dim's stride is never read
])
def test_view_strides_is_aten_compute_stride(old, strides, new, want):
    assert S.view_strides(old, strides, new) == want


def test_broadcasting_and_its_refusal():
    assert S.broadcast_shapes((3, 1), (1, 4)) == (3, 4)
    assert S.broadcast_shapes((5, 3, 1), (4,)) == (5, 3, 4)
    with pytest.raises(ValueError, match="Cannot broadcast"):
        S.broadcast_shapes((3,), (4,))


def test_broadcast_strides_is_expand():
    assert S.broadcast_strides((3, 1), (1, 1), (2, 3, 4)) == ((2, 3, 4), (0, 1, 0))
    assert S.broadcast_strides((3, 5), (5, 1), (-1, 5)) == ((3, 5), (5, 1))


def test_slice_bounds_clamp_and_refuse_a_step():
    assert S.slice_bounds(50, 80, 60) == (50, 60)
    assert S.slice_bounds(-3, None, 10) == (7, 10)
    assert S.slice_bounds(8, 2, 10) == (8, 8)
    with pytest.raises(RuntimeError, match="step 2"):
        S.slice_bounds(0, 10, 10, 2)


def test_max_linear_offset():
    assert S.max_linear_offset((2, 3), (3, 1)) == 5
    assert S.max_linear_offset((2, 3), (-3, 1)) == 5
    assert S.max_linear_offset((0, 3), (3, 1)) == 2
