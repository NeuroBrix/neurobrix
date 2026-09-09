"""A shape that was COMPUTED and went below zero is refused where it is made, not where it lands.

Wan2.1-VACE-1.3B, 2026-09-09: a conv3d received an input of (1, 3, 0, 450, 450) — zero frames —
against a temporal kernel of 3 with no temporal padding. The output extent computed to -2, the
byte count to -4,860,000, and the allocator answered `GPU malloc failed (error 2) for -4860000
bytes`, which reads as an out-of-memory and sends the reader to look for a bigger card.

Two guards, because the two answer different questions. The allocation boundary refuses ANY
negative extent, so the twelve extent formulas in the wrappers are covered once rather than
twelve times; and the conv3d path refuses its own non-positive temporal output, naming the frames
and the kernel, because that is where the cause is legible.
"""
from __future__ import annotations

import pytest

from neurobrix.kernels.nbx_tensor import NBXTensor


def test_a_negative_extent_is_refused_at_the_allocation_boundary():
    with pytest.raises(ValueError, match="negative extent"):
        NBXTensor.empty((1, 96, -2, 450, 450), dtype="float16")


def test_an_empty_tensor_is_still_legitimate():
    """Zero is a real extent — an empty tensor is a thing a caller means. Only the impossible
    is refused, or the guard would break every legitimate empty allocation."""
    t = NBXTensor.empty((0,), dtype="float16")
    assert tuple(t._shape) == (0,)
    t2 = NBXTensor.empty((4, 0, 8), dtype="float16")
    assert tuple(t2._shape) == (4, 0, 8)


class _View:
    """An NBXTensor as `_conv3d_via_conv2d` reads one before it allocates anything."""

    def __init__(self, shape):
        self.shape = shape
        self._device = "cpu"          # keeps the chunk-streaming gate out of the way
        self._dtype = None


def test_conv3d_refuses_zero_frames_by_naming_the_frames_not_the_bytes():
    from neurobrix.kernels.wrappers import _conv3d_via_conv2d
    x = _View((1, 3, 0, 450, 450))
    w = _View((96, 3, 3, 3, 3))
    with pytest.raises(RuntimeError, match=r"0 frame\(s\) in, temporal kernel 3"):
        _conv3d_via_conv2d(x, w, None, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1)


def test_conv3d_still_runs_the_shapes_that_have_an_output():
    """The guard must not stand in front of a convolution that has frames to convolve: it fires
    on the extent, and a valid extent walks past it into the real path."""
    from neurobrix.kernels import wrappers as W
    x = _View((1, 3, 8, 16, 16))
    w = _View((96, 3, 3, 3, 3))
    with pytest.raises(Exception) as e:          # it proceeds, and fails later for another reason
        W._conv3d_via_conv2d(x, w, None, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1)
    assert "frame(s) in, temporal kernel" not in str(e.value), (
        "a convolution with six frames of output was refused by the zero-frame guard")
