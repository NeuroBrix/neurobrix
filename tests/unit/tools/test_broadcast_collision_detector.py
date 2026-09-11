"""The detector finds the shape upstream #9 mis-lowers, and nothing else.

Two failures are possible and each is worse than no detector:

  * finding nothing — the census then reports "0 carry the collision" over an
    empty set, which is true and says nothing. The tool refuses an empty
    census for that reason; this pins the detector itself.

  * finding everything — the first census run flagged three of TinyLlama's
    fourteen decode kernels on pairs like `32x1 -> 32x32` with
    `32x1 -> 32x64`. Both expand axis 1, whose source extent is 1, so the
    index contribution is zero in both and one shared expression is correct.
    That would have been a false non-zero on the path that matters most, and
    it would have been acted on.

The trigger is not "same source, different target". It is two broadcasts of
one source shape expanding DIFFERENT AXES: upstream's case took `1x2x1` to
`1x2x4` (axis 2) and to `4x2x1` (axis 0), so two index expressions were owed
and one was emitted.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/tools/test_broadcast_collision_detector.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from broadcast_shape_collision import collisions_in, _expanded_axes  # noqa: E402


UPSTREAM = """
 %a = tt.broadcast %x : tensor<1x2x1xi32> -> tensor<1x2x4xi32>
 %b = tt.broadcast %y : tensor<1x2x1xi1> -> tensor<4x2x1xi1>
"""

ORDINARY_2D = """
 %c = tt.broadcast %p : tensor<32x1xf32> -> tensor<32x32xf32>
 %d = tt.broadcast %q : tensor<32x1xf32> -> tensor<32x64xf32>
 %e = tt.broadcast %r : tensor<1x64xf32> -> tensor<32x64xf32>
 %f = tt.broadcast %s : tensor<1x64xf32> -> tensor<64x64xf32>
"""


def test_the_upstream_shape_is_found():
    assert collisions_in(UPSTREAM) == [("1x2x1", ["1x2x4", "4x2x1"])]


def test_the_ordinary_two_dimensional_case_is_not_flagged():
    """`offs_m[:, None]` with `offs_n[None, :]` is nearly every 2-D kernel we
    have. Flagging it makes the census useless by making it always positive."""
    assert collisions_in(ORDINARY_2D) == []


def test_one_source_expanded_along_one_axis_to_two_widths_is_not_a_collision():
    assert collisions_in(
        " tt.broadcast %a : tensor<8x1xf32> -> tensor<8x16xf32>\n"
        " tt.broadcast %b : tensor<8x1xf32> -> tensor<8x64xf32>\n") == []


def test_one_source_expanded_along_two_different_axes_is():
    assert collisions_in(
        " tt.broadcast %a : tensor<4x1x4xf32> -> tensor<4x8x4xf32>\n"
        " tt.broadcast %b : tensor<4x1x4xf32> -> tensor<16x1x4xf32>\n")


def test_a_single_broadcast_is_never_a_collision():
    assert collisions_in(" tt.broadcast %a : tensor<1x2x1xi32> -> tensor<1x2x4xi32>\n") == []


def test_an_identical_repeat_is_not_a_collision():
    assert collisions_in(
        " tt.broadcast %a : tensor<1x2x1xi32> -> tensor<1x2x4xi32>\n"
        " tt.broadcast %b : tensor<1x2x1xi32> -> tensor<1x2x4xi32>\n") == []


def test_the_expanded_axis_is_read_and_not_guessed():
    assert _expanded_axes("1x2x1", "1x2x4") == (2,)
    assert _expanded_axes("1x2x1", "4x2x1") == (0,)
    assert _expanded_axes("32x1", "32x64") == (1,)
    # a rank change is not comparable and must not silently look equal
    assert _expanded_axes("8", "8x4")[0] == "rank"


def test_nothing_at_all_reports_nothing_at_all():
    """Distinct from the census's refusal, which covers the empty RUN. Here:
    a kernel with no broadcast simply has no collision."""
    assert collisions_in("%0 = arith.addi %a, %b : i32\n") == []
