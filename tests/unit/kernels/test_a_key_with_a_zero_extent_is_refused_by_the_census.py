"""A key whose extent is zero names a launch that cannot exist: no tensor has a side of zero,
no certifier can synthesise one, and no request forms it. The shadow reaches them anyway at
the bottom of a value-derived extent — a four-token speech through chatterbox's vocoder emits
convolution keys with in_width 0 — and one is already in the SERVED directory
(`conv2d_forward_kernel.fp32` carries a key whose batch_dim is 0), so a census recorded them
before anyone read one (2026-09-22). The census refuses them the way it already refuses a
negative extent, on the positions that are extents and not the ones that may legitimately
be zero.

What this test would do if the code were wrong: drop a position from `EXTENT_POSITIONS` and
that position's case goes green-to-red here; guard a padding position instead and the
"padding may be zero" case fails. Both injections were run.

Shapes: the convolution key is chatterbox's own vocoder form (one row, 64 channels, kernel 3)
with padding 1 and padding 0 both exercised, because padding is the one position where zero
is the ordinary case and a blanket refusal would silently shrink every census.
"""
from __future__ import annotations

import pytest

from neurobrix.kernels.autotune_certified import EXTENT_POSITIONS, degenerate_extent

CONV = "neurobrix.kernels.ops.conv2d.conv2d_forward_kernel"
DW = "neurobrix.kernels.ops.depthwise_conv2d.depthwise_conv2d_kernel"
MM = "neurobrix.kernels.ops.matmul.matmul_kernel"

GOOD_CONV = (1, 64, 1, 4021, 64, 1, 4021, 1, 3, 1, 1, 0, 1, 1, 1, 1, False, "fp16", "fp16", "fp16")
GOOD_DW = (64, 1, 4021, 1, 4021, 1, 3, 1, 1, 0, 1, True, "fp16", "fp16", "fp16")
GOOD_MM = (240, 2048, 2048, True, True, "fp16", "fp16", "fp32")


@pytest.mark.parametrize("qual, key", [(CONV, GOOD_CONV), (DW, GOOD_DW), (MM, GOOD_MM)])
def test_a_launch_that_can_exist_is_not_refused(qual, key):
    assert degenerate_extent(qual, key) is None


def test_the_padding_positions_may_be_zero():
    no_padding = GOOD_CONV[:11] + (0, 0) + GOOD_CONV[13:]
    assert degenerate_extent(CONV, no_padding) is None, no_padding


@pytest.mark.parametrize("pos", EXTENT_POSITIONS["conv2d_forward_kernel"])
def test_every_guarded_convolution_position_refuses_a_zero(pos):
    bad = GOOD_CONV[:pos] + (0,) + GOOD_CONV[pos + 1:]
    assert degenerate_extent(CONV, bad) == f"position {pos} = 0", bad


def test_the_width_a_four_token_speech_produced_is_refused():
    """The shadow's own record: a one-row convolution whose input width came out 0."""
    bad = (1, 64, 1, 0, 64, 1, 0, 1, 3, 1, 1, 0, 1, 1, 1, 1, False, "fp16", "fp16", "fp16")
    assert degenerate_extent(CONV, bad) == "position 3 = 0"


def test_the_entry_the_served_directory_already_holds_is_refused():
    """`conv2d_forward_kernel.fp32` carries this key; batch_dim 0 is not a launch."""
    entry = (0, 128, 448, 448, 128, 448, 448, 3, 3, 1, 1, 1, 1, 1, 1, 1, False, "fp32", "fp32", "fp32")
    assert degenerate_extent(CONV, entry) == "position 0 = 0"


def test_a_kernel_the_table_does_not_name_is_answered_none_never_guessed():
    assert degenerate_extent("neurobrix.kernels.ops.rms_norm.rms_norm_kernel", (0, 0, 0)) is None


def test_the_recorder_consults_the_refusal_and_writes_nothing(tmp_path, monkeypatch):
    """The refusal exists to be CALLED: a helper the recorder never consults is silence
    wearing the shape of a check."""
    from neurobrix.kernels import census

    rec = tmp_path / "keys"
    monkeypatch.setenv("NBX_KEY_RECORD", str(rec))
    monkeypatch.setattr(census, "key_line", lambda tuned, key: f"{CONV}::{tuple(key)!r}")
    census._RECORDED.clear()
    census._SAID.clear()

    bad = (1, 64, 1, 0, 64, 1, 0, 1, 3, 1, 1, 0, 1, 1, 1, 1, False, "fp16", "fp16", "fp16")
    census.record(object(), bad)
    assert not rec.exists() or rec.read_text() == "", rec.read_text()

    census.record(object(), GOOD_CONV)
    assert rec.exists() and str(GOOD_CONV[3]) in rec.read_text()
