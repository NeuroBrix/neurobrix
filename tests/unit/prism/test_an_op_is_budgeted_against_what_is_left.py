"""An op's transient is budgeted against what is LEFT of the card, not the card.

`overflow_ops` used to ask whether one op's own footprint -- its largest input,
its outputs and its workspace -- cleared 0.85 of the GPU. That is a question
about an EMPTY card, and the card is never empty when the op runs. Two things
are already on it:

  the component's weights   resident for the whole of its execution, and known
                            only to the caller, so passed in as `resident_bytes`
  the live activations      every tensor still live at that point in the
                            schedule, which the simulation loop computes and
                            used to discard before the overflow scan began

Measured at trace extents over the 56 local containers, the live-activation half
alone moves five component/card pairs of 308: CogVideoX-2b and CogVideoX-5b-I2V's
vae (one op each, `aten.convolution::91`, 13,131 MB of footprint against a
13,736 MB line with 635 MB already resident), SANA-Video_2B_720p's vae (ten ops)
and Sana_1600M_4Kpx's vae on both card sizes. Every crossing is narrow, which is
the regime where the question about an empty card gives the wrong answer.

It changes NOTHING on real-esrgan-x8 at 1024x1024, the case that motivated the
tiling rung: there the footprints are 20-45 GB against the same 13.7 GB line and
clear it with or without the resident term. This gate is not that bug.

SHAPES: the fixture's sizes are the test. `A` is 8,192 MB and stays live across
`mul::1` because its consumer is the last op; `mul::1` itself outputs 6,144 MB
from a 64 MB input. Under the old formula `mul::1` costs 64 + 6,144 = 6,208 MB
and is nowhere near the 13,736 MB line -- it is not a marginal case that a
rounding change would flip. Under the corrected one it costs 8,192 + 6,144 =
14,336 MB and clears it. Any size where the two formulas agree would leave this
file green against the code it is meant to pin.

SEEN RED: with `op_footprint = largest_in_bytes + out_bytes_total + ws_bytes`
restored at profiler.py, `test_a_live_neighbour_puts_an_op_over_the_line` and
`test_the_weights_come_off_the_budget` both fail -- 0 overflow ops instead of 1.

Run: PYTHONPATH=src python -m pytest tests/unit/prism/test_an_op_is_budgeted_against_what_is_left.py
"""
from __future__ import annotations

import json

import pytest

from neurobrix.core.prism.profiler import ActivationProfiler

MB = 1024 * 1024
CARD = 16160 * MB          # a 16 GB V100, the class the rung was written for
SAFETY = 0.85              # threshold 13,736 MB


def _t(shape):
    return {"shape": list(shape), "dtype": "float16"}


def _graph():
    """seed -> A (8,192 MB) and B (6,144 MB); A survives until the last op.

    fp16, so elements are 2 bytes: 1024*2048*2048 elements is 8,192 MB and
    768*2048*2048 is 6,144 MB.
    """
    return {
        "version": "0.1",
        "tensors": {
            "input::seed": _t([1, 8, 2048, 2048]),          #    64 MB
            "aten.mul::0::out_0": _t([1, 1024, 2048, 2048]),  # 8,192 MB
            "aten.mul::1::out_0": _t([1, 768, 2048, 2048]),   # 6,144 MB
            "aten.add::2::out_0": _t([1, 8, 2048, 2048]),     #    64 MB
        },
        "ops": {
            "aten.mul::0": {"op_type": "aten::mul",
                            "input_tensor_ids": ["input::seed"],
                            "output_tensor_ids": ["aten.mul::0::out_0"]},
            # Its own inputs are small. Only the NEIGHBOUR still live on the
            # card -- A -- puts it over, which is the whole point.
            "aten.mul::1": {"op_type": "aten::mul",
                            "input_tensor_ids": ["input::seed"],
                            "output_tensor_ids": ["aten.mul::1::out_0"]},
            "aten.add::2": {"op_type": "aten::add",
                            "input_tensor_ids": ["aten.mul::0::out_0",
                                                 "aten.mul::1::out_0"],
                            "output_tensor_ids": ["aten.add::2::out_0"]},
        },
        "execution_order": ["aten.mul::0", "aten.mul::1", "aten.add::2"],
    }


def _profiler(tmp_path):
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(_graph()))
    return ActivationProfiler.from_path(p)


def _uids(profile):
    return {entry[0] for entry in (profile.overflow_ops or [])}


def test_the_live_set_at_each_op_is_recorded(tmp_path):
    """The number has to exist before anything can be budgeted against it."""
    ap = _profiler(tmp_path).estimate_peak_memory(
        dtype_bytes=2, vram_per_gpu_bytes=CARD, mode="compiled", safety=SAFETY)
    assert ap.live_before_op is not None
    assert ap.live_before_op["aten.mul::0"] == 0          # nothing yet
    assert ap.live_before_op["aten.mul::1"] == 8192 * MB  # A is on the card
    assert ap.live_before_op["aten.add::2"] == 14336 * MB  # A and B both


def test_a_live_neighbour_puts_an_op_over_the_line(tmp_path):
    """mul::1 costs 6,208 MB of its own and 14,336 MB on a card holding A."""
    ap = _profiler(tmp_path).estimate_peak_memory(
        dtype_bytes=2, vram_per_gpu_bytes=CARD, mode="compiled", safety=SAFETY)
    assert "aten.mul::1" in _uids(ap)
    # and mul::0, which runs on an empty card, must NOT be caught -- otherwise
    # this passes for a formula that simply flags everything.
    assert "aten.mul::0" not in _uids(ap)


def test_the_weights_come_off_the_budget(tmp_path):
    """Weights are resident for the component's whole execution.

    mul::0 outputs 8,192 MB on an otherwise empty card: under the 13,736 MB
    line, and it stays under however the live set is counted. Give the card
    6,000 MB of weights to hold and the line falls to 7,736 MB, which 8,192
    clears. Nothing but the weights term can move this one.
    """
    prof = _profiler(tmp_path)
    without = prof.estimate_peak_memory(
        dtype_bytes=2, vram_per_gpu_bytes=CARD, mode="compiled", safety=SAFETY)
    assert "aten.mul::0" not in _uids(without)

    with_weights = _profiler(tmp_path).estimate_peak_memory(
        dtype_bytes=2, vram_per_gpu_bytes=CARD, mode="compiled", safety=SAFETY,
        resident_bytes=6000 * MB)
    assert "aten.mul::0" in _uids(with_weights)


def test_the_budget_never_goes_negative(tmp_path):
    """Weights larger than the card is a placement defect, not a crash here."""
    ap = _profiler(tmp_path).estimate_peak_memory(
        dtype_bytes=2, vram_per_gpu_bytes=CARD, mode="compiled", safety=SAFETY,
        resident_bytes=CARD * 4)
    # Everything overflows a card already full, which is the honest answer.
    assert len(ap.overflow_ops) == 3
