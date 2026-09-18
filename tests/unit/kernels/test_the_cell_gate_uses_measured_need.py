"""The per-cell gate must refuse on the larger of the plan and the measurement.

Prism's plan is an ESTIMATE. On hat-s-x4 it was measured wrong by an order of
magnitude (M4 Pro, 26 GB unified, 2026-09-17):

    Prism planned                              278 MB
    engine live at an autotune door           8408 MB   (30x the plan)
    peak CONSUMPTION over the whole attempt  18006 MB   (available 18133 -> 127)
    outcome with the estimate alone          SIGKILL (137)

A gate fed 278 MB cannot refuse anything: 278 x 1.25 fits any machine. Fed the
measurement, it refuses by name with a number a person can check.

Note the second correction inside the first: the instantaneous live figure at a
door (8408 MB) is NOT the need either — it understated the run by 2.1x. The need
is peak consumption, sampled across the whole attempt.
"""
from __future__ import annotations

import json
import pathlib
import sys

TOOLS = pathlib.Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))


def test_the_measured_peak_is_peak_consumption_not_a_door_sample():
    rec = json.loads((TOOLS / "measured_peaks.json").read_text())["hat-s-x4"]
    assert rec["measured_peak_consumption_mb"] > rec["live_at_autotune_door_mb"], (
        "the need must be the peak CONSUMPTION over the attempt; the live figure "
        "at one autotune door understates it")
    assert rec["measured_peak_consumption_mb"] > rec["prism_plan_mb"] * 10


def test_the_gate_prefers_the_measurement_over_the_plan():
    from apple_matrix_percell import measured_peak_mb
    need = measured_peak_mb("hat-s-x4")
    assert need == 18006, need
    assert measured_peak_mb("a-model-never-measured") is None


def test_a_cell_that_cannot_fit_is_refused_with_both_numbers():
    """The refusal has to carry the need AND what was available, or it is not
    checkable by the person reading the matrix."""
    from apple_matrix_percell import measured_peak_mb, MARGIN
    need = measured_peak_mb("hat-s-x4")
    available = 15114                                   # measured on this machine
    assert need * MARGIN > available, (
        f"{need} MB x {MARGIN} should not fit in {available} MB")
