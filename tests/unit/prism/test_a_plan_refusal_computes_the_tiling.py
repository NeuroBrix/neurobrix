"""A plan-stage refusal says how many tiles, not "a smaller input".

`_fail_error` has always ended with *"3. A smaller input (resolution, batch,
context)"*. That is true and it is not an answer — it names the remedy and
declines to take it, and it never says how much smaller.

The case that forced it is the Mac's, handed over 2026-09-18: `real-esrgan-x8` at
1024x1024 — the reproducer the adaptive-memory design is written against —
**refused at PLAN TIME in one second, before any allocation**, with 17237 MB
planned against 12598 MB available, of which 16.4 GB is activations and 32 MB is
weights. A controller hung on `DeviceOOMError` never fires there, because nothing
is ever allocated. And four tiles then rendered it by hand on that machine in
194 s, with an 8192x8192 artefact judged and its seam at +2.07 sigma.

The numbers below are that measurement, so this cell fails if the CUDA side stops
answering the Apple case.

Injection that turns these red: delete the `_reshape_block` computation from
`_fail_error`, or make `what_would_have_fit` return `[]`.
"""

from __future__ import annotations

import pytest

from neurobrix.core.prism import plan_advice as pa

_MB = 1024 * 1024
_GB = 1024 * _MB

# The Mac's reading, 2026-09-18.
APPLE_PLANNED_MB = 17237
APPLE_AVAILABLE_MB = 12598
APPLE_ACTIVATIONS = int(16.4 * _GB)
APPLE_WEIGHTS = 32 * _MB


def test_the_apple_refusal_becomes_a_tile_count():
    """17237 planned against 12598 available, and it must say a number."""
    overhead = APPLE_PLANNED_MB * _MB - APPLE_ACTIVATIONS - APPLE_WEIGHTS
    v = pa.assess("model", APPLE_WEIGHTS, APPLE_ACTIVATIONS, max(0, overhead),
                  APPLE_AVAILABLE_MB * _MB)
    assert v.reshapable, v
    assert v.bands is not None and v.bands >= 2, v
    # Every tile must fit the budget that was actually used.
    assert v.per_band_bytes is not None and v.per_band_bytes <= v.budget_bytes, v
    lines = pa.what_would_have_fit([v], spatial=True)
    assert lines and "tiles of about" in lines[0], lines
    assert "spatial" in lines[0], lines


def test_the_budget_carries_the_measured_reading_volatility():
    """Prism's own reading moved 12598 -> 15248 MB minutes apart on an idle machine.

    A band count divided straight out of the reading moves 21 % with it, so a plan
    that is exactly critical at one reading is short at the next. The budget is
    taken against the reading REDUCED by that swing, and this pins it — without it
    the margin could be silently removed and nothing would notice until a run.
    """
    avail = 12598 * _MB
    v = pa.assess("m", 32 * _MB, 16 * _GB, 64 * _MB, avail)
    assert v.budget_bytes == int(avail * (1.0 - pa.READING_VOLATILITY))
    assert v.budget_bytes < avail, "the budget is the reading itself — no margin"
    # And the margin is big enough to cover the measured swing between the two
    # readings, which is what it is for.
    assert (15248 - 12598) / 12598 <= pa.READING_VOLATILITY + 0.01


def test_weights_that_do_not_fit_are_a_different_refusal():
    """Every band pays the weights in full, so no band count reaches them.

    Saying "it fits in N tiles" there would be false, and saying "a smaller input"
    would send the reader to change the one thing that cannot help.
    """
    v = pa.assess("big", weight_bytes=20 * _GB, activation_bytes=1 * _GB,
                  overhead_bytes=0, available_bytes=12 * _GB)
    assert not v.reshapable
    assert v.bands is None
    assert v.refusal_reason and "no number of bands reaches it" in v.refusal_reason
    assert "tiles of about" not in " ".join(pa.what_would_have_fit([v]))


def test_a_component_that_already_fits_says_nothing():
    """Silence beats a sentence about a component that is not the problem."""
    v = pa.assess("small", 100 * _MB, 200 * _MB, 50 * _MB, 12 * _GB)
    assert v.bands == 1 and not v.reshapable
    assert pa.what_would_have_fit([v]) == []


def test_tiling_is_only_offered_where_activations_are_the_overflow():
    """A model whose WEIGHTS do not fit needs another rung, not this one."""
    assert pa.dominated_by_activations(APPLE_WEIGHTS, APPLE_ACTIVATIONS)
    assert not pa.dominated_by_activations(20 * _GB, 1 * _GB)
    # And the boundary is a class separation, not a tuned number: the Apple case
    # sits at a ratio of about 500 against a threshold of 4.
    assert APPLE_ACTIVATIONS / APPLE_WEIGHTS > 100


def test_the_side_scale_is_the_square_root_and_says_so():
    """Four tiles is each side at half — the figure the Mac's hand-run used."""
    assert pa.side_scale_for(4) == pytest.approx(0.5)
    assert pa.side_scale_for(1) == pytest.approx(1.0)
    assert pa.side_scale_for(9) == pytest.approx(1 / 3)
    doc = pa.side_scale_for.__doc__ or ""
    assert "spatial" in doc
    assert "sequence model" in doc, (
        "the assumption must be stated where the figure is produced")


def test_the_refusal_text_itself_carries_the_lines():
    """The wiring, not just the helper: `_fail_error` must put them in the message.

    A helper nothing calls is the failure mode this repository catalogues, so the
    cell reaches the real refusal rather than the function behind it.
    """
    from neurobrix.core.prism.solver import PrismSolver, ComponentMemory

    class _Dev:
        def __init__(self, mb):
            self.capacity_mb = mb
            self.device_string = "cuda:0"

    mem = ComponentMemory(component_name="model", weight_bytes=APPLE_WEIGHTS,
                          activation_bytes=APPLE_ACTIVATIONS,
                          overhead_bytes=200 * _MB)
    solver = PrismSolver.__new__(PrismSolver)
    with pytest.raises(RuntimeError) as e:
        solver._fail_error([("model", mem)], [_Dev(APPLE_AVAILABLE_MB)])
    text = str(e.value)
    assert "What tiling would do, computed rather than named" in text, text[-600:]
    assert "tiles of about" in text, text[-600:]
    assert "A smaller input" in text, "the original guidance must not be lost"
