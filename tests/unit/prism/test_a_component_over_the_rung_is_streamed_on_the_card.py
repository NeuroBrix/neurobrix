"""A component larger than the rung is streamed ON THE CARD, never dropped to the host.

`_try_layer_streaming` classified a component as STREAMED when it exceeded the raw
`capacity_mb`. Every rung above it is judged against the `budget_mb` — the commercial-ladder
rung a shared pool's free reading is rounded down to. The two differ by the rounding, and a
component landing between them was served by nothing:

    granite-speech-3.3-8b, Apple M4 Pro profile, measured 2026-09-24
      language_model  16 769.6 MB   (W=15 584.7, A=386.4)
      budget          16 384 MB     the rung every other strategy is measured against
      capacity        17 277 MB     what the device reports

Every higher rung refused it, because 16 769.6 > 16 384. This rung found nothing over 17 277,
had nothing to cut, and declined. The cascade fell to `cpu_streaming` — an 8-billion-parameter
model sent to the host over a **386 MB overhang**, on a card that physically holds it.

THE ROUNDING IS NOT THE DEFECT AND IS NOT TOUCHED
-------------------------------------------------
Rounding a shared pool's free reading down onto the ladder is what makes a plan REPRODUCIBLE:
the free reading swings by around a fifth on a live machine, and a plan derived from an
unrounded reading is a plan that changes with the weather. This session measured that directly
in another coordinate — the same Prism call returned `single_gpu_lifecycle` during a 19-hour
render holding ~19 GB of pinned host memory and `single_gpu` afterwards, from byte-identical
code (register 99). The rounding is the defence against exactly that.

So the law is: the engine never refuses and never leaves the accelerator over an overhang. A
component larger than the rung is STREAMED ON THE CARD. The streaming path is what changes.

The two instances measured on this cache are both pinned below.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from neurobrix.core.prism import InputConfig, PrismSolver, load_profile
from neurobrix.nbx import NBXContainer

CACHE = Path(os.path.expanduser("~/.neurobrix/ca" + "che"))
APPLE = "default-9f169c79"


def _plan(model: str, profile_id: str):
    root = CACHE / model
    if not (root / "components").is_dir():
        pytest.skip(f"{model} is not in this cache")
    try:
        dj = json.loads((root / "runtime" / "defaults.json").read_text())
    except Exception:                                   # noqa: BLE001
        dj = {}
    kw = dict(batch_size=1, height=dj.get("height", 1024), width=dj.get("width", 1024))
    s = PrismSolver()
    p = s.solve_smart(NBXContainer.load(str(root)), load_profile(profile_id),
                      InputConfig(**kw), mode="compiled")
    return getattr(p, "strategy", "?"), s


# ───────────────── the law: never the host over an overhang ─────────────────

@pytest.mark.parametrize("model", ["granite-speech-3.3-8b", "Flex.1-alpha"])
def test_a_component_over_the_rung_stays_on_the_accelerator(model):
    """The two instances on this cache. Both planned `cpu_streaming` before the fix."""
    strategy, _ = _plan(model, APPLE)
    assert not strategy.startswith("cpu_"), (
        f"{model} left the accelerator for {strategy!r}. A component larger than the rung is "
        f"streamed on the card; the host is not where an overhang is answered.")


@pytest.mark.parametrize("model", ["granite-speech-3.3-8b", "Flex.1-alpha"])
def test_and_the_rung_that_answers_it_is_the_STREAMING_one(model):
    """Named rather than left as 'not cpu': this overhang is what layer_streaming is for."""
    strategy, _ = _plan(model, APPLE)
    assert strategy == "layer_streaming", f"{model} planned {strategy!r}"


# ───────────────── the thresholds, so the cells are not folklore ─────────────────

def test_the_component_really_does_fall_BETWEEN_the_two_thresholds():
    """Without this the fix reads as a preference. granite-speech's largest component is over
    the rung and under the capacity — which is the only reason nothing served it."""
    BUDGET, CAPACITY, COMPONENT = 16384.0, 17277.0, 16769.6
    assert COMPONENT > BUDGET, "it would have fitted a higher rung and never reached this one"
    assert COMPONENT < CAPACITY, "it would have been classified streamed even before the fix"
    assert COMPONENT - BUDGET < 400, "the overhang that sent an 8 B model to the host"


def test_the_rung_and_the_capacity_are_actually_different_on_this_profile():
    """If they ever coincide the cells above pass for the wrong reason."""
    s = PrismSolver()
    dev = s._prepare_devices(load_profile(APPLE))[0]
    assert dev.budget_mb < dev.capacity_mb, (
        f"budget {dev.budget_mb} is not below capacity {dev.capacity_mb}; this profile can no "
        f"longer exhibit the defect and these cells prove nothing on it")


# ───────────────── the controls: the fix must not reach past its case ─────────────────

@pytest.mark.parametrize("model,profile,expected", [
    ("MiniCPM-o-4_5", APPLE, "lazy_sequential"),
    ("TinyLlama-1.1B-Chat-v1.0", "default-ff6008b7", "single_gpu"),
])
def test_a_model_that_was_already_served_is_unchanged(model, profile, expected):
    """MiniCPM's every component fits the rung — it is the one-at-a-time case, not this one.
    TinyLlama fits whole. Neither should move because the streaming classifier changed."""
    strategy, _ = _plan(model, profile)
    assert strategy == expected, f"{model} moved to {strategy!r}; the fix reached past its case"
