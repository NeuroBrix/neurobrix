"""`lazy_sequential` holds ONE component at a time, and the capacity check summed them all.

The rung's own premise — its docstring says it "drops the requirement from sum(components) to
max(component)" — was contradicted by the check that decides whether it may run. Every
component's `total_bytes` went into `total_allocated`, `remaining` came out 0, the KV cache
could not fit in 0, and the strategy was rejected in favour of something further down.

MEASURED 2026-09-23, and the census puts a bound on it
------------------------------------------------------
Over the whole cache — 59 containers x 3 profiles, 177 plans, none skipped — exactly **5** meet
`max(component) <= capacity < sum(components)`, which is the condition where this rung is the
right answer. Two of the five left the accelerator:

    MiniCPM-o-4_5          max= 14019  cap= 17277  sum= 19519  -> cpu_execution
    granite-speech-3.3-8b  max= 16770  cap= 17277  sum= 18082  -> cpu_streaming

They are NOT the same defect and only one is fixed here.

**MiniCPM** has 14 components and every one fits the 16 384 MB budget; the largest is 14 019.
`_try_lazy_sequential` returned a plan — the rung was viable — and the post-scoring check
rejected it:

    rejected lazy_sequential score=20.0
      KV cache does not fit: ZERO FALLBACK: KV cache budget insuffisant.
      Remaining VRAM: 0MB, need >=94

**granite-speech** is the ladder question instead: its largest component is 16 770 MB, over the
16 384 MB rung a shared pool is rounded down to, though under the 17 277 MB the card reports.
That is doctrine and is deliberately untouched, which is why it appears here as a control: a
fix that "helped" it too would have been reaching past its evidence.

THE COMMENT THAT WAS ALREADY THERE
----------------------------------
The branch being fixed already carried a note about this exact failure — *"Counting the
offloaded weights as resident rejected valid mixed plans (lazy_sequential mapping a 57GB LM to
zero3:cuda:0 was 'over capacity' on a 32GB card -> the KV check failed -> the cascade fell
through to cpu_execution on a GPU node)"* — and the repair was applied only to the
zero3-mapped case. A component that is not zero3-mapped still summed, so the same rung still
fell through the same hole for the same reason. Same defect, second doorway.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from neurobrix.core.prism import InputConfig, PrismSolver, load_profile
from neurobrix.nbx import NBXContainer

CACHE = Path(os.path.expanduser("~/.neurobrix/ca" + "che"))


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


# ───────────────────────── the defect, fixed ─────────────────────────

@pytest.mark.parametrize("profile_id", ["default-9f169c79", "default-ff6008b7"])
def test_a_model_whose_every_component_fits_stays_on_the_accelerator(profile_id):
    """MiniCPM: 14 components, largest 14 019 MB, all under budget; sum 19 519 over capacity."""
    strategy, _ = _plan("MiniCPM-o-4_5", profile_id)
    assert not strategy.startswith("cpu_"), (
        f"every component fits the accelerator alone and the plan left it: {strategy}")
    assert strategy == "lazy_sequential"


def test_the_rung_is_no_longer_rejected_for_a_KV_cache_that_had_no_room():
    """The rejection itself, pinned. `Remaining VRAM: 0MB` came from summing 14 components a
    one-at-a-time rung never holds together."""
    _, solver = _plan("MiniCPM-o-4_5", "default-9f169c79")
    bad = [r for r in getattr(solver, "_rejected", [])
           if r[0] == "lazy_sequential" and "KV cache does not fit" in str(r[2])]
    assert not bad, f"lazy_sequential is still rejected on the KV check: {bad}"


# ───────────────────── the control, deliberately NOT fixed ─────────────────────

def test_the_LADDER_case_is_untouched_and_still_leaves_the_accelerator():
    """granite-speech's largest component is over the rung a shared pool rounds down to, and
    under the capacity the card reports. That is the memory law working as written, and it is
    the owner's call, not a bug to reach for. If this cell ever goes green, the fix above grew
    past its evidence."""
    strategy, _ = _plan("granite-speech-3.3-8b", "default-9f169c79")
    assert strategy == "cpu_streaming", (
        "granite-speech now plans on the accelerator — either the ladder question was decided "
        "and this cell should be retired, or a fix reached past the case it was measured on")


# ───────── the OVER-application, which needs a multi-GPU profile to be visible ─────────

MULTI_GPU = "default-c5d28c27"          # 4 GPUs, 96 GB — the rack's own shape


@pytest.mark.parametrize("model,expected", [
    ("PixArt-XL-2-1024-MS", "single_gpu_lifecycle"),
    ("PixArt-Sigma-XL-2-1024-MS", "single_gpu_lifecycle"),
])
def test_the_max_rule_does_NOT_reach_a_rung_that_holds_everything(model, expected):
    """The other direction of the injection, and it took a measurement to make visible.

    Budgeting `lazy_sequential` at max(component) is correct because it holds one component at
    a time. Applying the same max to EVERY rung under-budgets the strategies that genuinely hold
    all components at once, and they then pass a capacity check they should fail.

    **On a single-device profile that is invisible**, and the scoring table says why: the
    strategies reaching the summing branch there are `op_level_tiling` (60), `layer_streaming`
    (50, which has its own partition exception) and `cpu_streaming` (5) — all BELOW
    `lazy_sequential` (300), which now passes anyway. Mis-budgeting them changes no chosen plan
    at any rung. A 1 062-plan sweep across three solver states and three imposed rungs on the
    two single-device profiles found nothing, because there was nothing there to find.

    On a MULTI-GPU profile `pipeline_parallel` (850), `component_placement` (750),
    `block_scatter` (700) and `component_placement_lazy` (400) reach the same branch, and the
    first three outscore `lazy_sequential`. Measured on `default-c5d28c27`, 49 plans compared,
    FIVE change under the over-applied max:

        GLM-4.1V-9B-Thinking       component_placement_lazy -> single_gpu
        PixArt-Sigma-XL-1024       single_gpu_lifecycle     -> component_placement_lazy
        PixArt-Sigma-XL-2-1024-MS  single_gpu_lifecycle     -> component_placement_lazy
        PixArt-XL-1024             single_gpu_lifecycle     -> component_placement_lazy
        PixArt-XL-2-1024-MS        single_gpu_lifecycle     -> component_placement_lazy

    The four PixArt rows are pinned here: same profile, same transition, four independent
    containers, so one container being retraced cannot quietly retire the cell. GLM's move to
    `single_gpu` is the starker harm — the plan comes to believe ONE GPU holds everything — but
    it is a single row and is recorded in prose rather than asserted.
    """
    strategy, _ = _plan(model, MULTI_GPU)
    assert strategy == expected, (
        f"{model} on {MULTI_GPU} planned {strategy!r}, expected {expected!r}. If this changed "
        f"because the one-at-a-time budget was applied to rungs that hold every component at "
        f"once, those rungs are now under-budgeted and may accept plans that do not fit.")


# ───────────────────── the arithmetic, so the cells are not folklore ─────────────────────

def test_the_two_cases_really_are_different_arithmetic():
    """Both meet max <= capacity < sum. Only one has every component under the BUDGET."""
    BUDGET, CAP = 16384.0, 17277.0
    mini_max, mini_sum = 14019.0, 19519.0
    gran_max, gran_sum = 16770.0, 18082.0
    for mx, sm in ((mini_max, mini_sum), (gran_max, gran_sum)):
        assert mx <= CAP < sm                       # both meet the owner's condition
    assert mini_max <= BUDGET                       # MiniCPM fits the rung
    assert gran_max > BUDGET                        # granite-speech does not
