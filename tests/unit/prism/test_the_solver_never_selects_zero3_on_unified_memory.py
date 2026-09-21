"""The solver never SELECTS zero3 where its premise is void.

test_zero3_frees_nothing_on_unified_memory (2026-09-09) settled the budget
half: on a unified device, zero3's "offload" moves nothing, and the budget
counts the weights. This test settles the selection half the Sana 4Kpx
compiled run exposed (2026-09-21): Strategy 3 still picked zero3 on mps —
a plan accepted under one memory model and executed under another, which
then died in zero3's CUDA machinery (`torch.cuda.set_device`) before a
single op ran. Selection must consult the same device door the budget does.
"""
from __future__ import annotations

from neurobrix.core.prism.loader import load_profile
from neurobrix.core.prism.solver import (
    ComponentMemory, DeviceState, PrismSolver,
)
from neurobrix.core.prism.structure import DeviceBrand, DeviceSpec


class _NoComponents:
    """Dtype door falls through to its default; the fgp probe reads an
    empty cache path and finds no block structure."""

    from pathlib import Path as _P
    cache_path = _P("/nonexistent-prism-fixture")

    def get_neural_components(self):
        return []


def _place(profile, device_string):
    solver = PrismSolver.__new__(PrismSolver)   # no machine probing
    from neurobrix.core.config.system import PRISM_DEFAULTS
    solver.oom_reserve_mb = PRISM_DEFAULTS["oom_reserve_mb"]
    spec = DeviceSpec(index=0, name="dev", memory_mb=24576,
                      compute_capability="0.0", supports_dtypes=["bfloat16"],
                      architecture="apple_silicon", brand=DeviceBrand.APPLE)
    dev = DeviceState(device_string=device_string, capacity_mb=17000.0,
                      spec=spec, recommended_mb=17000.0)
    mem = ComponentMemory(component_name="model",
                          weight_bytes=30 * 1024**3,      # never fits whole
                          activation_bytes=2 * 1024**3,   # zero3's own gate passes
                          overhead_bytes=0)
    return solver._place_component(
        _NoComponents(), "model", mem, [dev], {}, profile)


def test_zero3_is_not_selected_on_a_unified_device():
    profile = load_profile("default")
    profile.devices[0].unified_memory = True
    result = _place(profile, "mps:0")
    if result is not None:
        device_string, _ = result
        assert not str(device_string).startswith("zero3:"), (
            f"the solver placed a component on {device_string!r} on a UNIFIED "
            f"device — zero3's offload frees nothing there (the 2026-09-09 "
            f"budget finding) and its executor is CUDA machinery")


def test_zero3_stays_available_on_a_discrete_device():
    """The inertia proof: nothing moves for the discrete card."""
    profile = load_profile("a100-80g")
    result = _place(profile, "cuda:0")
    assert result is not None and str(result[0]).startswith("zero3:"), (
        f"expected zero3 on the discrete card, got {result!r} — the unified "
        f"guard must not reach machines whose premise holds")
