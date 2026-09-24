"""On a unified device, Prism never places a component on the HOST to save
memory — because host and device are the same bytes, so the offload frees
nothing (the same finding as zero3-on-unified, cafaf799).

Strategy 4 in `_place_component` is the last-resort CPU placement for a
component whose activations fit no GPU even with zero3. On a discrete card it
is legitimate: the component's weights and compute move to host RAM, off the
device. On a UNIFIED device it saves nothing and it breaks two things — the
NBX GPU-only kernels cannot run on a cpu-placed component (the census shadow
failed on CogVideoX/PixArt's text_encoder for exactly this, 2026-09-21), and
the plan is accepted under one memory model and executed under another. On
unified the honest outcomes are TILE (if spatial) or REFUSE; never a host
offload for memory.

Discrete cards are byte-unchanged: the inertness arm asserts Strategy 4 still
places on CPU on an a100 profile.
"""
from __future__ import annotations

import pytest

from tests.unit.prism._pinned_machine import APPLE_M4_PRO, profile as build_profile   # built, never a machine's own profile (register 102)
from neurobrix.core.prism.solver import ComponentMemory, DeviceState, PrismSolver
from neurobrix.core.prism.structure import DeviceBrand, DeviceSpec


class _NoComponents:
    from pathlib import Path as _P
    cache_path = _P("/nonexistent-prism-fixture")

    # The components this fixture places, DECLARED, with the dtype the cells assumed. It returned
    # [] and the solver answered every dtype question with an invented "bfloat16"; that default
    # is now a refusal (`_get_component_dtype`), so the fixture says what it holds.
    def get_neural_components(self):
        from types import SimpleNamespace
        return [SimpleNamespace(name=n, get_dominant_dtype=lambda: "bfloat16")
                for n in ['model']]


def _place(unified: bool):
    """Reproduce the CogVideoX condition: a GPU whose LIVE-free reading is
    ~10.7 GB (VM up) with the profile's 24 GB CPU RAM. A component whose
    activation exceeds the live GPU but whose total fits 70% of CPU RAM is
    exactly what Strategy 4 offloaded to the host."""
    profile = build_profile(APPLE_M4_PRO)            # apple, has a 24 GB cpu section
    profile.devices[0].unified_memory = unified
    solver = PrismSolver.__new__(PrismSolver)
    from neurobrix.core.config.system import PRISM_DEFAULTS
    solver.oom_reserve_mb = PRISM_DEFAULTS["oom_reserve_mb"]
    solver._component_tiling = {}
    spec = DeviceSpec(index=0, name="dev", memory_mb=24576,
                      compute_capability="0.0", supports_dtypes=["bfloat16"],
                      architecture="apple_silicon", brand=DeviceBrand.APPLE)
    dev = DeviceState(device_string="mps:0" if unified else "cuda:0",
                      capacity_mb=10700.0, spec=spec, recommended_mb=10700.0)
    # weight 2 GB + activation 12 GB: activation 12 > 10.7 GPU (zero3 fails),
    # total 14 GB < 0.7*24 = 16.8 GB CPU RAM (Strategy 4 fits the host).
    mem = ComponentMemory(component_name="model",
                          weight_bytes=2 * 1024**3,
                          activation_bytes=12 * 1024**3,
                          overhead_bytes=0)
    return solver._place_component(
        _NoComponents(), "model", mem, [dev], {}, profile)


def test_no_host_offload_for_memory_on_a_unified_device():
    result = _place(unified=True)
    # On unified, a component that fits no GPU must NOT be placed on the host
    # to save memory: the outcome is a refusal (None) or a non-cpu placement,
    # never ("cpu", ...).
    if result is not None:
        device_string, _ = result
        assert not str(device_string).startswith("cpu"), (
            f"Prism placed a component on the HOST ({device_string!r}) on a "
            f"UNIFIED device to save memory — it frees nothing (host==device) "
            f"and breaks the GPU-only kernels; tile or refuse instead")


def test_host_offload_stays_available_on_a_discrete_device(monkeypatch):
    """Inertness: the discrete card still offloads to host as before. The
    host-fit check reads live host memory (memory_budget._host_budget_mb) — pin
    it large so this proves the guard's inertness, not the machine's free RAM."""
    from neurobrix.core import host_memory as _hm
    from neurobrix.core.host_memory import MemoryState
    monkeypatch.setattr(_hm, "memory_state",
                        lambda: MemoryState(total_mb=65536, available_mb=60000,
                                            source="test"))
    result = _place(unified=False)
    assert result is not None and str(result[0]).startswith("cpu"), (
        f"expected the host offload on the discrete card, got {result!r} — "
        f"the unified guard must not reach a machine where the offload frees "
        f"real device memory")
