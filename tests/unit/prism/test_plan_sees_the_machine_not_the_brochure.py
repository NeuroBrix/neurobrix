"""A plan is sized on the pool the machine DECLARES; the moment's reading is the door.

The law since 2026-10-25 (the owner, 15:16): the plan is sized by the pool's DECLARED figure less the engine's
measured base — never by the moment's free reading; the reading is the EXECUTION DOOR (`execution_door`), which
refuses by name what the plan does not fit at run time, and a census rung imposed through NBX_PRISM_BUDGET_MB
replaces the pool. The scenarios below keep their machine and their readings; where a cell relied on the reading
to shrink the budget, the rung is now imposed through the door, the reading kept for what it is.

History (the rule this file first pinned, 2026-09-10): a plan sized on what the machine HAS, not on what the device
recommends.

`recommendedMaxWorkingSetSize` is a static property of the device. It does not
move when another process holds a third of the machine, and on unified memory
that is the same memory the plan will run in. Measured 2026-09-10 on this M4
Pro: 18 186 MB recommended, 10 099 MB actually free, a VM holding 8 615 MB and
5 566 MB of swap in use. An artefact of 12 298 MB was accepted at `single_gpu`
on the first number and killed by the system at step 3 of 20 — while its
largest component, 6 120 MB, would have fitted `layer_streaming` comfortably.

So the repair is not primarily a refusal: it is giving the CASCADE the true
number, so it descends by itself. The refusal is the last resort, for when no
rung fits, and it has to name all three figures because being killed at step 3
of 20 in silence teaches nothing.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/prism/test_plan_sees_the_machine_not_the_brochure.py -v
"""
from __future__ import annotations

import pytest

from neurobrix.core import host_memory
from neurobrix.core.host_memory import MemoryState, memory_state
from neurobrix.core.prism import solver as solver_mod
from neurobrix.core.prism.structure import DeviceBrand, DeviceSpec


def _spec(*, unified: bool, memory_mb: int = 18186) -> DeviceSpec:
    return DeviceSpec(
        index=0, name="test", memory_mb=memory_mb,
        compute_capability="apple-m4-pro" if unified else "8.0",
        supports_dtypes=["float32", "float16"],
        architecture="apple_silicon" if unified else "ampere",
        brand=DeviceBrand.APPLE if unified else DeviceBrand.NVIDIA,
        unified_memory=unified,
    )


class _Profile:
    def __init__(self, devices):
        self.devices = devices
        self.cpu = None


def _prepare(monkeypatch, spec, state):
    monkeypatch.setattr(solver_mod, "memory_state", lambda: state)
    inst = solver_mod.PrismSolver.__new__(solver_mod.PrismSolver)
    return solver_mod.PrismSolver._prepare_devices(inst, _Profile([spec]))


# ---------------------------------------------------------------------------
# The reading itself
# ---------------------------------------------------------------------------

def test_the_machine_is_read_and_the_reading_is_usable():
    state = memory_state()
    assert state.measured, f"could not read this machine: {state.source}"
    assert 0 < state.available_mb <= (state.total_mb or 10 ** 9)
    assert "machine memory:" in state.describe()


def test_the_reading_is_never_cached():
    """The whole point is that it moves. A cached availability would be a
    second `recommendedMaxWorkingSetSize` — the very defect it repairs."""
    first, second = memory_state(), memory_state()
    assert first is not second


def test_an_unreadable_platform_says_so_instead_of_inventing_a_number(monkeypatch):
    monkeypatch.setattr(host_memory.sys, "platform", "plan9")
    state = host_memory.memory_state()
    assert not state.measured
    assert state.available_mb is None
    assert "plan9" in state.source
    assert "NOT MEASURED" in state.describe()


# ---------------------------------------------------------------------------
# What the planner does with it
# ---------------------------------------------------------------------------

def test_a_unified_device_is_planned_against_its_declared_pool_and_keeps_the_reading_for_the_door(monkeypatch):
    """10 099 MB free of a 18 186 MB pool: the plan is sized on the pool (no engine base declared here), the
    reading stays on the device for the door. Before 2026-10-25 the capacity followed the reading (10 099)."""
    state = MemoryState(total_mb=24576, available_mb=10099, swap_used_mb=5566,
                        swap_total_mb=7168, source="test")
    devices = _prepare(monkeypatch, _spec(unified=True), state)
    assert devices[0].capacity_mb == 18186, devices[0].capacity_mb
    assert devices[0].recommended_mb == 18186, "the device's own figure is lost"
    assert devices[0].host_memory is not None and devices[0].host_memory.available_mb == 10099, "the door's figure is lost"


def test_a_discrete_device_is_not_constrained_by_host_pressure(monkeypatch):
    """On a discrete card the two pools are disjoint: a busy host does not
    shrink VRAM, and pretending it does would refuse plans that fit."""
    state = MemoryState(total_mb=24576, available_mb=512, source="test")
    devices = _prepare(monkeypatch, _spec(unified=False), state)
    assert devices[0].capacity_mb == 18186


def test_a_free_machine_leaves_the_device_figure_alone(monkeypatch):
    state = MemoryState(total_mb=65536, available_mb=60000, source="test")
    devices = _prepare(monkeypatch, _spec(unified=True), state)
    assert devices[0].capacity_mb == 18186, (
        "availability above the device budget must not raise it")


def test_an_unmeasurable_machine_does_not_silently_use_the_brochure(monkeypatch, caplog):
    state = MemoryState(source="no reader for platform 'plan9'")
    with caplog.at_level("WARNING"):
        devices = _prepare(monkeypatch, _spec(unified=True), state)
    assert devices[0].capacity_mb == 18186
    assert any("could not be measured" in r.getMessage()
               for r in caplog.records), (
        "falling back to the recommendation without saying so is the defect "
        "wearing a different hat")


def test_a_fresh_copy_keeps_both_numbers(monkeypatch):
    """Every strategy attempt copies the devices; a copy that dropped the two
    figures would leave the refusal with nothing to name."""
    state = MemoryState(total_mb=24576, available_mb=10099, source="test")
    devices = _prepare(monkeypatch, _spec(unified=True), state)
    inst = solver_mod.PrismSolver.__new__(solver_mod.PrismSolver)
    fresh = solver_mod.PrismSolver._fresh_devices(inst, devices)
    assert fresh[0].capacity_mb == 18186
    assert fresh[0].recommended_mb == 18186
    assert fresh[0].host_memory is devices[0].host_memory


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------

def test_the_refusal_names_the_device_figure_and_the_real_one(monkeypatch):
    state = MemoryState(total_mb=24576, available_mb=10099, swap_used_mb=5566,
                        swap_total_mb=7168,
                        largest_residents=(("prl_vm_app", 8615),), source="test")
    devices = _prepare(monkeypatch, _spec(unified=True), state)
    inst = solver_mod.PrismSolver.__new__(solver_mod.PrismSolver)
    verdict = solver_mod.PrismSolver._memory_verdict(inst, devices)
    assert "18186 MB" in verdict, verdict
    assert "10099 MB free" in verdict, verdict
    assert "prl_vm_app 8615 MB" in verdict, (
        "the refusal must name what is holding the memory, or the user cannot "
        "act on it")
    assert "swap 5566/7168 MB" in verdict


def test_the_verdict_is_readable_when_nothing_was_measured(monkeypatch):
    devices = _prepare(monkeypatch, _spec(unified=True), MemoryState(source="none"))
    inst = solver_mod.PrismSolver.__new__(solver_mod.PrismSolver)
    verdict = solver_mod.PrismSolver._memory_verdict(inst, devices)
    assert "NOT MEASURED" in verdict
