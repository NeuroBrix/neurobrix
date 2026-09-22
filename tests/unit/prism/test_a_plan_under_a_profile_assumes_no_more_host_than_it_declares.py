"""A plan taken under a PROFILE may not assume more host memory than that profile declares.

The cascade reads the PROFILE and not the card — that is the premise the census rests on,
and the reason another machine's plan is reproducible here at all. On a unified device the
budget was taken from the LIVE host reading with no bound, so the profile described one
machine and the budget described another.

MEASURED 2026-09-22, Flex.1-alpha under the Mac's own `default-9f169c79`
(`memory_mb: 18186`, `cpu.ram_mb: 24576`, `unified_memory: true`):

    device mps:0: capacity=17276.7  budget=131072.0

131 072 MB is the 128 GB rung of THIS rack's 251 GB host — a figure that appears nowhere in
the profile. Prism then took `single_gpu` for four components summing to **30 327 MB**
against a capacity of **17 277 MB**, and reported it as a plan. The same container under the
same profile on the machine the profile describes does not fit at all.

`_host_budget_mb` already applied exactly this bound — `min(r.free_mb, installed)` — so the
two halves of one law disagreed: the host budget was profile-bounded and the device budget
was not.

WHAT THIS DOES NOT CHANGE
-------------------------
A machine planning under its OWN profile: the declared figure and the live reading describe
the same machine, so the bound is inert. A BUSY machine still lowers below the declaration —
that is the 2026-09-10 repair for a render accepted at `single_gpu` and killed at step 3 of
20, and it is pinned below so this fix cannot undo it.
"""
from __future__ import annotations

import logging

import pytest

from neurobrix.core.host_memory import MemoryState
from neurobrix.core.prism import solver as _solver_mod
from neurobrix.core.prism.loader import load_profile
from neurobrix.core.prism.solver import PrismSolver

logging.disable(logging.WARNING)

APPLE = "default-9f169c79"          # the Mac's generated profile, carried here as a fixture


def _solver():
    s = PrismSolver.__new__(PrismSolver)
    s.safety_margin = 0.95
    return s


def _apple_profile():
    try:
        return load_profile(APPLE)
    except Exception:
        pytest.skip(f"{APPLE}.yml is not on this machine (it is gitignored)")


def _unified_device(profile, *, host_available_mb):
    """The prepared state of the profile's unified device under a given host reading."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_solver_mod, "memory_state",
                   lambda: MemoryState(total_mb=int(host_available_mb * 1.2),
                                       available_mb=host_available_mb, source="test"))
        mp.delenv("NBX_PRISM_BUDGET_MB", raising=False)
        want = profile.devices[0].get_device_string()
        devices = _solver()._prepare_devices(profile)
        # BY NAME, never by index: `_prepare_devices` sorts by capacity DESC, and on a rack
        # with four cards index 0 of the result is a different device (register entry 89).
        by_name = {d.device_string: d for d in devices}
        assert want in by_name, f"{want} missing from {sorted(by_name)}"
        return by_name[want]


def test_the_budget_never_exceeds_the_profiles_declared_host():
    """The defect itself: a 251 GB host must not lend its memory to a 24 GB profile."""
    profile = _apple_profile()
    declared = float(profile.cpu.ram_mb)
    d = _unified_device(profile, host_available_mb=251_000)
    assert d.budget_mb <= declared, (
        f"the plan assumed {d.budget_mb:.0f} MB on a profile that declares {declared:.0f} MB "
        f"of host memory — this rack's rung, not the profile's")


def test_the_budget_is_the_same_on_a_small_host_and_a_huge_one():
    """Reproducibility, stated as an equality: the same profile yields the same budget
    whatever machine reads it, which is what makes a foreign plan reproducible here."""
    profile = _apple_profile()
    big = _unified_device(profile, host_available_mb=251_000).budget_mb
    enough = _unified_device(profile, host_available_mb=int(profile.cpu.ram_mb)).budget_mb
    assert big == enough, (big, enough)


def test_a_busy_machine_still_lowers_below_the_declaration():
    """The 2026-09-10 repair, pinned so this fix cannot undo it: the bound is a CEILING,
    not a replacement. A machine with less free than the profile declares plans on less."""
    profile = _apple_profile()
    tight = _unified_device(profile, host_available_mb=6_000).budget_mb
    declared = _unified_device(profile, host_available_mb=int(profile.cpu.ram_mb)).budget_mb
    assert tight < declared, (tight, declared)


def test_the_bound_is_inert_for_a_machine_reading_its_own_profile():
    """Nothing moves where the declaration and the reading describe the same machine."""
    profile = _apple_profile()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_solver_mod, "memory_state",
                   lambda: MemoryState(total_mb=int(profile.cpu.ram_mb),
                                       available_mb=int(profile.cpu.ram_mb * 0.5),
                                       source="test"))
        mp.delenv("NBX_PRISM_BUDGET_MB", raising=False)
        s = _solver()
        dev = profile.devices[0]
        host = _solver_mod.memory_state()
        with_profile = s._device_reading(dev, dev.memory_mb * 0.95, host, profile).free_mb
        without = s._device_reading(dev, dev.memory_mb * 0.95, host, None).free_mb
        assert with_profile == without


def test_the_capacity_is_untouched():
    """Only the READING the law is applied to moves; `capacity_mb` is the 2026-09-10 path
    and keeps its own meaning."""
    profile = _apple_profile()
    d = _unified_device(profile, host_available_mb=251_000)
    assert d.capacity_mb == pytest.approx(profile.devices[0].memory_mb * 0.95, rel=1e-6)
