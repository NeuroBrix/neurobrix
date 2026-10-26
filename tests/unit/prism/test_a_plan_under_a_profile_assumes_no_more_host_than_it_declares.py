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
the same machine, so the bound is inert. A BUSY machine lowers the door's free reading, never the budget (the owner, 2026-10-25 15:16),
and is re-planned on what is free by `solve_smart` (2026-10-26 19:21) — pinned below.
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


def test_a_busy_reading_lowers_the_door_never_the_budget():
    """Since 2026-10-25 15:16 the budget is the pool's fact and the moment's reading is the door: a machine with less
    free than the profile declares keeps its budget and reads less free; the re-plan on a busy machine is
    `solve_smart`'s (test_a_busy_machine_is_replanned_not_refused). Before, the reading lowered the budget itself
    (the 2026-09-10 repair this replaced)."""
    profile = _apple_profile()
    tight = _unified_device(profile, host_available_mb=6_000)
    declared = _unified_device(profile, host_available_mb=int(profile.cpu.ram_mb))
    assert tight.budget_mb == declared.budget_mb, (tight.budget_mb, declared.budget_mb)
    assert tight.live_free_mb < declared.live_free_mb, (tight.live_free_mb, declared.live_free_mb)


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


def test_the_capacity_is_the_declared_working_set_within_what_the_system_leaves():
    """`capacity_mb` is the device's declared working set, never above the installed RAM less the system's measured
    share (`host_measured.system_share_mb`, 2026-10-26) — the host pool's own figure; no 0.95 (34171d6c)."""
    profile = _apple_profile()
    d = _unified_device(profile, host_available_mb=251_000)
    share = float(((getattr(profile, "host_measured", None) or {}).get("system_share_mb")) or 0)
    want = min(float(profile.devices[0].memory_mb), float(profile.cpu.ram_mb) - share)
    assert d.capacity_mb == pytest.approx(want, rel=1e-6), (d.capacity_mb, want)


# ─────────── and never more than the DEVICE can hold ───────────
# The host bound above stops another machine's RAM leaking into the plan. It does not stop
# the profile's OWN RAM exceeding its own GPU: on unified memory the device pool is a SUBSET
# of machine RAM, so a reading taken from host memory has to be capped by the device's
# capacity too. Measured 2026-09-22 on the Mac's profile with only the host bound in place:
# capacity 17 276.7 MB, budget 24 576.0 — cpu.ram_mb straight through. A plan budgeted at
# 24 GB against a 17 GB device accepts plans that cannot run: the same failure as the
# 131 072 MB reading, just smaller and harder to notice.

def test_the_budget_never_exceeds_the_devices_own_capacity():
    profile = _apple_profile()
    d = _unified_device(profile, host_available_mb=251_000)
    assert d.budget_mb <= d.capacity_mb, (
        f"budget {d.budget_mb:.1f} exceeds the device's capacity {d.capacity_mb:.1f} — "
        f"a budget larger than the thing it budgets")


def test_both_bounds_are_needed_and_neither_alone_is_enough():
    """Host bound alone lets cpu.ram_mb through; capacity bound alone lets another
    machine's free RAM through while it is below capacity. Pinned together."""
    profile = _apple_profile()
    declared = float(profile.cpu.ram_mb)
    cap = float(profile.devices[0].memory_mb)
    assert declared > cap, (
        "this profile no longer has more system RAM than GPU capacity, so it cannot "
        "distinguish the two bounds — pick another fixture")
    d = _unified_device(profile, host_available_mb=251_000)
    assert d.budget_mb <= min(declared, cap)
