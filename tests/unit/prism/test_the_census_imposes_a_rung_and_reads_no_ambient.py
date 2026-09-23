"""On a unified device, the census budgets the plan by the imposed rung and
reads no ambient — proven end to end through the NBX_PRISM_BUDGET_MB door.

The census plan was non-deterministic across runs: on a unified device
`_prepare_devices` sized against `memory_state().available_mb` — live host
memory the census shadow does not neutralise — so the same model censused as
lazy_sequential one minute and layer_streaming the next. The tiling standard
forbids that: the census enumerates every rung of the ladder and plans each,
never the ambient, so the same model at the same rung yields the same plan and
the same keys.

The mechanism is the ONE door `NBX_PRISM_BUDGET_MB` (core/prism/memory_budget.py,
the Dell's f5a88bff) — this session's earlier NBX_CENSUS_RUNG_MB knob was a
second mechanism for the same function and was removed. `budget_mb` at the law
level is pinned by test_the_memory_budget_is_one_law...; here it is pinned END
TO END through `_prepare_devices` on a unified device, VARYING the ambient,
which is the census's own guarantee.
"""
from __future__ import annotations

import pytest

from neurobrix.core.host_memory import MemoryState
from neurobrix.core.prism import solver as _solver_mod
from neurobrix.core.prism.loader import load_profile
from neurobrix.core.prism.solver import PrismSolver


def _solver():
    s = PrismSolver.__new__(PrismSolver)
    s.safety_margin = 0.95
    return s


def _budget(monkeypatch, *, available_mb, door=None):
    """The unified device's planned BUDGET under a given host reading and an
    optional imposed rung (the door)."""
    monkeypatch.setattr(_solver_mod, "memory_state",
                        lambda: MemoryState(total_mb=24576,
                                            available_mb=available_mb,
                                            source="test"))
    if door is None:
        monkeypatch.delenv("NBX_PRISM_BUDGET_MB", raising=False)
    else:
        monkeypatch.setenv("NBX_PRISM_BUDGET_MB", str(door))
    profile = load_profile("default")
    profile.devices[0].unified_memory = True
    # Read the budget of the device this cell MADE unified, BY NAME.
    # `_prepare_devices` ends with `devices.sort(key=lambda d: (-d.capacity_mb, ...))`, so
    # `devices[0]` is the LARGEST card, not the one touched above. On the Mac that sort is a
    # no-op (one device) and the cell was right by accident; on this rack `default.yml`
    # carries four V100s — 2x16 GB and 2x32 GB — so `devices[0]` came back as cuda:2, a
    # 32 GB DISCRETE card, whose budget is 32 462 MB whatever the host reads. All three
    # cells here were measuring it: this one failed honestly (`32462.0 < 32462.0`) and the
    # two below PASSED while proving nothing about unified memory at all.
    # The engine was never wrong: on cuda:0, the unified one, the budget is 4 096 MB at
    # 6 000 MB free and 16 384 MB at 20 000 MB free — it tracks the ambient and lands on
    # the ladder, which is exactly the law.
    want = profile.devices[0].get_device_string()
    devices = _solver()._prepare_devices(profile)
    by_name = {d.device_string: d for d in devices}
    assert want in by_name, f"{want} is not in the prepared devices: {sorted(by_name)}"
    return float(by_name[want].budget_mb)


def test_without_the_door_the_unified_budget_tracks_the_ambient():
    """The ordinary run reads what is actually free — the behaviour the census
    must NOT inherit (and did, as the coin toss)."""
    with pytest.MonkeyPatch.context() as mp:
        tight = _budget(mp, available_mb=6000)
        loose = _budget(mp, available_mb=20000)
    assert tight < loose, (tight, loose)


def test_the_door_fixes_the_unified_budget_regardless_of_the_ambient():
    with pytest.MonkeyPatch.context() as mp:
        cap = _budget(mp, available_mb=6000, door=8192)
    assert cap == 8192, cap                # the rung, not the 6000 MB free


def test_the_same_rung_yields_the_same_budget_in_any_memory_weather():
    """The census stops being a coin toss: same rung, same plan, whatever the
    room reads — including when the machine's memory cannot be measured."""
    with pytest.MonkeyPatch.context() as mp:
        starved = _budget(mp, available_mb=3000, door=8192)
        idle = _budget(mp, available_mb=23000, door=8192)
        unmeasured = _budget(mp, available_mb=None, door=8192)
    assert starved == idle == unmeasured == 8192, (starved, idle, unmeasured)
