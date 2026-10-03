"""On a unified device the budget is the machine's free memory, less the run's own host side, rounded DOWN onto the
ladder — once.

The owner's rule: the budget is the free memory rounded down onto the standard ladder (13 GB free gives
12). Measured on the Mac (2026-10-03 22:39): 11 581 MB available, Prism planned against 10 899 and the
8 192 rung — a 0.95 safety factor (dd9acdc1, 2026-09-10) taken before the ladder law's rounding
(2026-09-20), a whole rung lost.

The Mac (23:00): rounding the whole free reading left ~300 MB of the pool beside the plan for the run's own host
side; what this process holds now and the engine's measured base (the profile's cpu.runtime_base_mb) are taken off
first — read, not a factor.

What would this file do if the code were wrong? The factor before the rounding -> 8 192, RED; the host side not
taken off -> the second test RED.
"""
from types import SimpleNamespace as NS
from pathlib import Path

import pytest

from neurobrix.core import host_memory as HM
from neurobrix.core.prism import solver as S
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, profile


@pytest.mark.parametrize("free,rung", [(11581, 11264), (13000, 12288), (10380, 8192)])
def test_the_free_reading_is_rounded_once(monkeypatch, free, rung):
    st = HM.MemoryState(available_mb=free, source="the Mac's reading")
    monkeypatch.setattr(S, "memory_state", lambda: st)
    monkeypatch.setattr(S, "_census_shadow_active", lambda: False)
    devs = S.PrismSolver()._prepare_devices(profile(APPLE_M4_PRO))
    assert devs[0].budget_mb == rung, (free, devs[0].budget_mb)


def test_the_engine_s_measured_base_comes_off_before_the_rounding(monkeypatch):
    """The process's own resident memory is already out of the reading (it is read inside the process);
    the engine's base — what its device work adds once it starts, measured into the profile — is not."""
    st = HM.MemoryState(available_mb=11581, source="the Mac's reading")
    monkeypatch.setattr(S, "memory_state", lambda: st)
    monkeypatch.setattr(S, "_census_shadow_active", lambda: False)
    import dataclasses
    base = profile(APPLE_M4_PRO)

    def with_base(mb):
        return dataclasses.replace(base, cpu=dataclasses.replace(base.cpu, runtime_base_mb=mb))
    solver = S.PrismSolver()
    solver._mode = "triton"
    assert solver._prepare_devices(with_base({"triton": 1500}))[0].budget_mb == 8192   # 11 581 - 1 500 -> 8 192
    assert solver._prepare_devices(with_base({}))[0].budget_mb == 11264                # none measured: as read
