"""On a unified machine the pool a plan may fill is the device's declared working set, not the installed RAM: the
system keeps its own share of the RAM, and the device declares what its work may hold (the Mac M4 Pro, 24 576 MB
installed, 18 186 MB declared). Before: the pool was the installed RAM less the engine's base (24 403 MB), so
MiniCPM-o-4_5 under triton planned single_gpu at 20 852 MB of host side — over the device's declared 18 186 MB —
and the execution door refused it at every free reading this machine ever has (the Mac, 2026-10-26 10:41).
"""
from __future__ import annotations
from types import SimpleNamespace
from neurobrix.core.prism.solver import PrismSolver


def _solver(mode="triton"):
    s = PrismSolver.__new__(PrismSolver)
    s._mode = mode
    return s


def _profile(ram_mb, device_mb, base=None):
    dev = SimpleNamespace(memory_mb=device_mb, has_unified_memory=True)
    return SimpleNamespace(devices=[dev], cpu=SimpleNamespace(ram_mb=ram_mb, runtime_base_mb=base or {}))


def test_the_unified_pool_never_exceeds_the_devices_declared_working_set():
    pool = _solver()._unified_host_pool_mb(_profile(24576, 18186, {"triton": 173}))
    assert pool <= 18186, f"the pool {pool:.0f} MB is above the device's declared 18186 MB"
    assert pool == 18186 - 173, "the engine's measured base comes off the declared figure"


def test_an_installed_ram_below_the_declared_figure_is_the_bound():
    assert _solver()._unified_host_pool_mb(_profile(16384, 18186)) == 16384


def test_without_an_installed_figure_the_declared_one_is_the_pool():
    assert _solver()._unified_host_pool_mb(_profile(0, 18186)) == 18186
