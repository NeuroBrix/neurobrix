"""The weight reader's ceiling is a plan figure: the profile's threads, or fewer — the most reads in flight whose host
price still fits the host's fact (`load_ceiling.ceiling`, the launch-time cure 2026-11-19). It replaces the count a
profile kept from a trial ahead of the load: the price is the loader's own (`host_footprint` at k reads), the bound the
profile's installed RAM less the system's measured share, or a unified device's pool and the door's free memory.

What each test would do if the code were wrong: a ceiling past what the host holds (red), a ceiling lowered when the
price does not move with the count (red), the planning host's processors used where the profile states its threads (a
plan that is not the profile's: red), zero reads (red)."""
from types import SimpleNamespace

import pytest

from neurobrix.pytorch.prism import load_ceiling
from neurobrix.pytorch.prism.solver import PrismSolver

GB = 1 << 30


def _ceiling(profile, price):
    return load_ceiling.ceiling(profile, price, PrismSolver()._unified_host_pool_mb, PrismSolver._less_the_system_share)


def _profile(threads=40, ram_mb=64 * 1024, unified=False, share=None):
    return SimpleNamespace(cpu=SimpleNamespace(threads=threads, ram_mb=ram_mb, runtime_base_mb={}),
                           devices=[SimpleNamespace(has_unified_memory=unified, memory_mb=32768)],
                           host_measured={"system_share_mb": share} if share else {})


def _price(per_read_gb, fixed_gb=10):
    return lambda k: {"total_bytes": int((fixed_gb + per_read_gb * k) * GB), "base_bytes": 0, "resident_bytes": 0}


def test_a_price_that_does_not_move_takes_the_profile_s_threads():
    assert _ceiling(_profile(threads=40), _price(0)) == 40


def test_the_ceiling_is_the_most_reads_the_ram_holds():
    # 64 GB installed, 4 GB the system's share: 60 GB; 10 GB fixed + 2 GB a read -> 25 reads
    assert _ceiling(_profile(threads=40, share=4096), _price(2)) == 25


def test_not_even_one_read_fits_one_read_is_kept():
    assert _ceiling(_profile(threads=40, ram_mb=8 * 1024), _price(2)) == 1


def test_the_profile_s_threads_never_the_planning_host_s(monkeypatch):
    from neurobrix.core import workspace
    monkeypatch.setattr(workspace, "processors", lambda: 999)
    assert _ceiling(_profile(threads=12), _price(0.001)) == 12


def test_a_unified_pool_bounds_it(monkeypatch):
    monkeypatch.setattr(load_ceiling, "memory_state", lambda: SimpleNamespace(measured=False, available_mb=0))
    pool = lambda profile: 20 * 1024
    assert load_ceiling.ceiling(_profile(threads=12, unified=True), _price(1), pool, None) == 10
    monkeypatch.setattr(load_ceiling, "memory_state", lambda: SimpleNamespace(measured=True, available_mb=14 * 1024))
    assert load_ceiling.ceiling(_profile(threads=12, unified=True), _price(1), pool, None) == 4   # the door's free memory now
