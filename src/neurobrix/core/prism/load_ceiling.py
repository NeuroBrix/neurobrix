"""The weight reader's ceiling under a plan: how many reads a load may keep in flight, priced against the host.

The reader (`core.workspace.weight_reader`) grows its parallel reads by its own measure up to this ceiling; the plan
prices the host memory of a load at that many reads (`host_footprint`), so the ceiling is a plan figure and never the
planning machine's. No count is stored: the reader measures itself on each load (the launch-time cure, 2026-11-19).
"""
from __future__ import annotations

from typing import Callable

from nbxtensor.backend.host_memory import memory_state

_MB = 1 << 20


def _unified_need(pool_mb: float) -> Callable:
    """`need(fp)` on a unified device, to be judged against a room of 0: how far a footprint `fp` overruns the pool a plan
    may fill (`pool_mb`, the footprint less the engine base: the pool's loop) and the execution door's reading of the
    memory free now (the footprint less what this process holds), the larger of the two."""
    pool, host = pool_mb * _MB, memory_state()
    free = float(host.available_mb) * _MB if host.measured else None

    def need(fp):
        total = int(fp.get("total_bytes", 0))
        over_pool = total - int(fp.get("base_bytes", 0)) - pool
        return over_pool if free is None else max(over_pool, total - int(fp.get("resident_bytes", 0)) - free)
    return need


def _total_need(fp: dict) -> int:
    """`need(fp)` on a discrete-memory host, judged against the installed RAM less the system's share: the whole
    footprint."""
    return int(fp.get("total_bytes", 0))


def ceiling(profile, price: Callable, unified_pool_mb: Callable, less_the_system_share: Callable) -> int:
    """The weight reader's ceiling under this plan: the machine's threads as its profile states them (`cpu.threads`,
    the most reads and conversions that can work at once; the profile's, so a plan is the profile's and never the
    planning host's: `core.workspace.processors` only when the profile states no CPU), or fewer: the most whose host
    price (`price(k)`, the footprint at k reads in flight) still fits the host's fact: on a unified device the pool a
    plan may fill (`unified_pool_mb(profile)`, against the footprint less the engine base, as the pool's loop judges
    it) and the execution door's reading of the memory free now, else the installed RAM less the system's measured
    share (`less_the_system_share(profile, installed)`). One read when not even one fits (the plan's own doors then say
    it). A price that does not move with the count (nothing loaded per tensor) takes the threads."""
    from neurobrix.core.workspace import processors
    cpu = getattr(profile, "cpu", None)
    top = int(cpu.threads) if cpu is not None and getattr(cpu, "threads", None) else processors()
    at_top = price(top)
    if top == 1 or price(1).get("total_bytes") == at_top.get("total_bytes"):
        return top
    if any(getattr(d, "has_unified_memory", False) for d in profile.devices):
        room, need = 0, _unified_need(unified_pool_mb(profile))
    else:
        installed = float(getattr(cpu, "ram_mb", 0) or 0)
        if installed <= 0:
            return top               # no host fact to price against: the processors bound it, the price says it
        room, need = less_the_system_share(profile, installed) * _MB, _total_need
    return _most_that_fits(top, lambda k: need(price(k)) <= room)


def _most_that_fits(top: int, fits: Callable) -> int:
    """The most reads in 1..top that fit (`fits(k)` true for a count and every one below it): `top` when it fits, 1 when
    not even one does."""
    if fits(top):
        return top
    lo, hi = 1, top                  # fits(lo), not fits(hi); or lo cannot fit at all
    if not fits(lo):
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo


def settle(plan, profile, price: Callable, unified_pool_mb: Callable, less_the_system_share: Callable) -> None:
    """Prices `plan`'s host footprint at the ceiling of its reads: `plan.load_ceiling` is `ceiling(...)`, `plan.
    host_footprint` is `price` at it, and the loaders are told the ceiling (`core.workspace.declare_load_ceiling`), so
    the reader never grows past what the plan priced."""
    from neurobrix.core.workspace import declare_load_ceiling
    plan.load_ceiling = ceiling(profile, price, unified_pool_mb, less_the_system_share)
    plan.host_footprint = price(plan.load_ceiling)
    declare_load_ceiling(plan.load_ceiling)
