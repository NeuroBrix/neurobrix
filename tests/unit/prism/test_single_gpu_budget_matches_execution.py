"""The single-GPU rung budgets a plan under the memory model it will be EXECUTED under.

SINGLE_GPU is an eager strategy: the plan it produces carries `loading_mode="eager"` and
`SingleGPUStrategy.unload_weights` returns without unloading, so every component's weights
stay resident for the whole request. The rung's cold branch used to budget the largest
single component, on a premise the executor never honoured — a model whose weights SUM past
the card while its largest component fits was accepted here and OOMed later, at whichever
allocation came next rather than at the decision that caused it.

The one path where an eager-capable rung really does unload is the serve-cold fallback,
which forces `loading_mode="lazy"`; there the per-component budget is the true one.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism.solver import ComponentMemory, DeviceState, PrismSolver

MB = 1024 * 1024


def _mem(name: str, weight_mb: int, act_mb: int) -> ComponentMemory:
    return ComponentMemory(component_name=name, weight_bytes=weight_mb * MB,
                           activation_bytes=act_mb * MB, overhead_bytes=0)


def _comps():
    """Three components that each fit alone and cannot be resident together:
    6+6+6 GB of weights on a 16 GB card, largest single component 7 GB."""
    return [("a", _mem("a", 6144, 1024)), ("b", _mem("b", 6144, 512)),
            ("c", _mem("c", 6144, 256))]


def _devices(capacity_mb: float):
    return [DeviceState(device_string="cuda:0", capacity_mb=capacity_mb)]


def _try(solver, comps, capacity_mb):
    return solver._try_single_gpu(comps, {n: m for n, m in comps},
                                  _devices(capacity_mb), {}, None, None)


def test_the_eager_rung_refuses_what_it_cannot_keep_resident():
    solver = PrismSolver()
    comps = _comps()
    # Each component fits with room to spare; the three together do not.
    assert max(m.weight_bytes + m.activation_bytes for _, m in comps) / MB < 12000
    assert _try(solver, comps, 16000) is None, (
        "accepted a plan whose weights sum past the card — the executor never unloads")


def test_the_eager_rung_accepts_what_fits_resident():
    solver = PrismSolver()
    comps = _comps()
    got = _try(solver, comps, 32000)
    assert got is not None
    allocations, _ = got
    assert set(allocations) == {"a", "b", "c"}


def test_the_serve_cold_fallback_budgets_one_component_at_a_time():
    """There, and only there, the plan carries loading_mode=lazy and the flow handler
    unloads between components — so the per-component budget is the true one."""
    solver = PrismSolver()
    solver._serve_cold_fallback = True
    assert _try(solver, _comps(), 16000) is not None


def test_single_gpu_is_declared_eager():
    """The premise the budget rests on, read from the engine rather than assumed."""
    from neurobrix.core.prism.structure import AllocationStrategy
    assert AllocationStrategy.SINGLE_GPU.is_eager
