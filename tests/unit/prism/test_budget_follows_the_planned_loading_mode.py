"""A rung's budget is a function of the loading_mode its plan will carry — read, never assumed.

Two rungs reached the same fault from opposite directions. `_try_single_gpu` budgeted the
largest component with a comment saying one component is in VRAM at a time, while SINGLE_GPU is
eager and `unload_weights` returns without unloading. `_try_single_gpu_lifecycle` was budgeted on
a swap between persistent and transient components that the same eager class never performed.
Both accepted plans they could not run, and both are the same sentence: a budget written against
a residency the executor will not use.

So there is one rule and one place it is written. `planned_loading_mode` answers what the plan
WILL carry; `eager` gives sum(weights) + max(activation), `lazy` gives the per-component peak.
The serve-cold fallback needs no special case: it is simply a plan that will carry `lazy`.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism.solver import ComponentMemory, DeviceState, PrismSolver

MB = 1024 * 1024


def _mem(name, weight_mb, act_mb):
    return ComponentMemory(component_name=name, weight_bytes=weight_mb * MB,
                           activation_bytes=act_mb * MB, overhead_bytes=0)


def _comps():
    """Each fits alone on a 16 GB card; the three together do not."""
    return [("a", _mem("a", 6144, 1024)), ("b", _mem("b", 6144, 512)),
            ("c", _mem("c", 6144, 256))]


def _try(solver, comps, capacity_mb):
    return solver._try_single_gpu(comps, {n: m for n, m in comps},
                                  [DeviceState(device_string="cuda:0", capacity_mb=capacity_mb)],
                                  {}, None, None)


def test_the_rule_reads_the_mode_the_plan_will_carry():
    s = PrismSolver()
    assert s.planned_loading_mode("single_gpu") == "eager"
    assert s.planned_loading_mode("single_gpu_lifecycle") == "eager"
    assert s.planned_loading_mode("lazy_sequential") == "lazy"
    assert s.planned_loading_mode("cpu_streaming") == "lazy"
    s._serve_cold_fallback = True
    assert s.planned_loading_mode("single_gpu") == "lazy", (
        "serve degraded to cold forces lazy even on an eager-capable rung")


def test_an_eager_rung_refuses_what_it_cannot_keep_resident():
    s = PrismSolver()
    comps = _comps()
    assert max(m.weight_bytes + m.activation_bytes for _, m in comps) / MB < 12000
    assert _try(s, comps, 16000) is None
    assert _try(s, comps, 32000) is not None


def test_the_same_rung_accepts_it_when_the_plan_will_be_lazy():
    s = PrismSolver()
    s._serve_cold_fallback = True
    assert _try(s, _comps(), 16000) is not None, (
        "under a plan that will carry lazy, the per-component peak is the true budget")


def test_the_two_single_gpu_rungs_are_declared_eager():
    """The premise, read from the engine rather than assumed."""
    from neurobrix.core.prism.structure import AllocationStrategy
    assert AllocationStrategy.SINGLE_GPU.is_eager
    assert AllocationStrategy.SINGLE_GPU_LIFECYCLE.is_eager


def test_the_budget_and_the_plan_cannot_drift():
    """Whatever mode the rung budgeted against is the mode the plan carries, by construction:
    both call the same function. This is the property that was missing when one site said
    `serve_mode` and the other said `loading_mode`."""
    s = PrismSolver()
    for strategy in ("single_gpu", "single_gpu_lifecycle", "lazy_sequential", "cpu_streaming"):
        for fallback in (False, True):
            s._serve_cold_fallback = fallback
            assert s.planned_loading_mode(strategy) in ("eager", "lazy", "")
            if fallback:
                assert s.planned_loading_mode(strategy) == "lazy"
