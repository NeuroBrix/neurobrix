"""Prism grants a zero3 component the room its plan leaves on the card, and nothing more.

`PrismSolver._zero3_resident_budgets` sets `ComponentAllocation.resident_weight_mb`: the usable part
of the card (`_usable_mb`) less what the device holds while the component is loaded — each other
component of its phases at its plan cost, a zero3 component at its activations and overhead, the
KV cache as PLANNED in place of the estimate priced inside the language model — less the streaming
window (two of its largest blocks). The figure is the plan's arithmetic, so it is pinned here where
the answer is known exactly; the ratchet that spends it is pinned in
`tests/unit/strategies/test_zero3_keeps_the_blocks_its_plan_leaves_room_for.py`.

Janus-Pro-7B on a 16 GB V100 (2026-10-09): 9 245 MB of the 12 380 MB language model granted, 21 of
30 blocks and the non-block weights kept, 9 streamed per token instead of 30.
"""
from __future__ import annotations

from types import SimpleNamespace

from neurobrix.core.prism.solver import ComponentAllocation, ComponentMemory, PrismSolver

MB = 1024 * 1024


def _mem(name, w, a, o):
    return ComponentMemory(component_name=name, weight_bytes=int(w * MB), activation_bytes=int(a * MB),
                           overhead_bytes=int(o * MB))


def _alloc(name, strategy, device="cuda:0"):
    return ComponentAllocation(name=name, devices=[device], dtype="float16", memory_mb=0.0,
                               architecture="volta", vendor="nvidia", strategy=strategy)


def _solver(usable_mb=13_000.0, kv_estimate_mb=600.0, block_mb=400.0, unified=False):
    s = PrismSolver()
    s._mode = "compiled"
    s._lm_component_name = "lm"
    s._target_dtype_str = "float16"
    s._estimate_kv_cache_bytes = lambda c, d: int(kv_estimate_mb * MB)
    s._parse_blocks = lambda c, n: {"block_sizes": {0: block_mb, 1: block_mb, 2: block_mb - 100}}
    s._usable_mb = lambda dev: usable_mb
    s._get_component_dtype = lambda c, n: "float16"
    s._loads_on_demand = lambda strat: False          # the SUM: no phases
    profile = SimpleNamespace(devices=[SimpleNamespace(index=0, has_unified_memory=unified)])
    dev = SimpleNamespace(device_string="cuda:0", get_cost_multiplier=lambda d: 1.0)
    return s, profile, [dev]


def _plan(kv_planned_mb=500.0):
    return SimpleNamespace(
        strategy="lazy_sequential",
        components={"lm": _alloc("lm", "zero3"), "head": _alloc("head", "single_gpu")},
        component_memory={"lm": _mem("lm", 12_000, 2_000, 700), "head": _mem("head", 300, 50, 10)},
        kv_cache_plan=SimpleNamespace(memory_bytes=int(kv_planned_mb * MB)))


def test_the_grant_is_the_usable_room_less_what_is_held_less_the_window():
    s, prof, devs = _solver()
    plan = _plan()
    s._zero3_resident_budgets(None, plan, devs, prof)
    held = (2_000 + 700 - 600 + 500) + (300 + 50 + 10)   # lm live (estimate out, plan in) + head whole
    window = 2 * 400
    assert plan.components["lm"].resident_weight_mb == 13_000 - held - window
    assert plan.components["head"].resident_weight_mb == 0.0


def test_no_room_grants_nothing():
    s, prof, devs = _solver(usable_mb=3_000.0)
    plan = _plan()
    s._zero3_resident_budgets(None, plan, devs, prof)
    assert plan.components["lm"].resident_weight_mb == 0.0


def test_the_grant_never_exceeds_the_weights():
    s, prof, devs = _solver(usable_mb=100_000.0)
    plan = _plan()
    s._zero3_resident_budgets(None, plan, devs, prof)
    assert plan.components["lm"].resident_weight_mb == 12_000


def test_a_unified_device_is_granted_nothing():
    s, prof, devs = _solver(unified=True)
    plan = _plan()
    s._zero3_resident_budgets(None, plan, devs, prof)
    assert plan.components["lm"].resident_weight_mb == 0.0


def test_a_plan_with_no_zero3_component_is_left_untouched():
    s, prof, devs = _solver()
    plan = _plan()
    plan.components["lm"] = _alloc("lm", "single_gpu")
    s._zero3_resident_budgets(None, plan, devs, prof)
    assert all(a.resident_weight_mb == 0.0 for a in plan.components.values())

