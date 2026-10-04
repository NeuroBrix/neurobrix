"""The post-scoring KV check counts what a plan keeps resident — per component, the KV estimate off the
LM's own cost, SUMMED — and judges it against the budget the plan runs under.

`PrismSolver._resident_bytes_for_kv_check` is the one place the arithmetic lives: a cost per
component (a streamed segment's peak; activations and overhead only where zero3 maps it to a
DISCRETE card's host; the whole component otherwise), the KV estimate taken off the named LM's own
cost, then the SUM. Two earlier forms were wrong in ways these cells pin: a max inside the loop
beside a `+=` (order-dependent), and the estimate subtracted from the combined figure (off a
figure that could be another component's, and off plans where no LM carried it).

WHY SUM, EVEN FOR `lazy_sequential` AND `cpu_streaming`. They LOAD one component at a time; what
stays resident is the FLOW's decision. The VLM decode loop keeps the LM and its head together, the
speech legs load their talker groups beside or after the LM, the triton dual_ar flow loads its
quantizer with the model still resident. A MAX (the strategies' docstrings) over-accepted MiniCPM on
the Mac (registers 104, 119). The SUM is the bound that holds for every flow that declares no
phases; a flow whose handlers were read into `core/flow/base.py RESIDENT_PHASES` is held to what it
keeps with the cache (`test_a_model_that_fits_whole_is_planned_whole.py`) — the vlm flow since
2026-10-04: each tower and projection alone, then the LM, its head and the speech leg together.

The machine is built (register 102): hand-built components where the answer is known exactly, a
pinned Mac or a pinned V100 for the plan cells.
"""
from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, V100_16GB, container_root, no_door,
                                              pin_dedicated_card, pin_host, profile)

MB = 1024 * 1024


def _mem(total_mb, act_mb=0.0, over_mb=0.0):
    return SimpleNamespace(total_bytes=int(total_mb * MB), activation_bytes=int(act_mb * MB),
                           overhead_bytes=int(over_mb * MB))


# a resident 10 GB component, a 6 GB one, and a 20 GB one whose weights zero3 offloads
COMPS = {"a": _mem(10_000), "b": _mem(6_000), "c": _mem(20_000, act_mb=500, over_mb=100)}
DISCRETE = {"a": ("cuda:0", {}), "b": ("cuda:0", {}), "c": ("zero3:cuda:0", {})}
SUM_DISCRETE = (10_000 + 6_000 + 600) * MB


def _solver(lm=None):
    s = PrismSolver()
    s._lm_component_name = lm
    return s


# ───────────────────── the arithmetic, where the answer is known ─────────────────────

@pytest.mark.parametrize("strategy", ["lazy_sequential", "cpu_streaming", "component_placement",
                                      "pipeline_parallel", "block_scatter", "weight_sharding"])
def test_every_strategy_is_the_SUM_of_what_its_components_keep(strategy):
    got = _solver()._resident_bytes_for_kv_check(strategy, DISCRETE, COMPS, profile(V100_16GB))
    assert got == SUM_DISCRETE, got / MB


def test_the_answer_does_not_depend_on_the_order_components_are_visited():
    s, prof, seen = _solver(), profile(V100_16GB), set()
    for order in itertools.permutations(COMPS):
        seen.add(s._resident_bytes_for_kv_check("lazy_sequential", DISCRETE,
                                                {k: COMPS[k] for k in order}, prof))
    assert seen == {SUM_DISCRETE}, sorted(x / MB for x in seen)


def test_zero3_on_unified_memory_frees_nothing():
    allocs = {"a": ("mps:0", {}), "b": ("mps:0", {}), "c": ("zero3:mps:0", {})}
    got = _solver()._resident_bytes_for_kv_check("lazy_sequential", allocs, COMPS, profile(APPLE_M4_PRO))
    assert got == (10_000 + 6_000 + 20_000) * MB, got / MB


def test_the_kv_estimate_comes_off_the_LMs_own_cost():
    """LM `b` carries a 7 000 MB estimate against a 6 000 MB cost: taken off `b` it floors at 0;
    taken off the combined figure it would also eat 1 000 MB of `a` and `c`."""
    got = _solver(lm="b")._resident_bytes_for_kv_check(
        "lazy_sequential", DISCRETE, COMPS, profile(V100_16GB), kv_already_counted=7_000 * MB)
    assert got == (10_000 + 0 + 600) * MB, got / MB


def test_no_named_LM_subtracts_nothing():
    got = _solver(lm=None)._resident_bytes_for_kv_check(
        "component_placement", DISCRETE, COMPS, profile(V100_16GB), kv_already_counted=1_000 * MB)
    assert got == SUM_DISCRETE, got / MB


def test_a_component_the_container_does_not_declare_has_no_dtype():
    """`_get_component_dtype` and the three placement paths that inlined it answered an
    undeclared name with an invented "bfloat16" — the cost multiplier of every whole-component
    decision then rested on a guess. It is refused."""
    c = SimpleNamespace(get_neural_components=lambda: [
        SimpleNamespace(name="a", get_dominant_dtype=lambda: "float16")])
    s = PrismSolver()
    assert s._get_component_dtype(c, "a") == "float16"
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        s._get_component_dtype(c, "not_declared")


# ───────────────────── MiniCPM on the Mac: the refusal is honest ─────────────────────

def _plan_mac(monkeypatch):
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    s = PrismSolver()
    seen = {}
    real = s._compute_memory

    def spy(*a, **k):
        out = real(*a, **k)
        seen.update(out)
        return out

    s._compute_memory = spy
    c = NBXContainer.load(str(container_root("MiniCPM-o-4_5")))
    try:
        p = s.solve_smart(c, profile(APPLE_M4_PRO), InputConfig(batch_size=1), mode="triton")
        refusal = None
    except RuntimeError as e:
        p, refusal = None, str(e)
    return p, refusal, s, seen, c


def test_the_decode_phase_holds_the_LM_its_head_and_the_speech_leg_and_no_tower(monkeypatch):
    """The measurement register 104 rests on, from the solver's own estimate and the flow's own
    lifecycle — re-read 2026-10-04 (register 119), then again when the vlm flow declared its phases.
    Both engines' handlers (core/flow/vlm.py, triton/flow/vlm.py, all three paths) run each tower and
    each projection once and unload it before the next stage unless the session is persistent — an
    EAGER served plan, which reads no phases — then decode with the LM and its head together, the
    CFM speech leg loading its talker groups beside the still-loaded LM. So the towers are never
    beside the LM: the SUM of every component (18 119 MB here) counted 1 872 MB no moment holds. The
    decode phase — the LM, its head, every speech component — is what the check holds, and the pair
    alone (the first form of this cell, 16 516 MB with a phantom transposed head) is under it."""
    _, _, s, seen, c = _plan_mac(monkeypatch)
    flow = s._flow_topology(c)["flow"]
    vlm, speech = flow["vlm"], set(flow["speech"]["components"].values())
    towers = {vlm[k] for k in ("vision_component", "vision_projection_component",
                               "audio_component", "audio_projection_component")}
    decode = {vlm["lm_component"], vlm["head_component"]} | speech
    phases = s._flow_phases(c)
    assert sorted(map(sorted, phases)) == sorted([[t] for t in towers] + [sorted(decode)]), phases
    assert set(seen) == towers | decode, ("every component in a phase", sorted(set(seen) ^ (towers | decode)))
    mb = lambda names: sum(seen[n].total_bytes for n in names) / MB
    rung = s._effective_capacity_mb(s._prepare_devices(profile(APPLE_M4_PRO))[0])
    pair = mb({vlm["lm_component"], vlm["head_component"]})
    assert pair < mb(decode) < rung < mb(seen), (pair, mb(decode), rung, mb(seen))
    peak = s._phase_peak(c, {n: m.total_bytes for n, m in seen.items()}) / MB
    assert peak == mb(decode), (peak, mb(decode))


def test_and_the_whole_strategies_are_refused_on_it_while_the_lm_streams(monkeypatch):
    """The idle Mac does not hold MiniCPM whole: its decode phase passes the KV check at the 16 384
    rung (2026-10-04, the vlm phases), but the plan's host side does not fit the memory free, and at
    the rung below the LM, owner of the cache, is streamed — the owner's rule (restated 2026-10-03
    22:35: the engine never refuses; the LM streams its layers inside the rung). Until
    a-streamed-plan-states-its-window this cell asserted the refusal 'No strategy can fit model + KV cache'."""
    p, refusal, s, _seen, c = _plan_mac(monkeypatch)
    assert refusal is None and p.strategy == "layer_streaming", (getattr(p, "strategy", None), (refusal or "")[:200])
    lm = ((s._flow_topology(c).get("flow") or {}).get("generation") or {}).get("lm_component")
    assert lm and list(p.layer_stream_plan) == [lm], (lm, sorted(p.layer_stream_plan))   # the flow's LM, alone


# ───────────────────── a host strategy is judged against the host ─────────────────────

def test_cpu_streaming_sizes_its_cache_against_the_host_it_runs_on(monkeypatch):
    """A HOST strategy's cache is budgeted against the host. It was judged against the GPUs'
    summed capacity. The scenario puts the two figures apart — the first version of this cell ran
    on the Mac, where host rung and GPU rung are the same 16 384 MB, and passed with the defect
    injected (register 103) — a dedicated V100-16GB (rung 15 564.8) beside a host with 22 000 MB
    free (rung 20 480), in serve mode, where the cache grows to whatever budget it is judged by.
    The held figure is what the check counts — cpu_streaming loads on demand, so the flow's dearest
    phase (the vlm decode phase since 2026-10-04; the SUM of every component before) — plus the cache."""
    no_door(monkeypatch)
    pin_host(monkeypatch, 257530, 22000, "a host with 22 000 MB free")
    pin_dedicated_card(monkeypatch, 16151, 306, "a dedicated V100-16GB")
    monkeypatch.setenv("NBX_FORCE_STRATEGY", "cpu_streaming")
    s = PrismSolver()
    c = NBXContainer.load(str(container_root("MiniCPM-o-4_5")))
    p = s.solve_smart(c, profile(V100_16GB), InputConfig(batch_size=1), serve_mode=True, mode="triton")
    assert p.strategy == "cpu_streaming", p.strategy
    host = s._host_budget_mb(profile(V100_16GB))
    gpu = s._effective_capacity_mb(s._prepare_devices(profile(V100_16GB))[0])
    lm, kv_est = s._lm_component_name, s._estimate_kv_cache_bytes(c, s._target_dtype_str)
    assert s._loads_on_demand(p.strategy) and s._flow_phases(c), "precondition: the check reads the phases"
    held = s._phase_peak(c, {n: m.total_bytes - (kv_est if n == lm else 0)
                             for n, m in p.component_memory.items()}) / MB
    kv = p.kv_cache_plan.memory_bytes / MB
    assert host > gpu, ("precondition: the host budget must exceed the GPU rung", host, gpu)
    assert held + kv <= host + 1e-6, (held, kv, host)
    assert held + kv > gpu, (f"the cache ({kv:.1f} MB beside {held:.1f} MB) fits the GPU rung "
                             f"{gpu:.1f}: it was sized against the GPU, not the {host:.1f} MB host")
