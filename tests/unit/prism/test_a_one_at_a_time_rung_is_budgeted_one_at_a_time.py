"""`lazy_sequential` holds ONE component at a time, and the capacity check summed them all.

The rung's own premise — its docstring says it "drops the requirement from sum(components) to
max(component)" — was contradicted by the check that decides whether it may run. Every
component's `total_bytes` went into `total_allocated`, `remaining` came out 0, the KV cache
could not fit in 0, and the strategy was rejected in favour of something further down:

    MiniCPM-o-4_5, Apple M4 Pro profile, 2026-09-23
      14 components, largest 14 019 MB, every one under the 16 384 MB rung; sum 19 519 MB
      against 17 277 MB of capacity.
      rejected lazy_sequential: "KV cache does not fit: ... Remaining VRAM: 0MB, need >=94"
      -> cpu_execution, off the accelerator, on a model whose every component fits it alone.

The residency arithmetic now lives in one method, `_resident_bytes_for_kv_check`: a cost per
component (streamed segment peak; activations only when zero3 maps it to a DISCRETE card's
host; the whole component otherwise), then MAX for lazy_sequential and SUM for every strategy
that holds its components together. The first repair did the max inside the loop beside a
`+=` for zero3 components, which made the answer depend on the order components were visited;
combining after every cost is known removes that.

THESE CELLS CONSTRUCT THEIR SCENARIO
------------------------------------
The arithmetic is tested on hand-built components, where the answer is known exactly. The
plan-level cells pin the machine they plan for (injected host reading; the rung through the
door, or an injected dedicated card reading), and a precondition cell proves the case really is
`max(component) <= capacity < sum(components)`.

RETIRED from the first version, with the reason
-----------------------------------------------
* `test_the_LADDER_case_is_untouched_and_still_leaves_the_accelerator` asserted granite-speech
  plans `cpu_streaming`. That question was decided by the streaming fix on main (69c98647): a
  component over the rung streams on the card. Its gate is
  `test_a_component_over_the_rung_is_streamed_on_the_card.py`.
* `test_the_max_rule_does_NOT_reach_a_rung_that_holds_everything` pinned four PixArt plans on the
  rack's live multi-GPU profile. Register 99 established that the five transitions it was built
  on were RACK STATE (a 19-hour render's pinned host memory), with 0 stable differences across 40
  plans under controlled alternation. It could not discriminate the injection it named. The
  over-application is now caught where it lives: the SUM cells below.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import neurobrix.core.host_memory as host_memory
import neurobrix.core.prism.solver as solver_mod
from neurobrix.core.host_memory import MemoryState
from neurobrix.core.prism import InputConfig, PrismSolver, load_profile
from neurobrix.core.prism.memory_budget import DeviceReading
from neurobrix.nbx import NBXContainer

CACHE = Path(os.path.expanduser("~/.neurobrix/ca" + "che"))
APPLE = "default-9f169c79"          # unified mps:0, memory_mb 18 186, ram 24 576
V100 = "default-ff6008b7"           # one discrete cuda:0, 16 384 MB
MB = 1024 * 1024


def _mem(total_mb, act_mb=0.0, over_mb=0.0):
    return SimpleNamespace(total_bytes=int(total_mb * MB), activation_bytes=int(act_mb * MB),
                           overhead_bytes=int(over_mb * MB))


# three components: a resident 10 GB one, a 6 GB one, and a 20 GB one whose weights zero3 offloads
COMPS = {"a": _mem(10_000), "b": _mem(6_000), "c": _mem(20_000, act_mb=500, over_mb=100)}
ALLOCS_DISCRETE = {"a": ("cuda:0", {}), "b": ("cuda:0", {}), "c": ("zero3:cuda:0", {})}


# ───────────────────── the arithmetic, on components whose answer is known ─────────────────────

def test_one_at_a_time_is_the_MAX_of_what_each_component_holds():
    s = PrismSolver()
    got = s._resident_bytes_for_kv_check("lazy_sequential", ALLOCS_DISCRETE, COMPS, load_profile(V100))
    assert got == 10_000 * MB, got / MB          # max(10 000, 6 000, 500 + 100)


@pytest.mark.parametrize("strategy", ["component_placement", "pipeline_parallel", "block_scatter",
                                      "component_placement_lazy", "weight_sharding"])
def test_every_strategy_that_holds_them_together_is_the_SUM(strategy):
    """The over-application direction: a max applied here would under-budget a strategy that
    really holds every component at once, and let it pass a check it should fail."""
    s = PrismSolver()
    got = s._resident_bytes_for_kv_check(strategy, ALLOCS_DISCRETE, COMPS, load_profile(V100))
    assert got == (10_000 + 6_000 + 600) * MB, got / MB


def test_the_answer_does_not_depend_on_the_order_components_are_visited():
    s = PrismSolver()
    prof = load_profile(V100)
    seen = set()
    for order in itertools.permutations(COMPS):
        comps = {k: COMPS[k] for k in order}
        seen.add(s._resident_bytes_for_kv_check("lazy_sequential", ALLOCS_DISCRETE, comps, prof))
    assert seen == {10_000 * MB}, sorted(x / MB for x in seen)


def test_zero3_on_unified_memory_frees_nothing():
    """On a unified device the "offloaded" weights stay in the same pool, so they count."""
    s = PrismSolver()
    allocs = {"a": ("mps:0", {}), "b": ("mps:0", {}), "c": ("zero3:mps:0", {})}
    got = s._resident_bytes_for_kv_check("lazy_sequential", allocs, COMPS, load_profile(APPLE))
    assert got == 20_000 * MB, got / MB


# ───────────────────── the plan: MiniCPM, on a pinned machine ─────────────────────

def _pin_apple(monkeypatch):
    st = MemoryState(total_mb=24576, available_mb=18186, source="injected: an idle Mac")
    monkeypatch.setattr(solver_mod, "memory_state", lambda: st)
    monkeypatch.setattr(host_memory, "memory_state", lambda: st)
    monkeypatch.setenv("NBX_PRISM_BUDGET_MB", "16384")


def _pin_v100(monkeypatch):
    big = MemoryState(total_mb=257530, available_mb=200000, source="injected: an idle host")
    monkeypatch.delenv("NBX_PRISM_BUDGET_MB", raising=False)
    monkeypatch.setattr(solver_mod, "memory_state", lambda: big)
    monkeypatch.setattr(host_memory, "memory_state", lambda: big)
    monkeypatch.setattr(solver_mod, "read_device_sharing", lambda i: DeviceReading(
        kind="device", capacity_mb=16151, free_mb=15845, own_context_mb=306, measured=True,
        source="injected: a dedicated V100-16GB"))


PINS = {APPLE: _pin_apple, V100: _pin_v100}


def _plan(monkeypatch, profile_id, refusal_ok=False):
    root = CACHE / "MiniCPM-o-4_5"
    if not (root / "components").is_dir():
        pytest.skip("MiniCPM-o-4_5 is not in this cache")
    PINS[profile_id](monkeypatch)
    dj_path = root / "runtime" / "defaults.json"
    dj = json.loads(dj_path.read_text()) if dj_path.is_file() else {}
    s = PrismSolver()
    seen = {}
    real = s._compute_memory

    def spy(*a, **k):
        out = real(*a, **k)
        seen.update(out)
        return out

    s._compute_memory = spy
    try:
        p = s.solve_smart(NBXContainer.load(str(root)), load_profile(profile_id),
                          InputConfig(batch_size=1, height=dj.get("height", 1024),
                                      width=dj.get("width", 1024)), mode="compiled")
    except RuntimeError:
        if not (refusal_ok and seen):
            raise
        p = None
    return p, s, seen


@pytest.mark.parametrize("profile_id", [APPLE, V100])
def test_the_case_is_max_under_the_capacity_and_sum_over_it(monkeypatch, profile_id):
    _, s, seen = _plan(monkeypatch, profile_id, refusal_ok=True)
    dev = s._prepare_devices(load_profile(profile_id))[0]
    sizes = [m.total_bytes / MB for m in seen.values()]
    cap = s._effective_capacity_mb(dev)
    assert max(sizes) <= cap < sum(sizes), (
        f"max {max(sizes):.1f} / rung {cap:.1f} / sum {sum(sizes):.1f}: not the one-at-a-time case")


@pytest.mark.parametrize("profile_id", [APPLE, V100])
def test_a_model_whose_every_component_fits_stays_on_the_accelerator(monkeypatch, profile_id):
    p, s, _ = _plan(monkeypatch, profile_id)
    bad = [r for r in getattr(s, "_rejected", [])
           if r[0] == "lazy_sequential" and "KV cache does not fit" in str(r[2])]
    assert not bad, f"lazy_sequential is still rejected on the KV check: {bad}"
    assert p.strategy == "lazy_sequential", p.strategy
