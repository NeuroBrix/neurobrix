"""On a unified device the cascade routes a too-big component to layer_streaming,
never to a host offload.

The owner's finding (2026-09-21): on a unified device Prism chose
`Strategy: lazy_sequential` and sent `text_encoder → cpu`. On unified, host and
device are the same bytes, so that offload frees nothing and it breaks the
GPU-only kernels. NeuroBrix never refuses for memory either — so the honest
destination for a component that fits no GPU whole is `layer_streaming`, which
holds one SEGMENT of the component at a time, reading it block by block from
storage (`manages_weight_residency=True`).

This is proven at the cascade level in two halves, because the routing is the
product of both:

1. THE GUARD removes the host offer. `_try_lazy_sequential` (and the other
   whole-component strategies through `_place_component`) can no longer place a
   too-big component on the host on a unified device — it returns None, so the
   strategy produces no candidate. On a DISCRETE device the same call still
   offloads to the host (inertness), because there the offload frees real device
   memory.

2. THE SCORE and ORDER make layer_streaming the destination among the survivors:
   it is tried before, and scores above, the host rungs (`cpu_execution`,
   `cpu_streaming`) — so once the whole-component host offloads are gone,
   layer_streaming (accelerator, streamed from storage) wins over leaving the
   accelerator.

Red then green: with the unified guard neutralised, half 1's unified assertion
fails — lazy_sequential offers a cpu placement (score 300) that would outrank
layer_streaming (50) and win.
"""
from __future__ import annotations

import ast
import inspect

import pytest

from neurobrix.core.prism.loader import load_profile
from neurobrix.core.prism.solver import (
    ComponentMemory, DeviceState, PrismSolver)
from neurobrix.core.prism.structure import DeviceBrand, DeviceSpec

_GB = 1024 ** 3


class _NoComponents:
    from pathlib import Path as _P
    cache_path = _P("/nonexistent-prism-fixture")

    def get_neural_components(self):
        return []


def _solver_and_scene(unified: bool):
    """The CogVideoX condition: a live-constrained GPU (VM up) with the profile's
    24 GB CPU. A 'text_encoder' whose activations exceed the GPU but whose total
    fits host RAM is exactly what lazy_sequential offloaded to the host. A small
    'vae' fits whole beside it."""
    profile = load_profile("default")            # apple, unified, 24 GB cpu
    profile.devices[0].unified_memory = unified
    solver = PrismSolver.__new__(PrismSolver)
    from neurobrix.core.config.system import PRISM_DEFAULTS
    solver.oom_reserve_mb = PRISM_DEFAULTS["oom_reserve_mb"]
    solver._component_tiling = {}
    spec = DeviceSpec(index=0, name="dev", memory_mb=24576,
                      compute_capability="0.0", supports_dtypes=["bfloat16"],
                      architecture="apple_silicon", brand=DeviceBrand.APPLE)
    dev = DeviceState(device_string="mps:0" if unified else "cuda:0",
                      capacity_mb=10700.0, spec=spec, recommended_mb=10700.0)
    # sorted largest-first, as solve() sorts them
    big = ("text_encoder", ComponentMemory(component_name="text_encoder",
           weight_bytes=2 * _GB, activation_bytes=12 * _GB, overhead_bytes=0))
    small = ("vae", ComponentMemory(component_name="vae",
             weight_bytes=1 * _GB, activation_bytes=3 * _GB, overhead_bytes=0))
    return solver, [big, small], [dev], profile


def _lazy(unified: bool):
    solver, comps, devs, profile = _solver_and_scene(unified)
    return solver._try_lazy_sequential(comps, dict(comps), devs, {}, profile,
                                       _NoComponents())


# ── half 1: the guard removes the host offer on unified, keeps it on discrete ──

def test_lazy_sequential_offers_no_host_placement_on_a_unified_device():
    result = _lazy(unified=True)
    # The whole-component host-offload strategy must produce NO candidate: the
    # too-big component cannot go to the host (it frees nothing on unified), so
    # lazy_sequential returns None and the cascade routes past it.
    if result is not None:
        allocations, _ = result
        for name, alloc in allocations.items():
            dev = alloc[0] if isinstance(alloc, tuple) else alloc
            assert not str(dev).startswith("cpu"), (
                f"lazy_sequential placed {name!r} on the HOST ({dev!r}) on a "
                f"UNIFIED device — the offload frees nothing; the cascade must "
                f"route the component to layer_streaming instead")


def test_lazy_sequential_still_offloads_to_host_on_a_discrete_device(monkeypatch):
    """Inertness: on a discrete card the offload frees real device memory, so it
    stays available and lazy_sequential uses it. The host-fit check reads live
    host memory (memory_budget._host_budget_mb) — pin it large so the arm proves
    the GUARD's inertness, not the test machine's current free RAM."""
    from neurobrix.core import host_memory as _hm
    from neurobrix.core.host_memory import MemoryState
    monkeypatch.setattr(_hm, "memory_state",
                        lambda: MemoryState(total_mb=65536, available_mb=60000,
                                            source="test"))
    result = _lazy(unified=False)
    assert result is not None, (
        "on a discrete card the too-big component's weights belong on the host; "
        "lazy_sequential must still offer that plan")
    allocations, _ = result
    assert any(str((a[0] if isinstance(a, tuple) else a)).startswith("cpu")
               for a in allocations.values()), (
        f"expected a host placement on the discrete card, got {allocations!r}")


# ── half 2: the score and order make layer_streaming the destination ──

def _cascade_lists():
    tree = ast.parse(inspect.getsource(PrismSolver))
    lists = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        names = [elt.elts[0].value for elt in node.elts
                 if (isinstance(elt, ast.Tuple) and len(elt.elts) == 2
                     and isinstance(elt.elts[0], ast.Constant)
                     and isinstance(elt.elts[1], ast.Attribute)
                     and elt.elts[1].attr.startswith("_try_"))]
        if names:
            lists.append(names)
    return lists


def test_layer_streaming_is_tried_before_the_host_rungs_everywhere():
    lists = _cascade_lists()
    assert len(lists) >= 3, f"expected the three cascades, found {len(lists)}"
    for names in lists:
        if "layer_streaming" not in names:
            continue
        i = names.index("layer_streaming")
        for host in ("cpu_execution", "cpu_streaming"):
            if host in names:
                assert i < names.index(host), (
                    f"layer_streaming must be tried before {host}: {names}")


def test_layer_streaming_outscores_the_host_rungs():
    """Once the whole-component host offloads are gone, layer_streaming must beat
    what is left — leaving the accelerator (cpu_execution / cpu_streaming)."""
    tree = ast.parse(inspect.getsource(PrismSolver))
    scores = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        if "layer_streaming" not in keys or "cpu_execution" not in keys:
            continue
        for k, v in zip(node.keys, node.values):
            if (isinstance(k, ast.Constant) and isinstance(v, ast.Constant)
                    and isinstance(v.value, (int, float))):
                scores[k.value] = v.value
        break
    for name in ("layer_streaming", "cpu_execution", "cpu_streaming"):
        assert name in scores, f"{name} has no score: {sorted(scores)}"
    assert scores["layer_streaming"] > scores["cpu_execution"] > scores["cpu_streaming"], scores
