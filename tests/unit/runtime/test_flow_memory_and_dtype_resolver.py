"""Unit tests: compiled-side brick-consolidation helpers (CPU-only, no GPU).

Covers the compiled half of the P-BRICK-CONSOLIDATION latent-correctness
clusters:

  - `MemoryManager.release_flow_memory` — the flow-boundary release that
    replaced the hand-rolled `gc.collect(); device_empty_cache(dev)` pairs
    in core/flow/. The sync-BEFORE-collect ordering (err-700 protection) is
    asserted via recorded call sequences.
  - `FlowContext.compute_dtype` — the SINGLE compiled-side compute-dtype
    resolver: Prism-RESOLVED plan is the authority, the manifest only the
    no-plan fallback. Pins the fix for the dead `plan.allocations` walk
    (ExecutionPlan carries `.components`, never `.allocations`) and for the
    three flow copies that read `manifest["dtype"]` directly.

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/runtime/test_flow_memory_and_dtype_resolver.py -v
  - script:  PYTHONPATH=src python3 tests/unit/runtime/test_flow_memory_and_dtype_resolver.py
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from neurobrix.core.flow.base import FlowContext
from neurobrix.core.memory import manager as memory_manager


# ---------------------------------------------------------------------------
# MemoryManager.release_flow_memory — sync-before-free ordering
# ---------------------------------------------------------------------------

def _patched_release(monkeypatch, device):
    events = []
    monkeypatch.setattr(memory_manager, "device_sync",
                        lambda d=None: events.append(f"sync:{d}"))
    monkeypatch.setattr(memory_manager, "device_empty_cache",
                        lambda d=None: events.append(f"empty:{d}"))
    monkeypatch.setattr(memory_manager, "gc",
                        SimpleNamespace(collect=lambda: events.append("gc")))
    memory_manager.MemoryManager.release_flow_memory(device)
    return events


def test_release_flow_memory_orders_sync_gc_empty(monkeypatch):
    # sync BEFORE gc (finalizers → cudaFree must not race in-flight
    # kernels: the err-700 class), empty_cache AFTER the collect.
    assert _patched_release(monkeypatch, "cuda:0") == \
        ["sync:cuda:0", "gc", "empty:cuda:0"]


def test_release_flow_memory_none_device_still_collects(monkeypatch):
    # device_sync(None)/device_empty_cache(None) are documented no-ops in
    # device_utils; the collect must still run.
    events = _patched_release(monkeypatch, None)
    assert "gc" in events


def test_release_flow_memory_convenience_function(monkeypatch):
    events = []
    monkeypatch.setattr(memory_manager, "device_sync",
                        lambda d=None: events.append(f"sync:{d}"))
    monkeypatch.setattr(memory_manager, "device_empty_cache",
                        lambda d=None: events.append(f"empty:{d}"))
    monkeypatch.setattr(memory_manager, "gc",
                        SimpleNamespace(collect=lambda: events.append("gc")))
    memory_manager.release_flow_memory("cuda:1")
    assert events == ["sync:cuda:1", "gc", "empty:cuda:1"]


# ---------------------------------------------------------------------------
# FlowContext.compute_dtype — Prism plan is the authority
# ---------------------------------------------------------------------------

def _flow_ctx(plan, manifest_dtype="float16"):
    manifest = {} if manifest_dtype is None else {"dtype": manifest_dtype}
    return FlowContext(
        pkg=SimpleNamespace(manifest=manifest),
        plan=plan,
        variable_resolver=None,
        executors={},
        modules={},
        strategy=None,
        connections_index={},
        loop_id="",
        nbx_path_str="",
    )


def _plan(components=None, target=None):
    return SimpleNamespace(components=components, target_dtype=target)


def test_compute_dtype_plan_wins_over_manifest():
    # bf16 manifest resolved to fp32 by Prism (e.g. non-bf16 hardware):
    # the RESOLVED allocation dtype must win — this is the stale-authority
    # bug the three manifest-reading flow copies carried.
    plan = _plan({"model": SimpleNamespace(dtype="float32")}, target="float32")
    assert _flow_ctx(plan, "bfloat16").compute_dtype() == torch.float32


def test_compute_dtype_component_selects_its_allocation():
    plan = _plan({"vae": SimpleNamespace(dtype="float32"),
                  "lm": SimpleNamespace(dtype="float16")})
    ctx = _flow_ctx(plan)
    assert ctx.compute_dtype("lm") == torch.float16
    assert ctx.compute_dtype("vae") == torch.float32
    # unnamed → first allocation carrying a dtype
    assert ctx.compute_dtype() == torch.float32
    # unknown component → same first-allocation answer, no crash
    assert ctx.compute_dtype("missing") == torch.float32


def test_compute_dtype_target_dtype_fallback():
    plan = _plan(components={}, target="float16")
    assert _flow_ctx(plan, "bfloat16").compute_dtype() == torch.float16


def test_compute_dtype_manifest_only_without_plan():
    assert _flow_ctx(None, "bfloat16").compute_dtype() == torch.bfloat16
    assert _flow_ctx(None, None).compute_dtype() == torch.float16


def test_compute_dtype_reads_components_not_allocations():
    # Regression pin for the dead-code walk: the old tts_llm copy guarded
    # on `hasattr(plan, 'allocations')`, an attribute ExecutionPlan never
    # had, and silently fell back to defaults.json. A plan exposing ONLY
    # `.components` must be honoured.
    plan = SimpleNamespace(components={"m": SimpleNamespace(dtype="float32")},
                           target_dtype="float32")
    assert not hasattr(plan, "allocations")
    assert _flow_ctx(plan, "float16").compute_dtype() == torch.float32


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
