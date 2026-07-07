"""Unit tests: triton-side brick-consolidation helpers (CPU-only, no GPU).

Covers the three consolidated triton-side bricks of P-BRICK-CONSOLIDATION
(latent-correctness half):

  - `device_transfer.parse_device_idx` / `parse_device_idxs` — the SINGLE
    triton device-string parser that replaced six divergent
    `_parse_device_idx` copies in the flow handlers. The table below pins
    the canonical semantics, including the forms the old copies disagreed
    on (compound "fgp:" strings, bare comma lists).
  - `memory_pool.release_flow_memory` — the flow-boundary release mirror of
    the compiled MemoryManager discipline: sync-before-free ordering is
    asserted via recorded call sequences (no CUDA needed).
  - `dtype.resolve_compute_dtype` — the Prism-plan (not manifest) compute
    dtype resolver, string-dtype boundary (R33).

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/triton/test_device_parse_and_flow_memory.py -v
  - script:  PYTHONPATH=src python3 tests/unit/triton/test_device_parse_and_flow_memory.py
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from neurobrix.triton import memory_pool
from neurobrix.triton.device_transfer import parse_device_idx, parse_device_idxs
from neurobrix.triton.dtype import resolve_compute_dtype


# ---------------------------------------------------------------------------
# parse_device_idx / parse_device_idxs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device,expected", [
    # single-device forms
    ("cuda:0", 0),
    ("cuda:2", 2),
    ("cuda:13", 13),
    # bare ordinal strings (historical flow behaviour)
    ("0", 0),
    ("3", 3),
    (2, 2),
    # compound Prism placement strings — the divergence class the old
    # copies disagreed on. Canonical: FIRST cuda ordinal = primary device.
    ("fgp:cuda:0,cuda:1", 0),      # naive old copy CRASHED here (int("cuda"))
    ("fgp:cuda:1,cuda:3", 1),      # three old copies silently returned 0 here
    ("tp:cuda:2,cuda:3", 2),
    ("zero3:cuda:1", 1),
    ("cuda:2,cuda:3", 2),          # old fast path returned the LAST (3) here
    # degenerate forms — every historical copy resolved these to 0
    ("cuda", 0),
    ("cpu", 0),
    ("", 0),
    (None, 0),
    ("not-a-device", 0),
    ("cuda:abc", 0),
])
def test_parse_device_idx(device, expected):
    assert parse_device_idx(device) == expected


@pytest.mark.parametrize("device,expected", [
    ("cuda:2", [2]),
    ("fgp:cuda:0,cuda:1", [0, 1]),
    ("tp:cuda:2,cuda:3", [2, 3]),
    ("cuda:2,cuda:3", [2, 3]),
    ("cuda", []),
    ("cpu", []),
    (None, []),
])
def test_parse_device_idxs(device, expected):
    assert parse_device_idxs(device) == expected


# ---------------------------------------------------------------------------
# release_flow_memory — sync-before-free ordering (recorded, no CUDA)
# ---------------------------------------------------------------------------

class _FakeDeviceAllocator:
    """Records the DeviceAllocator calls release_flow_memory makes."""

    def __init__(self, events, current=7):
        self.events = events
        self._current = current

    def get_device(self):
        self.events.append(f"get:{self._current}")
        return self._current

    def set_device(self, idx):
        self.events.append(f"set:{idx}")

    def sync_device(self):
        self.events.append("sync")

    def empty_cache_pool(self):
        self.events.append("pool_flush")
        return 0


def _patched_release(monkeypatch, device, current=7):
    events = []
    fake = _FakeDeviceAllocator(events, current=current)
    monkeypatch.setattr(memory_pool, "DeviceAllocator", fake)
    monkeypatch.setattr(
        memory_pool, "gc", SimpleNamespace(collect=lambda: events.append("gc")))
    memory_pool.release_flow_memory(device)
    return events


def test_release_flow_memory_sync_before_gc_single_device(monkeypatch):
    events = _patched_release(monkeypatch, "cuda:1")
    # sync the named device BEFORE gc runs the finalizers (err-700 class),
    # then restore the previously-current device, then collect, then flush.
    assert events == ["get:7", "set:1", "sync", "set:7", "gc", "pool_flush"]


def test_release_flow_memory_syncs_every_compound_device(monkeypatch):
    events = _patched_release(monkeypatch, "fgp:cuda:0,cuda:1")
    assert events == ["get:7", "set:0", "sync", "set:1", "sync", "set:7",
                      "gc", "pool_flush"]


def test_release_flow_memory_cpu_skips_device_calls(monkeypatch):
    events = _patched_release(monkeypatch, "cpu")
    assert events == ["gc", "pool_flush"]


# ---------------------------------------------------------------------------
# resolve_compute_dtype — Prism plan is the authority, manifest the fallback
# ---------------------------------------------------------------------------

def _ctx(plan, manifest_dtype="float16"):
    manifest = {} if manifest_dtype is None else {"dtype": manifest_dtype}
    return SimpleNamespace(plan=plan, pkg=SimpleNamespace(manifest=manifest))


def _plan(components=None, target=None):
    return SimpleNamespace(components=components, target_dtype=target)


def test_resolve_compute_dtype_plan_wins_over_manifest():
    # bf16 manifest resolved to fp32 by Prism (e.g. non-bf16 hardware):
    # the RESOLVED plan dtype must win — this is the stale-authority bug.
    plan = _plan({"model": SimpleNamespace(dtype="float32")}, target="float32")
    assert resolve_compute_dtype(_ctx(plan, "bfloat16")) == "float32"


def test_resolve_compute_dtype_component_selects_its_allocation():
    plan = _plan({"vae": SimpleNamespace(dtype="float32"),
                  "lm": SimpleNamespace(dtype="float16")})
    assert resolve_compute_dtype(_ctx(plan), component="lm") == "float16"
    assert resolve_compute_dtype(_ctx(plan), component="vae") == "float32"
    # unnamed → first allocation carrying a dtype
    assert resolve_compute_dtype(_ctx(plan)) == "float32"
    # unknown component → same first-allocation answer, no crash
    assert resolve_compute_dtype(_ctx(plan), component="missing") == "float32"


def test_resolve_compute_dtype_target_dtype_fallback():
    plan = _plan(components={}, target="float16")
    assert resolve_compute_dtype(_ctx(plan, "bfloat16")) == "float16"


def test_resolve_compute_dtype_manifest_only_without_plan():
    assert resolve_compute_dtype(_ctx(None, "bfloat16")) == "bfloat16"
    assert resolve_compute_dtype(_ctx(None, None)) == "float16"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
