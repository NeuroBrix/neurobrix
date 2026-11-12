"""The expert slot store's miss copy launches as this machine measured it (the copy is 75-93 % of a mixture-of-experts
decode's GPU time on a V100): the profile's `measured.expert_copy` (block, warps), measured by
`nbxtensor.ops.moe_device_residency.measure_miss_copy` — the fixed 4096 x 4 the store once launched gave 3.35 GB/s where
the grid's best gave 3.98-4.09 on a quiet V100 (2026-11-03). What this test would do if the code were wrong: a launch
that ignores the profile (a constant in the store) — RED; a profile without the fact served anyway — RED (no refusal);
a launch carried from another card — RED. No card."""
from __future__ import annotations

import types

import pytest


def _store(idx=0):
    from nbxtensor.memory.expert_slots import ExpertSlotStore
    s = types.SimpleNamespace(device_idx=idx, _copy=None)
    return lambda: ExpertSlotStore._copy_launch(s)


def _profile(monkeypatch, measured, idx=0):
    prof = types.SimpleNamespace(id="default-x", devices=[types.SimpleNamespace(index=idx, measured=measured)])
    monkeypatch.setattr("nbxtensor.ops.wrappers.get_hardware_profile", lambda: prof)


def test_the_store_launches_the_measured_copy(monkeypatch):
    _profile(monkeypatch, {"expert_copy": {"block": 16384, "warps": 4, "gbps": 3.98}})
    assert _store()() == (16384, 4)


def test_without_the_fact_the_copy_is_refused_by_name(monkeypatch):
    _profile(monkeypatch, {"vmm_chunk_bytes": 1})
    with pytest.raises(RuntimeError, match="expert_copy"):
        _store()()


def test_a_card_the_profile_does_not_list_is_refused(monkeypatch):
    _profile(monkeypatch, {"expert_copy": {"block": 16384, "warps": 4}}, idx=0)
    with pytest.raises(RuntimeError, match="device 1"):
        _store(1)()
