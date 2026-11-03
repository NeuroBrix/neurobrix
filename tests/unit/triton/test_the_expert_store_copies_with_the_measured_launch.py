"""The expert store's miss copy launches as this machine measured it (the supervisor 2026-11-03 10:32: the copy is 75-93 %
of a mixture-of-experts decode's GPU time on the V100s): the profile's `measured.expert_copy` (block, warps), written at
the first Triton launch by `nbxtensor.ops.moe_device_residency.measure_miss_copy` where the profile carries none — the fixed 4096 x 4 the store
launched gave 3.35 GB/s where the grid's best gave 3.98-4.09 on a quiet V100 (2026-11-03). What this test would do if
the code were wrong: a launch that ignores the profile (a constant in the store) — RED; a profile without the fact
served anyway — RED (no refusal); a first launch that measures again what the profile carries, or never measures — RED.
No card."""
from __future__ import annotations

import types

import pytest


def _store(idx=0):
    from neurobrix.triton.expert_store import NBXExpertStore
    s = types.SimpleNamespace(device_idx=idx, _copy=None)
    return lambda: NBXExpertStore._copy_launch(s)


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


def test_the_first_triton_launch_measures_the_copy_only_where_the_profile_has_none(monkeypatch, tmp_path):
    from neurobrix.core.prism import autodetect as AD
    import yaml
    monkeypatch.setattr(AD, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr("neurobrix.core.paths.cache_dir", lambda: tmp_path / "no-cache")   # no reload probe here
    monkeypatch.setattr("neurobrix.core.config.loader.vendor_config_for_device",
                        lambda *a, **k: {"memory": {"triton_arena_activation_factor": 1.35}})
    probed = []
    monkeypatch.setattr("nbxtensor.ops.moe_device_residency.measure_miss_copy", lambda *a, **k: probed.append(a) or {"block": 8192, "warps": 2, "gbps": 3.6,
                                                                                "grid": {"8192x2": 3.6}})
    for carried, expect in (({"block": 4096, "warps": 4}, 0), (None, 1)):
        measured = {"source": "s", "vmm_chunk_bytes": 1, "gemv_gbps": {"triton": {}}}
        if carried:
            measured["expert_copy"] = carried
        spec = {"id": "default-x", "devices": [{"index": 0, "brand": "nvidia", "architecture": "volta", "model": "V100",
                                                "measured": measured, "runtime_total_mb": 16151, "runtime_context_mb": 306}],
                "host_measured": {"host_matmul_gflops": 1}}
        (tmp_path / "default-x.yml").write_text(yaml.safe_dump(spec))
        probed.clear()
        AD.ensure_measured_facts("default-x", "triton")
        assert len(probed) == expect
        written = yaml.safe_load((tmp_path / "default-x.yml").read_text())["devices"][0]["measured"]["expert_copy"]
        assert written == (carried or {"block": 8192, "warps": 2, "gbps": 3.6})     # the grid is not written


def test_a_unified_device_is_not_probed(monkeypatch, tmp_path):
    """On a unified device the experts are mapped in place (Strategy 2.5): the first launch measures no card-side copy
    (whose Triton launch does not bind on Metal, the Mac 2026-11-03 10:43) and writes none."""
    from neurobrix.core.prism import autodetect as AD
    import yaml
    monkeypatch.setattr(AD, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr("neurobrix.core.paths.cache_dir", lambda: tmp_path / "no-cache")   # no reload probe here
    monkeypatch.setattr("neurobrix.core.config.loader.vendor_config_for_device",
                        lambda *a, **k: {"memory": {"triton_arena_activation_factor": 1.19}})
    probed = []
    monkeypatch.setattr("nbxtensor.ops.moe_device_residency.measure_miss_copy", lambda *a, **k: probed.append(a) or {"block": 1, "warps": 1, "gbps": 1})
    spec = {"id": "default-m", "devices": [{"index": 0, "brand": "apple", "architecture": "apple_silicon", "model": "M4",
                                            "unified_memory": True,
                                            "measured": {"source": "s", "vmm_chunk_bytes": 0, "gemv_gbps": {"triton": {}}}}],
            "host_measured": {"host_matmul_gflops": 1}}
    (tmp_path / "default-m.yml").write_text(yaml.safe_dump(spec))
    AD.ensure_measured_facts("default-m", "triton")
    assert probed == []
    assert "expert_copy" not in yaml.safe_load((tmp_path / "default-m.yml").read_text())["devices"][0]["measured"]
