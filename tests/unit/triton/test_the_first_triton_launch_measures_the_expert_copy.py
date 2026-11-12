"""The first Triton launch measures the expert store's miss copy where the profile carries none, and only there (the
supervisor 2026-11-03 10:32: the copy is 75-93 % of a mixture-of-experts decode's GPU time on the V100s): the profile's
`measured.expert_copy` (block, warps), written by `autodetect.ensure_measured_facts` from
`nbxtensor.ops.moe_device_residency.measure_miss_copy`. What the store does with the fact — launch it, refuse without
it — is the library's test (`nbxtensor/tests/test_the_expert_slots_copy_with_the_measured_launch.py`). What this test
would do if the code were wrong: a first launch that measures again what the profile carries, or never measures — RED;
a unified device probed — RED. No card."""
from __future__ import annotations


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
