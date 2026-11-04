"""S2 (the supervisor, 2026-10-27 22:20 and the decision of 23:00): a machine whose GENERATED profile and architecture
vendor file state no measured facts is probed at its first launch (`autodetect.ensure_measured_facts`, through
`facts_probe`, the measuring tool's own primitives) for the device and host facts; the engine's LOADER rate is measured
by the loader itself at its first real load (`note_loader_rate`), cold then warm — until then Prism ranks without an
estimate and says so.

What each test would do if the code were wrong: a generated profile left unprobed (red); a committed profile written
(red); a fact the profile or its vendor file already states measured again (red); a loader rate kept after its warm
reading, or written for a load too small to time (red)."""
from __future__ import annotations

import json

import pytest
import yaml

from neurobrix.core.prism import autodetect as A
from neurobrix.core.prism import facts_probe as P

CALLS = []


@pytest.fixture
def hw(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "HARDWARE_DIR", tmp_path)
    CALLS.clear()
    monkeypatch.setattr(P, "matmul_tflops", lambda t, *a, **k: CALLS.append(("mm", t)) or {"best": 10.0})
    monkeypatch.setattr(P, "copies", lambda *a, **k: CALLS.append(("copies",)) or
                        {"memory_gbps": 500.0, "h2d_gbps": 10.0, "d2h_gbps": 9.0, "h2d_pageable_gbps": 5.0})
    monkeypatch.setattr(P, "host_matmul_gflops", lambda *a, **k: CALLS.append(("hmm",)) or 300.0)
    monkeypatch.setattr(P, "host_memory_gbps", lambda *a, **k: CALLS.append(("hmem",)) or 20.0)
    monkeypatch.setattr(P, "vmm_chunk_bytes", lambda *a, **k: CALLS.append(("vmm",)) or 2 << 20)
    return tmp_path


def _profile(hw, name, device):
    (hw / f"{name}.yml").write_text(yaml.safe_dump({"id": name, "devices": [device, dict(device)]}))


DEV = {"brand": "nvidia", "model": "Unknown-Card", "architecture": "unknownarch", "supports_dtypes": ["float16", "float32"]}


def test_a_generated_profile_without_facts_is_probed_once_and_written(hw):
    _profile(hw, "default-abc", DEV)
    assert set(A.ensure_measured_facts("default-abc")) == {"measured", "host_measured", "vmm_chunk_bytes"}
    data = yaml.safe_load((hw / "default-abc.yml").read_text())
    for d in data["devices"]:
        assert d["measured"]["matmul_tflops"] == {"float16": 10.0, "float32": 10.0} and d["measured"]["h2d_gbps"] == 10.0
        assert "first launch" in d["measured"]["source"] and "weight_load_gbps" not in d["measured"]
        assert d["measured"]["vmm_chunk_bytes"] == 2 << 20
    assert data["host_measured"]["host_memory_gbps"] == 20
    CALLS.clear()
    assert A.ensure_measured_facts("default-abc") == {} and CALLS == []


def test_a_committed_profile_is_never_written(hw, tmp_path):
    _profile(hw, "c4140-something", DEV)
    before = (hw / "c4140-something.yml").read_text()
    assert A.ensure_measured_facts("c4140-something") == {}
    assert not A.note_loader_rate("c4140-something", "triton", "bfloat16", "device", 1 << 30, 1.0, "m:c")
    assert (hw / "c4140-something.yml").read_text() == before and CALLS == []


def test_what_the_vendor_file_states_is_not_probed_again(hw):
    """A V100 (volta.yml states the device facts): only the host facts, which volta.yml does not state, are probed."""
    _profile(hw, "default-v100", {"brand": "nvidia", "model": "Tesla V100-SXM2-16GB", "architecture": "volta",
                                  "compute_capability": "7.0", "supports_dtypes": ["float16", "float32"]})
    # the expandable pool's chunk is the machine's measure, never the vendor file's (R16)
    assert set(A.ensure_measured_facts("default-v100")) == {"host_measured", "vmm_chunk_bytes"}
    assert not any(c[0] in ("mm", "copies") for c in CALLS), CALLS


def test_the_loader_measures_itself_cold_then_warm_then_stops(hw):
    _profile(hw, "default-abc", DEV)
    A.ensure_measured_facts("default-abc")
    assert not A.note_loader_rate("default-abc", "triton", "bfloat16", "device", 1 << 20, 0.01, "m:c"), "too small"
    assert A.note_loader_rate("default-abc", "triton", "bfloat16", "device", 2 * 10 ** 9, 2.0, "m:c")
    assert A.note_loader_rate("default-abc", "triton", "bfloat16", "device", 4 * 10 ** 9, 2.0, "m:c")
    assert not A.note_loader_rate("default-abc", "triton", "bfloat16", "device", 4 * 10 ** 9, 1.0, "m:c")
    d = yaml.safe_load((hw / "default-abc.yml").read_text())["devices"][1]["measured"]
    assert d["weight_load_gbps"]["triton"]["BF16"]["device"] == {"cold": 1.0, "warm": 2.0}
    assert d["matmul_tflops"]["float16"] == 10.0, "the probe's facts kept beside the loader's"


def test_the_engine_measures_its_per_token_reload_once_on_a_generated_profile(hw):
    """The supervisor 2026-11-02 06:40 (B): the per-token reload rate a layer_streaming plan pays is a profile fact the
    engine measures itself at a real reload (`note_reload_rate`), per engine and stored dtype, written once into a
    GENERATED profile's host_measured; a committed profile is never written, a reload too small to time is not kept.
    What this test would do if the code were wrong: a second measure overwriting the first, a committed profile written,
    or the rate filed under another key than the estimator reads (time_estimate.reload_bps) — RED."""
    from neurobrix.core.prism import time_estimate as T
    _profile(hw, "default-abc", DEV)
    A.ensure_measured_facts("default-abc")
    assert not A.note_reload_rate("default-abc", "compiled", "bfloat16", 1 << 20, 0.01, "m:c"), "too small"
    assert A.note_reload_rate("default-abc", "compiled", "bfloat16", 2 * 10 ** 9, 2.0, "m:c")
    assert not A.note_reload_rate("default-abc", "compiled", "bfloat16", 8 * 10 ** 9, 2.0, "m:c"), "written once"
    assert A.note_reload_rate("default-abc", "triton", "bfloat16", 3 * 10 ** 9, 2.0, "m:c")
    host = yaml.safe_load((hw / "default-abc.yml").read_text())["host_measured"]
    assert host["weight_reload_gbps"] == {"compiled": {"BF16": 1.0}, "triton": {"BF16": 1.5}}
    assert abs(T.reload_bps(T.host_facts(host), "compiled", "bfloat16") - 1e9) < 1
    _profile(hw, "c4140-something", DEV)
    assert not A.note_reload_rate("c4140-something", "compiled", "bfloat16", 2 * 10 ** 9, 2.0, "m:c")


def test_the_reload_rate_is_written_only_from_three_steady_runs(hw, monkeypatch):
    """The supervisor 2026-11-02 09:50: a profile fact a neighbour can disturb is no fact. The engine's reload rate is
    written from three consecutive runs whose rates agree within 5 % (their median), and an unsteady reading is never
    written. What this test would do if the code wrote the first reading: a fact after one run, or after three runs
    spread 2x — RED."""
    from neurobrix.core.strategies import layer_streaming as LS
    _profile(hw, "default-abc", DEV)
    A.ensure_measured_facts("default-abc")
    monkeypatch.setattr(A, "ACTIVE_HARDWARE_ID", "default-abc")
    monkeypatch.setattr(A, "stored_major", lambda *a: "bfloat16")
    monkeypatch.setattr(LS, "_RELOAD_DRAWS", {})
    ex = type("E", (), {"mode": "compiled"})()

    def written():
        h = yaml.safe_load((hw / "default-abc.yml").read_text()).get("host_measured") or {}
        return (h.get("weight_reload_gbps") or {}).get("compiled", {}).get("BF16")

    for b, s in ((10 ** 9, 1.0), (10 ** 9, 2.0), (10 ** 9, 1.0)):       # 1, 0.5, 1 GB/s: not steady
        LS._note_reload(ex, "/x", "lm", b, s)
    assert written() is None
    for b, s in ((10 ** 9, 1.0), (10 ** 9, 1.02), (10 ** 9, 0.99)):      # steady around 1 GB/s
        LS._note_reload(ex, "/x", "lm", b, s)
    assert written() is not None and abs(written() - 0.98) < 0.03, written()


def test_a_profile_s_host_facts_lie_over_its_vendor_s_key_by_key():
    """The Dell's 1.12 rule (2026-11-03 22:54): the vendor's host_measured (the architecture's) under the profile's (this
    machine's, measured by the engine), key by key into its tables. What this would do if the profile's block replaced the
    vendor's (the rule before): the vendor's host_slab_gbps and its triton reload rate lost — RED."""
    from neurobrix.core.prism.loader import _over
    vendor = {"host_slab_gbps": 3.0, "host_matmul_gflops": 100, "weight_reload_gbps": {"triton": {"BF16": 2.1}}}
    prof = {"host_matmul_gflops": 300, "weight_reload_gbps": {"compiled": {"BF16": 1.0}, "triton": {"F32": 0.5}}}
    assert _over(vendor, prof) == {"host_slab_gbps": 3.0, "host_matmul_gflops": 300,
                                   "weight_reload_gbps": {"triton": {"BF16": 2.1, "F32": 0.5}, "compiled": {"BF16": 1.0}}}
    assert _over(None, prof) == prof and _over(vendor, None) == vendor


def _container(root, name, comp, dtype, nbytes):
    c = root / name / "components" / comp
    c.mkdir(parents=True)
    (root / name / "manifest.json").write_text("{}")
    (c / "graph.json").write_text("{}")
    (c / "weights_index.json").write_text(json.dumps({"tensors": {"w": {"dtype": dtype, "size_bytes": nbytes}}}))
    with open(c / "w.safetensors", "wb") as f:
        f.truncate(nbytes)                       # sparse: the size is what the probe reads


def test_the_first_launch_reloads_the_smallest_component_of_each_unstated_stored_dtype(hw, tmp_path, monkeypatch):
    """The supervisor 2026-11-03 22:50: the per-piece reload rate is measured by the engine at the first launch, per
    stored dtype of the cache neither block states, on the smallest component stored so (at least the loader-rate floor)
    — before, only a streamed run measured it and Prism refused to plan one without it. What this would do if the code
    were wrong: no probe (the fact never measured), a component under the floor or not the smallest, or a stated rate
    measured again — RED."""
    from neurobrix.core.prism import arena_probe as AP
    import neurobrix.core.paths as paths
    cache = tmp_path / "cache"
    floor = A.LOADER_RATE_MIN_BYTES
    _container(cache, "big", "lm", "bfloat16", 4 * floor)
    _container(cache, "small", "lm", "bfloat16", 2 * floor)
    _container(cache, "tiny", "lm", "bfloat16", floor // 2)
    _container(cache, "f32", "net", "float32", 3 * floor)
    monkeypatch.setattr(paths, "cache_dir", lambda: cache)
    monkeypatch.setattr(P, "gemv_gbps", lambda *a, **k: 100.0)          # the other first-launch probes: not this test's
    probed = []
    monkeypatch.setattr(AP, "child", lambda d, c, h, e: probed.append((d.rsplit("/", 1)[1], c, e)) or
                        {"container": d, "component": c, "draws": []})
    _profile(hw, "default-abc", DEV)
    A.ensure_measured_facts("default-abc", "compiled")
    assert sorted(probed) == [("f32", "net", "compiled"), ("small", "lm", "compiled")], probed
    data = yaml.safe_load((hw / "default-abc.yml").read_text())
    data["host_measured"]["weight_reload_gbps"] = {"compiled": {"BF16": 2.0, "F32": 1.0}}
    (hw / "default-abc.yml").write_text(yaml.safe_dump(data))
    probed.clear()
    A.ensure_measured_facts("default-abc", "compiled")
    assert probed == [], "a stated rate is not measured again"


def test_the_reload_counts_an_nbxtensor_s_bytes():
    """Flex.1-alpha triton, 2026-11-04: `_weights_bytes` took `nbytes` first, a METHOD on NBXTensor (ATen's API), and died
    on int(method) at the first streamed reload. What this would do if the order came back: TypeError — RED."""
    from neurobrix.core.strategies import layer_streaming as LS
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.dtype.nbx_dtype import parse_dtype
    ex = type("E", (), {"_weights": {"w": NBXTensor.zeros((4, 8), parse_dtype("float32"), "cpu"), "none": None}})()
    assert LS._weights_bytes(ex) == 4 * 8 * 4
