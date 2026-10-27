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
    return tmp_path


def _profile(hw, name, device):
    (hw / f"{name}.yml").write_text(yaml.safe_dump({"id": name, "devices": [device, dict(device)]}))


DEV = {"brand": "nvidia", "model": "Unknown-Card", "architecture": "unknownarch", "supports_dtypes": ["float16", "float32"]}


def test_a_generated_profile_without_facts_is_probed_once_and_written(hw):
    _profile(hw, "default-abc", DEV)
    assert set(A.ensure_measured_facts("default-abc")) == {"measured", "host_measured"}
    data = yaml.safe_load((hw / "default-abc.yml").read_text())
    for d in data["devices"]:
        assert d["measured"]["matmul_tflops"] == {"float16": 10.0, "float32": 10.0} and d["measured"]["h2d_gbps"] == 10.0
        assert "first launch" in d["measured"]["source"] and "weight_load_gbps" not in d["measured"]
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
    assert set(A.ensure_measured_facts("default-v100")) == {"host_measured"}
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
