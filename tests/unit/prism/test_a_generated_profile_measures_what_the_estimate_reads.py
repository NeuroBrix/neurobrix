"""S2 (the supervisor, 2026-10-27 22:20): a machine whose GENERATED profile and architecture vendor file state no
measured facts is measured at its first launch (`autodetect.ensure_measured_facts`, through `facts_probe`, the measuring
tool's own primitives) instead of planned without a time estimate — the device facts, the host facts, and the engine's
loader rate for each stored dtype the model holds — written once into that profile with their source.

What each test would do if the code were wrong: a generated profile left unmeasured (red); a committed profile
written (red); a fact the profile or its vendor file already states measured again (red); a loader rate measured for
another engine than the run's (red)."""
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

    def loads(root, comp, gb, engines):
        CALLS.append(("load", comp, tuple(engines)))
        return {"dtype": "BF16", "stored_bytes": int(gb * 1e9), "weights": 3,
                **{e: {"device": {"cold": 1.0, "warm": 2.0}, "host": {"cold": 0.5, "warm": 0.6}} for e in engines}}
    monkeypatch.setattr(P, "weight_load_gbps", loads)
    return tmp_path


def _profile(hw, name, device):
    (hw / f"{name}.yml").write_text(yaml.safe_dump({"id": name, "devices": [device, dict(device)]}))


def _model(tmp_path):
    c = tmp_path / "model"
    (c / "components" / "lm").mkdir(parents=True, exist_ok=True)
    (c / "components" / "lm" / "weights_index.json").write_text(json.dumps(
        {"tensors": {"w": {"dtype": "bfloat16", "size_bytes": 10}}}))
    return c


DEV = {"brand": "nvidia", "model": "Unknown-Card", "architecture": "unknownarch", "supports_dtypes": ["float16", "float32"]}


def test_a_generated_profile_without_facts_is_measured_once_and_written(hw, tmp_path):
    _profile(hw, "default-abc", DEV)
    added = A.ensure_measured_facts("default-abc", _model(tmp_path), "triton")
    assert set(added) == {"measured", "host_measured", "weight_load_gbps"}
    data = yaml.safe_load((hw / "default-abc.yml").read_text())
    for d in data["devices"]:
        m = d["measured"]
        assert m["matmul_tflops"] == {"float16": 10.0, "float32": 10.0} and m["h2d_gbps"] == 10.0
        assert m["weight_load_gbps"]["triton"]["BF16"]["host"]["cold"] == 0.5 and "compiled" not in m["weight_load_gbps"]
        assert "first launch" in m["source"]
    assert data["host_measured"]["host_memory_gbps"] == 20
    assert ("load", "lm", ("triton",)) in CALLS
    CALLS.clear()
    assert A.ensure_measured_facts("default-abc", _model(tmp_path), "triton") == {}
    assert CALLS == [], f"measured again: {CALLS}"
    A.ensure_measured_facts("default-abc", _model(tmp_path), "compiled")
    assert CALLS == [("load", "lm", ("compiled",))], CALLS


def test_a_committed_profile_is_never_written(hw, tmp_path):
    _profile(hw, "c4140-something", DEV)
    before = (hw / "c4140-something.yml").read_text()
    assert A.ensure_measured_facts("c4140-something", _model(tmp_path), "triton") == {}
    assert (hw / "c4140-something.yml").read_text() == before and CALLS == []


def test_what_the_vendor_file_states_is_not_measured_again(hw, tmp_path):
    """A V100 (volta.yml states the device facts and the rack's loader rates): only the host facts, which volta.yml
    does not state, are measured."""
    _profile(hw, "default-v100", {"brand": "nvidia", "model": "Tesla V100-SXM2-16GB", "architecture": "volta",
                                  "compute_capability": "7.0", "supports_dtypes": ["float16", "float32"]})
    added = A.ensure_measured_facts("default-v100", _model(tmp_path), "triton")
    assert set(added) == {"host_measured"}, added
    assert not any(c[0] in ("mm", "copies", "load") for c in CALLS), CALLS
