"""A gate cell runs at the model's floor; the model is judged at its judging request (the validation regime, rule 2).

Hocine, 2026-10-13 14:20: a gate run lasts at most ten minutes — it proves rc 0, zero certified miss and the arms'
agreement, not quality; a video model's gate request is the smallest that exercises the path (steps, frames,
resolution at the model's floor). Quality is judged once per model and branch at the real request. On the mq25
landing gate Allegro-TI2V took 1.2-1.6 h per run; fourteen cells above ten minutes have been carried by record since.

The model's gate values are carried by its container (`extracted_values._global.gate`, written by Forge from the
registry with their source), read by `trace_request.gate_request` over the judging request; the harness's cells run
it and the census covers it beside the judging request.

What this test does if the code is wrong: the floor cell fails when the gate request does not carry the container's
steps and frames (red on main: no gate_request), or carries the citation; the default cell fails when a container
stating no gate value is gated at anything but its judging request; the refusal cells fail when an unknown key or
half a size is accepted; the harness cell fails when the cell's request is not the gate request; the census cell
fails when the census does not cover the gate request beside the judging one.

Run: python -m pytest tests/unit/tools/test_a_gate_runs_at_the_models_floor.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))
import trace_request as T  # noqa: E402

MODEL = "FakeVideo-720x1280"
SRC = ["measured: the cell at 1 step and 9 frames runs every op of the loop in 6 min"]


def _topo(gate=None, conf=None):
    g = {"envelope": {"min": {"height": 720, "width": 1280}, "source": ["card"]}}
    if conf is not None:
        g["confirmation"] = conf
    if gate is not None:
        g["gate"] = gate
    return {"extracted_values": {"_global": g},
            "components": {"transformer": {"shapes": {"hidden_states": [2, 4, 22, 90, 160]}}}}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    from neurobrix.nbx.neurotax import NEUROTAX_VERSION
    d = tmp_path / "cache" / MODEL
    (d / "runtime").mkdir(parents=True)
    (d / "components" / "transformer").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "model_name": MODEL, "family": "video", "vae_scale_factor": 8, "neurotax_version": NEUROTAX_VERSION}))
    (d / "runtime" / "variables.json").write_text("{}")
    (d / "runtime" / "defaults.json").write_text("{}")
    (d / "components" / "transformer" / "graph.json").write_text("{}")
    monkeypatch.setenv("NEUROBRIX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(T.Z, "CACHE", tmp_path / "cache")

    def write(topology):
        from neurobrix.nbx.fixed_point import FIXED_POINT_FLAG
        topology = json.loads(json.dumps(topology))
        topology.setdefault("extracted_values", {})["transformer"] = {FIXED_POINT_FLAG: "fixture"}
        (d / "topology.json").write_text(json.dumps(topology))
    return write


def _flag(req, flag):
    assert req.count(flag) == 1, (flag, req)
    return req[req.index(flag) + 1]


def test_the_gate_request_carries_the_models_floor(cache):
    cache(_topo(gate={"steps": 1, "num_frames": 9, "source": SRC}, conf={"steps": 10, "source": ["judging"]}))
    judging = T.derived_request(MODEL, "video")
    gate = T.gate_request(MODEL, "video")
    assert _flag(judging, "--steps") == "10" and "--num-frames" not in judging
    assert (_flag(gate, "--steps"), _flag(gate, "--num-frames"), _flag(gate, "--height"), _flag(gate, "--width")) \
        == ("1", "9", "720", "1280"), gate
    assert "--source" not in gate and not any("measured" in str(a) for a in gate)


def test_a_model_stating_no_gate_is_gated_at_its_judging_request(cache):
    cache(_topo(conf={"steps": 10, "source": ["judging"]}))
    assert T.gate_request(MODEL, "video") == T.derived_request(MODEL, "video")


def test_an_unknown_gate_key_and_half_a_size_are_refused(cache):
    cache(_topo(gate={"stpes": 1, "source": SRC}))
    with pytest.raises(ValueError, match="stpes"):
        T.gate_request(MODEL, "video")
    cache(_topo(gate={"height": 480, "source": SRC}))
    with pytest.raises(ValueError, match="height 480 and width None"):
        T.gate_request(MODEL, "video")


def test_the_harness_cell_runs_the_gate_request(cache, monkeypatch):
    import regression_matrix as RM
    monkeypatch.setattr(RM.Z, "family_of", lambda m: "video")
    cache(_topo(gate={"steps": 1, "num_frames": 9, "source": SRC}))
    assert RM.cell_request(MODEL) == T.gate_request(MODEL, "video")


def test_the_census_covers_the_gate_request_beside_the_judging_one(cache):
    import certified_census as CC
    cache(_topo(gate={"steps": 1, "num_frames": 9, "source": SRC}))
    reqs = CC.census_requests(MODEL, "video", [], None)
    assert reqs == [T.derived_request(MODEL, "video"), T.gate_request(MODEL, "video")]
    cache(_topo())
    assert CC.census_requests(MODEL, "video", [], None) == [T.derived_request(MODEL, "video")]
