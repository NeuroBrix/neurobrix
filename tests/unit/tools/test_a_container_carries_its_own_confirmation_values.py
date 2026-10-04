"""A container's own confirmation values override its family's — the model carries its value.

The supervisor, 2026-10-04 00:33: a model is judged at the smallest request inside its documented
envelope where the vendor's own pipeline gives a correct output. Allegro's vendor pipeline is blank
at the video family's confirmation (4 steps, 256x640) and recognisable at 720x1280 / 10 steps. The
step count is declared for the model in Forge's registry with its source, written by the build (and
the in-place pass) into topology.json's extracted_values["_global"]["confirmation"], and read here
over the family's `confirmation:` section. No model is named in the tool.

The fake container below is a video container traced at 720x1280 (a latent 90x160 under a VAE scale
of 8) — the arithmetic is by hand: the family's size_fraction 0.5 -> 360x640, the height at three
quarters -> 270, on the 32 lattice -> 256x640; under a stated minimum of 720x1280 it moves up to it.

What would this file do if the code were wrong?
  * the container's value ignored -> `--steps` stays the family's: the second and the end-to-end
    tests fail (seen RED, injection: `model_confirmation` returning the family's section);
  * the family's value dropped when the container declares none, or the family's other keys lost
    when it declares one -> the first and second fail;
  * an unknown key passed through as a flag -> the third fails;
  * the citation (`source`) emitted as a request flag -> the end-to-end test fails on `--source`.
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
SRC = ["measured: the vendor's own pipeline is blank at 4 steps and recognisable at 10"]


def _topo(conf=None, env=None):
    g = {}
    if conf is not None:
        g["confirmation"] = conf
    if env is not None:
        g["envelope"] = env
    return {"extracted_values": {"_global": g},
            "components": {"transformer": {"shapes": {"hidden_states": [2, 4, 22, 90, 160]}}}}


def test_the_family_value_is_kept_when_the_container_declares_none():
    fam = T.confirmation("video")
    assert fam["steps"] != 10, "the family's own value must differ from the test's override to judge anything"
    assert T.model_confirmation("video", _topo()) == fam
    assert T.model_confirmation("video", {}) == fam
    assert T.model_confirmation("video", {"extracted_values": {}}) == fam


def test_the_containers_value_wins_and_the_familys_other_values_stay():
    fam = T.confirmation("video")
    got = T.model_confirmation("video", _topo({"steps": 10, "source": SRC}))
    assert got == {**fam, "steps": 10}
    assert got["size_fraction"] == fam["size_fraction"] and "source" not in got


def test_an_unknown_key_is_refused_by_name():
    with pytest.raises(ValueError, match=r"\['stpes'\].*video.*steps"):
        T.model_confirmation("video", _topo({"stpes": 10, "source": SRC}))
    with pytest.raises(ValueError, match="not a mapping"):
        T.model_confirmation("video", _topo(10))


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A cache holding one video container traced at 720x1280; returns a writer of its topology."""
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
        (d / "topology.json").write_text(json.dumps(topology))
    return write


def _flag(req, flag):
    assert req.count(flag) == 1, (flag, req)
    return req[req.index(flag) + 1]


def test_the_derived_request_carries_the_containers_steps_at_its_envelope(cache):
    fam_steps = str(T.confirmation("video")["steps"])
    # declares nothing: the family's steps at the half size
    cache(_topo())
    req = T.derived_request(MODEL, "video")
    assert (_flag(req, "--steps"), _flag(req, "--height"), _flag(req, "--width")) == (fam_steps, "256", "640"), req
    # declares its steps and its envelope: both carried, the citation never a flag
    cache(_topo({"steps": 10, "source": SRC}, {"min": {"height": 720, "width": 1280}, "source": ["card"]}))
    req = T.derived_request(MODEL, "video")
    assert (_flag(req, "--steps"), _flag(req, "--height"), _flag(req, "--width")) == ("10", "720", "1280"), req
    assert "--source" not in req and not any("measured" in str(a) for a in req), req
    # a typo in the carried value stops the derivation, it never reaches a run
    cache(_topo({"stpes": 10, "source": SRC}))
    with pytest.raises(ValueError, match="stpes"):
        T.derived_request(MODEL, "video")


def test_a_model_absent_from_the_cache_is_refused_by_name(cache):
    with pytest.raises(FileNotFoundError, match="Absent-Model"):
        T.container_topology("Absent-Model")
