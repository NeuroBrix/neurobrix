"""Every audio codec whose length comes from what the model generated is censused at every length.

The 2026-09-28 zero-miss verification left three audio models with misses on both memory
classes, all of one class — a stage whose length the flow learns only from VALUES:

* chatterbox 32 misses (16 GB): its vocoder's walk (`census.walk_extent`) was wired, but the
  census tool only walked when asked (`--walk-extents`), and the catalogue census never asked;
* openaudio-s1-mini 26 misses (16 GB): the DualAR flow decoded its codec at the one frame count
  the shadow's draws happened to reach;
* orpheus-3b-0.1-ft 22 misses (32 GB), every one in its SNAC codec: the shadow's draws fall
  below `audio_token_start`, no frame survives the filter, and the codec NEVER ran — its census
  recorded 8 keys, all from the prefill.

What each test would do if the code were wrong: a census tool that walked only when asked would
record no walking shadow and the first test fails (seen red: the flag was the only way in); an
autoregressive flow without the walk would run the codec zero times and the second test fails;
a DualAR flow without it would decode at the shadow's one length and the third test fails.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neurobrix.kernels import census

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def shadow(monkeypatch):
    monkeypatch.setitem(census._ACTIVE, "census", True)
    monkeypatch.setenv("NBX_CENSUS_EXTENTS", "1")
    yield
    monkeypatch.setitem(census._ACTIVE, "census", False)


def _tool():
    spec = importlib.util.spec_from_file_location("certified_census_walk_under_test",
                                                  ROOT / "tools" / "certified_census.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_the_census_tool_walks_the_extents_without_being_asked(monkeypatch, tmp_path):
    m = _tool()
    calls = []

    def fake_shadow(model, req, mode, hw, n_dev, timeout, log_dir, rung_mb=0, tag="", walk_extents=False):
        calls.append((mode, tag, rung_mb, walk_extents))
        return {"mode": mode, "rung_mb": rung_mb, "rc": 0, "wall_s": 0.0, "keys": [], "error": ""}

    monkeypatch.setattr(m, "shadow", fake_shadow)
    monkeypatch.setattr(m, "_family", lambda model: "tts")
    monkeypatch.setattr(m, "_graph_sha", lambda model: "0")
    monkeypatch.setattr(m, "frozen_dims", lambda model: [])
    monkeypatch.setattr(m, "_device_count", lambda hw: 1)
    monkeypatch.setattr(m, "_tiling_probe", lambda *a, **k: ["--probe"])
    m.census_model("M", "hw", ["triton", "triton-sequential"], [], [["--prompt", "x"]], 60, tmp_path,
                   rungs=[4096, 8192])
    walking = [c for c in calls if c[3]]
    # the model's own request, at the top rung, in every mode — and nothing else walks
    assert walking == [("triton", "", 8192, True), ("triton-sequential", "", 8192, True)], calls


def _recording_run(ran, key_of):
    def run(n):
        ran.append(n)
        for obs in census._OBSERVERS:
            obs.add(f"k::({key_of(n)},)")
    return run


def test_the_snac_codec_is_walked_although_no_frame_survives_the_shadow(shadow, monkeypatch):
    from neurobrix.triton.flow import autoregressive as ar
    monkeypatch.setattr(ar, "NBXTensor", SimpleNamespace(from_numpy=lambda a: a))
    rv = {}
    frames = []

    def execute(name, method, _):
        frames.append(rv["c0"].shape[1])
        for obs in census._OBSERVERS:
            obs.add(f"codec::({rv['c0'].shape[1]},)")      # an exact extent: every frame count a class
        rv["codec.decoder.output_0"] = "wav"

    h = ar.TritonAutoregressiveHandler.__new__(ar.TritonAutoregressiveHandler)
    h.ctx = SimpleNamespace(pkg=SimpleNamespace(defaults={"audio_output_type": "snac_tokens",
                                                          "audio_token_start": 1000, "vocab_size": 2000}),
                            executors={"codec.decoder": object()},
                            variable_resolver=SimpleNamespace(resolved=rv))
    h._ensure_weights_loaded = lambda name: None
    h._execute_component = execute
    # the shadow's draws: token 1, below the audio range — no frame survives the filter
    h._run_snac_codec_decoder([1] * 40, max_tokens=70)
    assert sorted(set(frames)) == list(range(1, 11)), frames      # 70 tokens = 10 frames, every one met
    assert rv["global.output_audio"] == "wav"


def test_a_live_run_decodes_its_own_frames_once(monkeypatch):
    from neurobrix.triton.flow import autoregressive as ar
    monkeypatch.setattr(ar, "NBXTensor", SimpleNamespace(from_numpy=lambda a: a))
    rv, frames = {}, []
    h = ar.TritonAutoregressiveHandler.__new__(ar.TritonAutoregressiveHandler)
    h.ctx = SimpleNamespace(pkg=SimpleNamespace(defaults={"audio_output_type": "snac_tokens",
                                                          "audio_token_start": 1000, "vocab_size": 2000}),
                            executors={"codec.decoder": object()},
                            variable_resolver=SimpleNamespace(resolved=rv))
    h._ensure_weights_loaded = lambda name: None
    h._execute_component = lambda *a: frames.append(rv["c0"].shape[1])
    h._run_snac_codec_decoder([1000 + i for i in range(21)] + [5], max_tokens=70)
    assert frames == [3]
    h._run_snac_codec_decoder([1, 2, 3], max_tokens=70)             # no frame: nothing runs, as before
    assert frames == [3] and "global.output_audio" not in rv


def test_the_dual_ar_codec_is_walked_over_its_frames(shadow, monkeypatch):
    from neurobrix.triton.flow import dual_ar as da
    monkeypatch.setattr(da, "NBXTensor", SimpleNamespace(from_numpy=lambda a: a))
    rv, seen = {}, []

    class Quantizer:
        def run(self, inputs):
            return {"output": inputs["indices"]}

    def execute(name, method, _):
        t = rv["codec.decoder.x"].shape[2]
        seen.append(t)
        for obs in census._OBSERVERS:
            obs.add(f"codec::({t},)")

    e = da.TritonDualAREngine.__new__(da.TritonDualAREngine)
    e.ctx = SimpleNamespace(executors={"codec.quantizer": Quantizer(), "codec.decoder": object()},
                            variable_resolver=SimpleNamespace(resolved=rv), persistent_mode=True)
    e._ensure_weights_loaded = lambda name: None
    e._execute_component = execute
    codes = np.zeros((1, 9, 3), dtype=np.int64)                     # the shadow reached 3 frames
    e._decode_codes(codes, "model", [{"component": "model"}, {"component": "codec.decoder"}], max_tokens=12)
    assert sorted(set(seen)) == list(range(1, 13)), seen
