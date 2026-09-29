"""The census's tiling probe is the one the CONTAINER carries, the family profile's only when it carries none.

The owner's method (2026-09-29 13:45): no size in a tool's code, no per-model branch. The family
constant (video 720x1280) asked Wan2.1-VACE-1.3B for a size its vendor documents as unsupported;
the build now writes the model's largest documented request, with its source, into topology.json
extracted_values["_global"]["tiling_probe"].

What would this file do if the code were wrong? The container's probe ignored (the family's read
first, or only) -> the first test fails; a container without one given no probe instead of the
family's -> the second; the ordinary request's own size kept beside the probe's -> the third.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))
import certified_census as CC  # noqa: E402

PROBE = {"height": 480, "width": 832, "source": ["https://huggingface.co/x/README.md (L1): 480P"]}
REQUEST = ["--prompt", "a boat", "--height", "240", "--width", "416", "--num-frames", "33"]


def _container(tmp_path, name, global_values):
    root = tmp_path / name
    root.mkdir()
    (root / "topology.json").write_text(json.dumps({"extracted_values": {"_global": global_values}}))


def test_the_container_s_probe_comes_first(tmp_path, monkeypatch):
    monkeypatch.setattr(CC, "CACHE", tmp_path)
    _container(tmp_path, "M", {"dominant_dtype": "bfloat16", "tiling_probe": PROBE})
    assert CC.tiling_probe_spec("M", "video") == (PROBE, "container")
    req = CC._tiling_probe("M", "video", REQUEST, tmp_path)
    assert req[-4:] == ["--height", "480", "--width", "832"]


def test_a_container_without_one_takes_the_family_s(tmp_path, monkeypatch):
    monkeypatch.setattr(CC, "CACHE", tmp_path)
    _container(tmp_path, "M", {"dominant_dtype": "bfloat16"})
    spec, origin = CC.tiling_probe_spec("M", "video")
    assert origin == "family" and spec.get("height") and spec.get("width")
    assert CC.tiling_probe_spec("M", "llm") == ({}, None)
    assert CC._tiling_probe("M", "llm", REQUEST, tmp_path) is None


def test_the_probe_replaces_the_request_s_own_size(tmp_path, monkeypatch):
    monkeypatch.setattr(CC, "CACHE", tmp_path)
    _container(tmp_path, "M", {"tiling_probe": PROBE})
    req = CC._tiling_probe("M", "video", REQUEST, tmp_path)
    assert req.count("--height") == 1 and req.count("--width") == 1
    assert req[:2] == ["--prompt", "a boat"] and "--num-frames" in req and "33" in req
