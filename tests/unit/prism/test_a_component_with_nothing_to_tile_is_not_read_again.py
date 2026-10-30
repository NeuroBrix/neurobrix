"""A component with nothing Prism can tile is declined from the graph the container holds, never re-read from disk.

The spatial tiling read graph.json and profile.json again for every component it was asked about, only to decline an
LLM ("no rank-4 or rank-5 input"): 4.9 s of one core for Qwen3-30B-A3B's graph (2026-10-30). The decline is the same,
said the same way; a component that can be tiled still reads its own copy, as before.

What would this file do if the re-read came back? The json.load counter below fails; if the early decline caught a
component with a spatial input, the second test fails."""
from __future__ import annotations

import json
from types import SimpleNamespace

from neurobrix.core.prism.solver import PrismSolver


def _container(tmp_path, input_shape):
    comp = tmp_path / "components" / "model"
    comp.mkdir(parents=True)
    graph = {"tensors": {"x": {"shape": input_shape, "dtype": "float16"}}, "input_tensor_ids": ["x"], "ops": {}}
    (comp / "graph.json").write_text(json.dumps(graph))
    (comp / "profile.json").write_text("{}")
    return SimpleNamespace(_cache_path=tmp_path, _components={"model": SimpleNamespace(graph=graph)})


def _solver():
    s = PrismSolver.__new__(PrismSolver)
    s._input_config = SimpleNamespace()
    s._stream_declaration = lambda container, name: None
    s._vendor_tiling_declaration = lambda name: None
    s._tiling_decline = lambda name, why: ("declined", why)
    return s


def test_a_graph_without_a_spatial_input_is_declined_without_reading_it_again(tmp_path, monkeypatch):
    reads = []
    real = json.load
    monkeypatch.setattr(json, "load", lambda f, *a, **k: reads.append(f.name) or real(f, *a, **k))
    got = _solver()._spatial_component_tiling(_container(tmp_path, ["s0", 2048]), "model", mem=None, tile_rung_mb=1)
    assert got == ("declined", "its graph has no rank-4 or rank-5 input to tile")
    assert reads == [], f"read again from disk: {reads}"


def test_a_graph_with_a_spatial_input_still_reads_its_own_copy(tmp_path, monkeypatch):
    reads = []
    real = json.load
    monkeypatch.setattr(json, "load", lambda f, *a, **k: reads.append(f.name) or real(f, *a, **k))
    try:
        _solver()._spatial_component_tiling(_container(tmp_path, [1, 3, 64, 64]), "model", mem=None, tile_rung_mb=1)
    except Exception:   # noqa: BLE001 — past the reads, this fake carries none of what the tiling goes on to need
        pass
    assert any(r.endswith("graph.json") for r in reads), "a component that can be tiled no longer read its graph"
