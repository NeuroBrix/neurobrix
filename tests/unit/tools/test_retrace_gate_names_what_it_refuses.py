"""The gate names an op or tensor difference it refuses — never a bare count."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_zoo as R  # noqa: E402
import precision_zoo_campaign as C  # noqa: E402

S0 = {"type": "symbol", "id": "s0", "trace": 23}


def _graph(alpha):
    return {"symbolic_context": {"symbols": {"s0": {"name": "s0", "trace_value": 23}}},
            "ops": [{"op_uid": "aten.softmax::0", "op_type": "aten.softmax", "input_tensor_ids": ["x"], "output_tensor_ids": ["y"],
                     "attributes": {"args": [{"type": "tensor", "tensor_id": "x"}], "kwargs": {}, "dim": -1, "alpha": alpha}}],
            "tensors": {"x": {"shape": [1, 23], "dtype": "float16", "symbolic_shape": {"dims": [1, S0], "concrete": [1, 23]}, "consumer_op_uids": ["aten.softmax::0"]},
                        "y": {"shape": [1, 23], "dtype": "float16", "symbolic_shape": {"dims": [1, S0], "concrete": [1, 23]}, "consumer_op_uids": []}},
            "outputs": ["y"]}


def test_a_refused_op_difference_is_named_with_its_path(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "family_of", lambda n: "tts")
    monkeypatch.setattr(R, "CACHE", tmp_path / "cache")
    args = types.SimpleNamespace(out=str(tmp_path / "out"), backup=str(tmp_path / "backup"), models_root=str(tmp_path / "b"),
                                 tmp=str(tmp_path / "t"), gpu=None, src=None, extra=[], timeout=1, trace_timeout=1, restore_mbps=1.0, upload_mbps=0.0)
    m = R.Model("m", args)
    for root, alpha in ((Path(args.backup) / "m", 1.0), (R.CACHE / "m", 2.0)):
        (root / "components" / "core").mkdir(parents=True)
        (root / "components" / "core" / "graph.json").write_text(json.dumps(_graph(alpha)))
    rep = m.graph_diff()
    core = rep["components"]["core"]
    assert core["op_diffs"] == 1 and rep["beyond_annotation"] == 1
    site = core["op_diff_sites"][0]
    assert site["op"] == "aten.softmax::0" and site["diffs"] == [{"path": "attributes.alpha", "old": "1.0", "new": "2.0"}]


def test_a_tensor_only_one_graph_carries_is_named(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "family_of", lambda n: "tts")
    monkeypatch.setattr(R, "CACHE", tmp_path / "cache")
    args = types.SimpleNamespace(out=str(tmp_path / "out"), backup=str(tmp_path / "backup"), models_root=str(tmp_path / "b"),
                                 tmp=str(tmp_path / "t"), gpu=None, src=None, extra=[], timeout=1, trace_timeout=1, restore_mbps=1.0, upload_mbps=0.0)
    m = R.Model("m", args)
    g_old, g_new = _graph(1.0), _graph(1.0)
    g_new["tensors"]["z"] = {"shape": [1], "dtype": "float16", "symbolic_shape": {"dims": [1], "concrete": [1]}, "consumer_op_uids": []}
    for root, g in ((Path(args.backup) / "m", g_old), (R.CACHE / "m", g_new)):
        (root / "components" / "core").mkdir(parents=True)
        (root / "components" / "core" / "graph.json").write_text(json.dumps(g))
    core = m.graph_diff()["components"]["core"]
    assert core["tensor_diffs_beyond"] == 1 and core["tensor_diff_sites"] == [{"tensor": "z", "only_in": "new"}]


def test_a_topology_whose_routing_changed_is_beyond_the_annotation(tmp_path, monkeypatch):
    """VibeVoice: flow type `next_token_diffusion` → `audio`, the graph gate silent."""
    monkeypatch.setattr(C, "family_of", lambda n: "tts")
    monkeypatch.setattr(R, "CACHE", tmp_path / "cache")
    args = types.SimpleNamespace(out=str(tmp_path / "out"), backup=str(tmp_path / "backup"), models_root=str(tmp_path / "b"),
                                 tmp=str(tmp_path / "t"), gpu=None, src=None, extra=[], timeout=1, trace_timeout=1, restore_mbps=1.0, upload_mbps=0.0)
    m = R.Model("m", args)
    for root, ftype, comps in ((Path(args.backup) / "m", "next_token_diffusion", ["core"]), (R.CACHE / "m", "audio", ["core", "model"])):
        (root / "components" / "core").mkdir(parents=True)
        (root / "components" / "core" / "graph.json").write_text(json.dumps(_graph(1.0)))
        (root / "topology.json").write_text(json.dumps({"flow": {"type": ftype, "order": ["core"]}, "connections": [{"from": "global.x", "to": "core.x"}],
                                                        "synthesis": {}, "components": {c: {"type": "neural_component"} for c in comps}}))
    rep = m.graph_diff()
    assert rep["components"]["core"]["op_diffs"] == 0
    paths = [x["path"] for x in rep["topology"]]
    assert "flow.type" in paths and "components.model" in paths and rep["beyond_annotation"] == 2
    (R.CACHE / "m" / "topology.json").write_text((Path(args.backup) / "m" / "topology.json").read_text())
    assert m.graph_diff()["topology"] == []
