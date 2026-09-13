"""A DAG rewrite that adds readers loads what it now reads.

2026-09-13: Ming-Lite-Omni under --triton, quiet rig, frozen tree. The MoE
fusion of a LM packaged under a non-llm family runs when the flow declares
it, at execute time — after the weights were loaded from the un-fused graph,
in which no op consumes the experts the trace never routed to. The fused
kernel read them all and met None. Register entry 54: after the declared
fusion the executor recomputes the consumed set and loads the difference
through its own loader, in both engines.

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_a_declared_fusion_loads_what_it_now_reads.py
"""
import json

import torch

from neurobrix.core.runtime.graph_executor import GraphExecutor
import neurobrix.core.io as io_mod
import neurobrix.core.runtime.graph.moe_fusion as M


PARAMS = ["block.0.attn.key.weight", "block.0.ffn.expert.0.down.weight",
          "block.0.ffn.expert.1.down.weight", "token_embed.weight"]


def _executor(tmp_path, monkeypatch, mode):
    ex = GraphExecutor.__new__(GraphExecutor)
    ex.mode = mode
    ex.device = "cpu"
    ex.family = "multimodal"
    ex._component_handler = None
    monkeypatch.setattr(ex, "_placement_torch_dtype", lambda: torch.float32, raising=False)
    tensors = {f"param::{n}": {"is_parameter": True, "weight_name": n} for n in PARAMS}
    # the trace routed to expert 0 only
    ex._dag = {"tensors": tensors,
               "ops": {"op0": {"input_tensor_ids": ["param::block.0.attn.key.weight",
                                                    "param::block.0.ffn.expert.0.down.weight"]}},
               "execution_order": ["op0"]}
    comp = tmp_path / "components" / "c"; comp.mkdir(parents=True)
    (comp / "weights_index.json").write_text(json.dumps({"tensors": {k: {} for k in PARAMS}}))
    return ex


class _FakeLoader:
    calls = []
    def __init__(self, path): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def load_component(self, component, device, dtype, only=None):
        _FakeLoader.calls.append(set(only or [])); return {k: torch.ones(1) for k in (only or [])}
    def load_component_with_shard_map(self, component, shard_map, dtype, only=None):
        _FakeLoader.calls.append(set(only or [])); return {k: torch.ones(1) for k in (only or [])}


def _fake_fusion(dag, family, norm_topk_prob=True, declared=False):
    # the declared fusion: one fused op that reads EVERY expert
    dag["ops"]["moe_fused::block.0"] = {
        "op_type": "custom::moe_fused",
        "input_tensor_ids": ["param::block.0.ffn.expert.0.down.weight",
                             "param::block.0.ffn.expert.1.down.weight"]}
    dag["execution_order"].append("moe_fused::block.0")
    return dag


def test_the_declared_fusion_loads_the_experts_it_adds_compiled(tmp_path, monkeypatch):
    _FakeLoader.calls = []
    monkeypatch.setattr(io_mod, "WeightLoader", _FakeLoader)
    monkeypatch.setattr(M, "detect_and_fuse_moe", _fake_fusion)
    ex = _executor(tmp_path, monkeypatch, "compiled")
    ex.load_weights(str(tmp_path), "c", None)
    assert _FakeLoader.calls[0] == {"block.0.attn.key.weight", "block.0.ffn.expert.0.down.weight",
                                    "token_embed.weight"}, _FakeLoader.calls
    assert "block.0.ffn.expert.1.down.weight" not in ex._weights, "unrouted before the fusion"
    ex.set_moe_config(norm_topk_prob=True)
    assert _FakeLoader.calls[1] == {"block.0.ffn.expert.1.down.weight"}, _FakeLoader.calls
    assert "block.0.ffn.expert.1.down.weight" in ex._weights
    assert len(ex._weights) == 4


def test_the_declared_fusion_loads_the_experts_it_adds_triton(tmp_path, monkeypatch):
    # The triton path resolves its device and compute dtype from the plan and
    # needs a card; the seam under test is WHICH KEYS it is asked for and
    # that the second load merges — so the loader call is stood in for.
    calls = []

    def fake_triton_load(self, nbx_path, component, shard_map, only=None):
        keys = only if only is not None else self._consumed_in_loader_space(
            self.consumed_weight_names(), nbx_path, component)
        calls.append(set(keys))
        loaded = {k: object() for k in keys}
        if only is not None:
            self._weights.update(loaded)
        else:
            self._weights = loaded
    monkeypatch.setattr(GraphExecutor, "_load_weights_triton", fake_triton_load)
    monkeypatch.setattr(GraphExecutor, "_load_constants_from_graph", lambda self: None)
    monkeypatch.setattr(M, "detect_and_fuse_moe", _fake_fusion)
    ex = _executor(tmp_path, monkeypatch, "triton")
    ex.load_weights(str(tmp_path), "c", None)
    assert calls[0] == {"block.0.attn.key.weight", "block.0.ffn.expert.0.down.weight", "token_embed.weight"}
    ex.set_moe_config(norm_topk_prob=False)
    assert calls[1] == {"block.0.ffn.expert.1.down.weight"}, calls
    assert len(ex._weights) == 4


def test_nothing_to_load_when_the_fusion_adds_no_reader(tmp_path, monkeypatch):
    _FakeLoader.calls = []
    monkeypatch.setattr(io_mod, "WeightLoader", _FakeLoader)
    monkeypatch.setattr(M, "detect_and_fuse_moe", lambda dag, *a, **k: dag)
    ex = _executor(tmp_path, monkeypatch, "compiled")
    ex.load_weights(str(tmp_path), "c", None)
    ex.set_moe_config(norm_topk_prob=True)
    assert len(_FakeLoader.calls) == 1, "one load, no second"
