"""A streamed HEAD component the flow reads by name is held on its base and reserved by Prism; a
head that holds no weight is refused, never replaced by the token embedding.

Measured 2026-10-08 on the M4 Pro, GLM-4.1V-9B-Thinking native (a7858de7, the Mac's own plan:
layer_streaming at the 4 096 MB rung, 'model.language_model' 7 segments, 'lm_head' 1 segment):
the output was " avoidance//-----", byte-identical before and after the KV and piece-binding
fixes; MiniCPM-o the same. Both VLM handlers compute the logits OUTSIDE the graph, from the
2-D weight their `head_component`'s executor holds (`_compute_logits`). Streamed, that base held
nothing: the one rule saying which components a flow reads by name (`flow_embeds_into`: the graph
takes `inputs_embeds`) does not see a head, whose graph takes hidden states. `_compute_logits`
found no 2-D weight and fell through to the token EMBEDDING — GLM's head is not tied — and decoded
151 552-wide garbage (`stream_defect_2026_10_08/glm.probe.log`: "head weights {}").

The oracle reads the head's name from the container's topology and its bytes from its index here.

Run: PYTHONPATH=src python -m pytest tests/unit/prism/test_a_streamed_head_the_flow_reads_is_held_on_its_base.py
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, container_root, impose_rung,
                                              pin_host, profile)

MODEL = "GLM-4.1V-9B-Thinking"


def _head_of(root):
    return json.loads((root / "topology.json").read_text())["flow"]["vlm"]["head_component"]


def test_the_plan_reserves_the_weights_of_a_streamed_head(monkeypatch):
    from neurobrix.core.prism import InputConfig, PrismSolver
    from neurobrix.core.prism.solver import _graph_constant_bytes
    from neurobrix.nbx import NBXContainer
    pin_host(monkeypatch, 24576, 17667, "the Mac's reading, 2026-10-08 10:18")
    impose_rung(monkeypatch, 4096)
    monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    root = container_root(MODEL)
    head = _head_of(root)
    s = PrismSolver()
    p = s.solve_smart(NBXContainer.load(str(root)), profile(APPLE_M4_PRO),
                      InputConfig(batch_size=1), mode="compiled")
    assert p.strategy == "layer_streaming" and head in p.layer_stream_plan, (
        f"precondition: the Mac's plan streams {head!r} ({p.strategy!r}, {list(p.layer_stream_plan or {})})")
    graphs = {c: json.loads((root / "components" / c / "graph.json").read_text())
              for c in p.layer_stream_plan}
    read = [c for c in graphs if c == head or "input::inputs_embeds" in graphs[c]["input_tensor_ids"]]
    from neurobrix.triton.weight_loader import is_block_key
    expected = 0
    for c in read:
        tensors = json.loads((root / "components" / c / "weights_index.json").read_text())["tensors"]
        expected += sum(int(v["size_bytes"]) for k, v in tensors.items() if not is_block_key(k))
    # mode="compiled" on an Apple GPU: the DtypeEngine stores no fp64 there (the profile's
    # precision.supports_fp64), the answer the solver priced its constants by.
    reserved = s._layer_stream_constant_bytes - sum(_graph_constant_bytes(g, stores_fp64=False)
                                                    for g in graphs.values())
    assert reserved == expected, (
        f"{reserved / 2**20:.1f} MB reserved beside the pieces, {expected / 2**20:.1f} MB the flow "
        f"reads by name from {read}")


def _strategy(topology, root="/nowhere"):
    from neurobrix.core.strategies.base import StrategyContext
    from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy
    ctx = StrategyContext(strategy_name="layer_streaming", allocations={}, component_executors={},
                          topology=topology, runtime_package=SimpleNamespace(cache_path=root))
    return LayerStreamingStrategy(ctx, "layer_streaming")


class _Base:
    """A streamed base executor: its graph takes hidden states, not `inputs_embeds`."""
    def __init__(self):
        self._dag = {"input_tensor_ids": ["input::hidden_states"]}
        self.loaded = None

    def non_block_keys(self, nbx_path, component):
        return {"weight"}

    def load_flow_read_weights(self, nbx_path, component, shard_map=None, keys=None):
        self.loaded = set(keys)
        return len(keys)


def test_the_strategy_holds_the_flow_s_head_on_its_base():
    base = _Base()
    _strategy({"flow": {"vlm": {"lm_component": "lm", "head_component": "lm_head"}}}
              )._ensure_flow_reads("lm_head", base)
    assert base.loaded == {"weight"}, base.loaded


def test_a_component_no_flow_names_still_holds_nothing():
    base = _Base()
    _strategy({"flow": {"vlm": {"lm_component": "lm", "head_component": "lm_head"}}}
              )._ensure_flow_reads("vae", base)
    assert base.loaded is None, base.loaded


class _Hidden:
    ndim = 3

    def __getitem__(self, _):
        return self


def _engine(weights):
    head = SimpleNamespace(_weights=weights)
    return SimpleNamespace(ctx=SimpleNamespace(executors={"lm_head": head}),
                           _ensure_weights_loaded=lambda name: None)


@pytest.mark.parametrize("module,cls", [("neurobrix.core.flow.vlm", "VLMEngine"),
                                        ("neurobrix.triton.flow.vlm", "TritonVLMEngine")])
def test_a_head_that_holds_no_weight_is_refused_not_replaced_by_the_embedding(module, cls):
    import importlib
    engine_cls = getattr(importlib.import_module(module), cls)
    with pytest.raises(RuntimeError, match="lm_head"):
        engine_cls._compute_logits(_engine({}), _Hidden(), object(), "lm_head", "lm_head")
