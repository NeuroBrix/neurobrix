"""The compiled loader loads what the plan budgeted: the weights the graph or
the flow reads, not every weight in the shard.

2026-09-13: Prism sizes a component by the weights its graph consumes (the
DeepSeek lesson of 2026-09-09: 11.8 GB of unrouted MoE experts out of 30.6),
and the solver's comment said "the engine skips those". The TRITON loader did;
the compiled loader read every key of every shard. Four native cells of the
suite — Qwen3-Omni, Qwen3-VL, Qwen3-Coder (30B MoE) and DeepSeek-Coder-V2-Lite
— were planned single_gpu on one 32 GB card at 5-19 GB and died loading 30-57 GB
onto it, on a quiet rig, from a frozen tree. A plan is budgeted under the
memory model it is executed under; here the two engines had two. Register 53.

Run: PYTHONPATH=src python -m pytest tests/unit/core/test_compiled_loader_loads_what_the_plan_budgeted.py
"""
import torch
from safetensors.torch import save_file

from neurobrix.core.io.weight_loader import WeightLoader


def _loader(only):
    ld = WeightLoader.__new__(WeightLoader)      # no container: the seam under test is the file reader
    ld._only = only
    ld.use_cache = False
    return ld


def test_a_safetensors_shard_yields_only_the_wanted_keys(tmp_path):
    f = tmp_path / "shard_000.safetensors"
    save_file({"block.0.attn.key.weight": torch.ones(2, 2), "token_embed.weight": torch.ones(3, 2),
               "block.0.mlp.experts.7.down.weight": torch.ones(2, 2)}, str(f))
    got = _loader({"block.0.attn.key.weight", "token_embed.weight"})._load_with_pinned_dma(str(f), "cpu", None, False)
    assert set(got) == {"block.0.attn.key.weight", "token_embed.weight"}, set(got)


def test_none_still_means_every_key(tmp_path):
    f = tmp_path / "shard_000.safetensors"
    save_file({"a.weight": torch.ones(1), "b.weight": torch.ones(1)}, str(f))
    assert set(_loader(None)._load_with_pinned_dma(str(f), "cpu", None, False)) == {"a.weight", "b.weight"}


def test_a_pytorch_shard_is_filtered_the_same_way(tmp_path):
    f = tmp_path / "shard_000.bin"
    torch.save({"a.weight": torch.ones(1), "b.weight": torch.ones(1)}, str(f))
    got = _loader({"a.weight"})._load_pytorch_with_pinned_dma(str(f), "cpu", None, False)
    assert set(got) == {"a.weight"}, set(got)


def test_the_compiled_load_path_hands_the_loader_the_filtered_set(tmp_path, monkeypatch):
    # The plumbing, not the reader: _load_weights_native computes the set from
    # the graph and the index and passes it as only= (review, 2026-09-13).
    import json
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    import neurobrix.core.io as io_mod
    seen = {}

    class FakeLoader:
        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def load_component(self, component, device, dtype, only=None):
            seen["only"] = only; return {k: torch.ones(1) for k in only}
        def load_component_with_shard_map(self, component, shard_map, dtype, only=None):
            seen["only"] = only; return {}
    monkeypatch.setattr(io_mod, "WeightLoader", FakeLoader)
    ex = GraphExecutor.__new__(GraphExecutor)
    ex.device = "cpu"
    monkeypatch.setattr(ex, "_placement_torch_dtype", lambda: torch.float32, raising=False)
    params = ["block.0.attn.key.weight", "block.1.attn.key.weight", "token_embed.weight"]
    ex._dag = {"tensors": {f"param::{n}": {"is_parameter": True, "weight_name": n} for n in params},
               "ops": {"op0": {"input_tensor_ids": ["param::block.0.attn.key.weight"]}},
               "execution_order": ["op0"]}
    comp = tmp_path / "components" / "c"; comp.mkdir(parents=True)
    (comp / "weights_index.json").write_text(json.dumps({"tensors": {k: {} for k in params}}))
    ex._load_weights_native(str(tmp_path), "c", None)
    assert seen["only"] == {"block.0.attn.key.weight", "token_embed.weight"}, seen
    assert set(ex._weights) == seen["only"]
