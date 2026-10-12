"""The runtime half of the unified expert store (Prism `ComponentAllocation.expert_files`).

On a unified device the routed experts are never loaded: the loader loads the dense part and binds
each expert weight to a private view of its bytes in its shard (`core/runtime/expert_files.py`);
nothing of the component is pinned (`expert_store.holds_expert_files`); the torch store copies on
the device's one queue, without CUDA streams or events (`TorchExpertStore`, non-CUDA mode).

Unit scope: a two-expert synthetic shard on disk and CPU tensors. The run proof (a routed model
on the M4, both engines) is owed separately.
"""
import json
import struct
from types import SimpleNamespace

import pytest
import torch

from neurobrix.core.runtime.expert_files import (
    bind_views, expert_graph_names, expert_loader_keys, locate)
from neurobrix.core.runtime.expert_residency import FUSED_OP, ExpertResidency
from neurobrix.core.runtime.graph.expert_store import TorchExpertStore
from neurobrix.core.strategies.expert_store import holds_expert_files
from neurobrix.nbx.weight_presence import AbsentWeightError

SHAPE = (4, 3)
_SF = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}


def _dag(n=2):
    ids = {p: [f"param::model.layers.0.mlp.experts.{e}.{p}_proj.weight" for e in range(n)]
           for p in ("gate", "up", "down")}
    return {"ops": {"moe::0": {"op_type": FUSED_OP, "attributes": {
        "num_experts": n, "top_k": 1,
        "expert_gate_weight_ids": ids["gate"], "expert_up_weight_ids": ids["up"],
        "expert_down_weight_ids": ids["down"]}}}}


def _tensors(dtype=torch.bfloat16, n=2):
    out = {}
    for e in range(n):
        for i, p in enumerate(("gate", "up", "down")):
            out[f"model.layers.0.mlp.experts.{e}.{p}_proj.weight"] = (
                torch.arange(12, dtype=torch.float32).view(SHAPE) + 100 * e + 10 * i).to(dtype)
    out["model.embed_tokens.weight"] = torch.ones(SHAPE, dtype=dtype)
    return out


def _write_component(root, tensors):
    """A container component as the loaders read it: weights_index.json + one safetensors shard."""
    comp = root / "components" / "language_model"
    (comp / "weights").mkdir(parents=True)
    header, blobs, at = {}, [], 0
    for k, t in tensors.items():
        b = t.contiguous().view(torch.uint8).numpy().tobytes()
        header[k] = {"dtype": _SF[t.dtype], "shape": list(t.shape), "data_offsets": [at, at + len(b)]}
        blobs.append(b)
        at += len(b)
    h = json.dumps(header).encode()
    (comp / "weights" / "shard_0.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    (comp / "weights_index.json").write_text(json.dumps(
        {"tensors": {k: {"shard": "shard_0.safetensors"} for k in tensors}}))
    return root


def _torch_build(m, span, key):
    dt = {v: k for k, v in _SF.items()}[span.dtype]
    n = span.nbytes // torch.empty((), dtype=dt).element_size()
    return torch.frombuffer(m, dtype=dt, count=n, offset=span.offset).view(span.shape)


def test_the_experts_are_views_of_their_stored_bytes_not_loaded_copies(tmp_path):
    tensors = _tensors()
    root = _write_component(tmp_path, tensors)
    keys = expert_loader_keys(_dag(), None, tensors)
    assert "model.embed_tokens.weight" not in keys and len(keys) == 6
    spans = locate(root, "language_model", keys)
    views, maps = bind_views(spans, lambda sf: sf, _torch_build)
    assert len(maps) == 1
    for k in keys:
        assert torch.equal(views[k], tensors[k]), k
        assert views[k].shape == SHAPE and views[k].is_contiguous()


def test_an_expert_stored_in_another_dtype_than_it_computes_in_is_refused_by_name(tmp_path):
    tensors = _tensors(torch.bfloat16)
    root = _write_component(tmp_path, tensors)
    spans = locate(root, "language_model", expert_loader_keys(_dag(), None, tensors))
    with pytest.raises(RuntimeError, match="stored BF16 and this engine computes it as F16"):
        bind_views(spans, lambda sf: "F16", _torch_build)


def test_an_expert_the_load_set_or_the_index_does_not_hold_is_refused(tmp_path):
    tensors = _tensors()
    held = set(tensors) - {"model.layers.0.mlp.experts.1.down_proj.weight"}
    with pytest.raises(AbsentWeightError, match="not in the load set"):
        expert_loader_keys(_dag(), None, held)
    root = _write_component(tmp_path, tensors)
    with pytest.raises(AbsentWeightError, match="no shard in its weights index"):
        locate(root, "language_model", {"model.layers.9.mlp.experts.0.gate_proj.weight"})
    with pytest.raises(RuntimeError, match="no fused routed layer"):
        expert_loader_keys({"ops": {}}, None, tensors)


def test_the_graph_names_reach_the_loader_keys_through_the_binding():
    names = expert_graph_names(_dag())
    binding = {n: "lm." + n for n in names}
    assert expert_loader_keys(_dag(), binding, set(binding.values())) == set(binding.values())


def test_the_first_load_splits_the_experts_off_and_a_later_load_does_not():
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._dag, ex._pending_weight_binding, ex._expert_files = _dag(), None, True
    load_set = set(_tensors())
    dense, mapped = ex._split_expert_files(load_set, None)
    assert dense == {"model.embed_tokens.weight"} and len(mapped) == 6
    assert ex._split_expert_files(load_set, {"x"}) == (load_set, set())
    with pytest.raises(RuntimeError, match="no key set"):
        ex._split_expert_files(None, None)
    ex._expert_files = False
    assert ex._split_expert_files(load_set, None) == (load_set, set())


def test_the_torch_store_copies_on_a_device_without_cuda_streams_and_returns_no_events():
    like = tuple(torch.empty(SHAPE, dtype=torch.bfloat16) for _ in range(3))
    store = TorchExpertStore(ExpertResidency(2), torch.device("cpu"), like)
    tensors = _tensors()
    host = lambda e: tuple(tensors[f"model.layers.0.mlp.experts.{e}.{p}_proj.weight"]
                           for p in ("gate", "up", "down"))
    on_card, events = store.acquire("L0", [0, 1], host)
    assert events == {}
    for e in (0, 1):
        for got, want in zip(on_card[e], host(e)):
            assert torch.equal(got, want) and got.data_ptr() != want.data_ptr()
    assert store.bytes_in == 6 * 12 * 2
    store.release()
    assert store.slots == []


def _alloc(strategy="expert_store", expert_files=True):
    return {"strategy": strategy, "device": "mps:0", "expert_files": expert_files}


def test_only_a_unified_expert_store_is_held_from_the_files():
    assert holds_expert_files({"lm": _alloc()}, "lm")
    assert not holds_expert_files({"lm": _alloc(expert_files=False)}, "lm")
    assert not holds_expert_files({"lm": _alloc("zero3")}, "lm")
    assert not holds_expert_files({"lm": ("mps:0", {})}, "lm")


@pytest.mark.parametrize("brick", ["zero3", "lazy_sequential"])
def test_nothing_of_a_unified_expert_store_is_pinned(brick, monkeypatch):
    if brick == "zero3":
        from neurobrix.core.strategies.zero3 import Zero3Strategy as S
    else:
        from neurobrix.core.strategies.lazy_sequential import LazySequentialStrategy as S
    s = S.__new__(S)
    s.context = SimpleNamespace(allocations={"lm": _alloc()}, _plan=None)
    view = torch.ones(SHAPE, dtype=torch.bfloat16)
    ex = SimpleNamespace(_weights={"e": view})
    monkeypatch.setattr(torch.Tensor, "pin_memory",
                        lambda self, *a, **k: pytest.fail("a file-backed expert was pinned"))
    s._pin_cpu_weights("lm", ex)
    assert ex._weights["e"] is view
