"""A streamed piece binds its weights over its COMPONENT's parameters, and takes from its base a
name the flow put there.

Measured 2026-10-08 on the M4 Pro, canary-qwen-2.5b native greedy (f6529de4): whole
"And so my fellow Americans ask not what your country can do", layer_streaming forced
"tigner</act</act</act". The last piece's graph names one parameter, `head.weight`
[151936, 2048]; the index holds no such key (the head is tied: the audio-LLM flow puts the token
embedding on the base under that name). Bound over the piece's own one-name set, the bare suffix
`weight` is unique, so pass 1 of `bind_weight_keys` bound the head to the last key it walked,
`model.block.9.attn.value.weight` [1024, 2048], and the piece's "logits" came out 1 024 wide.
Over the whole component's set the suffix is ambiguous and the head stays unbound, as in the
whole run — the rule `consumed_in_loader_space` states ("`graph_params` is every parameter the
graph names"). Same cut as the precision contract (`_contract_from`).

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_a_piece_binds_its_weights_over_its_component.py
"""
import json

import pytest

from neurobrix.core.runtime.graph_executor import GraphExecutor

INDEX = ["model.embed.weight", "model.norm.weight",
         "model.block.0.attn.value.weight", "model.block.1.attn.value.weight",
         "model.block.1.ffn.up.weight"]


def _executor(params, **attrs):
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._dag = {"tensors": {f"param::{p}": {"shape": [1]} for p in params}}
    for k, v in attrs.items():
        setattr(ex, k, v)
    return ex


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "components" / "llm"
    d.mkdir(parents=True)
    (d / "weights_index.json").write_text(json.dumps({"tensors": {k: {} for k in INDEX}}))
    return tmp_path


def _base_and_last_piece(head):
    whole = [k for k in INDEX if k != "model.embed.weight"] + ["head.weight"]
    base = _executor(whole, _weights={"model.embed.weight": "embed", "model.norm.weight": "norm",
                                      "head.weight": head})
    piece = _executor(["head.weight"], _component_from=base, _borrow_from=base,
                      _flow_reads_weights=False)
    return base, piece


def test_a_one_parameter_piece_binds_no_block_weight_to_a_name_the_index_lacks(root):
    _, piece = _base_and_last_piece("tied")
    wanted = piece._consumed_in_loader_space({"head.weight"}, str(root), "llm")
    assert wanted == set(), wanted
    assert "head.weight" not in (piece._pending_weight_binding or {}), piece._pending_weight_binding


def test_the_piece_takes_the_name_the_flow_put_on_its_base(root):
    tied = object()
    _, piece = _base_and_last_piece(tied)
    wanted = piece._consumed_in_loader_space({"head.weight"}, str(root), "llm")
    keep, taken = piece._borrow(wanted)
    assert keep == set(), keep
    assert taken == {"head.weight": tied}, taken


def test_a_block_piece_still_binds_and_loads_its_own_keys(root):
    base, _ = _base_and_last_piece("tied")
    piece = _executor(["model.block.1.attn.value.weight", "model.block.1.ffn.up.weight"],
                      _component_from=base, _borrow_from=base, _flow_reads_weights=False)
    consumed = {"model.block.1.attn.value.weight", "model.block.1.ffn.up.weight"}
    wanted = piece._consumed_in_loader_space(consumed, str(root), "llm")
    assert wanted == consumed, wanted
    keep, taken = piece._borrow(wanted)
    assert keep == consumed and taken == {}, (keep, taken)
