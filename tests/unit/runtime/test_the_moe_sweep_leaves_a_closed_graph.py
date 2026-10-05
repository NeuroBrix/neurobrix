"""The MoE fusion's dead-op sweep leaves a closed graph at its fixed point.

`_fuse_one_moe_layer` replaces a layer's expert ops by one custom::moe_fused op, then sweeps twice:
the ops that read a tensor it removed (input side), and the ops left with no live reader (output side).
Both sweeps are worklists over the shared consumer map (2026-10-05): the whole-order rescans they
replaced made Qwen3-30B-A3B's fusion 48.9 s per graph, read twice per plan; the worklists give 12.8 s
and the byte-identical fused DAG on every MoE graph of the cache (18 of 18, both `declared` values).

What a wrong sweep does, and which assert sees it:
  * too timid (an op whose outputs nobody reads survives) -> the fixed point fails;
  * too greedy (an op whose output a survivor or the graph's outputs read is removed) -> a survivor
    reads a tensor nothing produces, or a graph output is never produced.
The two cascades (a removed op's outputs seed more removals; a removed op re-queues its producers)
are what make the result independent of the visiting order. With the execution order topological and
the sweep visiting it backwards, the set reached is the same without them, so they are guarded here by
the result they must reach, not by a run that only they could change.

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_the_moe_sweep_leaves_a_closed_graph.py
"""
from __future__ import annotations

import copy
import json

import pytest

from neurobrix.core.paths import cache_dir
from neurobrix.core.runtime.graph import moe_fusion as MF

from test_a_stacked_expert_block_routed_softmax_first_is_fused import qwen3vl_block


def _protected(op):
    parent = op.get("parent_module", "")
    return op.get("op_type") == "custom::moe_fused" or "shared" in parent


def assert_closed(before, after):
    """`after` (fused) reads only what it produces or what `before` never produced, produces every
    graph output, and keeps no unprotected op whose outputs nobody reads."""
    ops, order = after["ops"], after["execution_order"]
    produced_before = {t for op in before["ops"].values() for t in op.get("output_tensor_ids", [])}
    outputs = set(after.get("output_tensor_ids", []))
    live, readers = set(), {}
    for uid in order:
        op = ops[uid]
        for t in MF._collect_input_tids(op):
            readers.setdefault(t, set()).add(uid)
            if t in produced_before and t not in live:
                raise AssertionError(f"{uid} ({op['op_type']}) reads {t}, which no surviving op produces")
        live.update(op.get("output_tensor_ids", []))
    missing = sorted(t for t in outputs if t in produced_before and t not in live)
    assert not missing, f"graph outputs no surviving op produces: {missing[:5]}"
    unread = []
    for uid in order:
        op = ops[uid]
        outs = op.get("output_tensor_ids", [])
        if not outs or _protected(op):
            continue
        if all(t not in outputs and not (readers.get(t, set()) - {uid}) for t in outs):
            unread.append(f"{uid} ({op['op_type']})")
    assert not unread, f"the sweep stopped short of its fixed point; unread: {unread[:5]}"


def _fused(dag, family="llm", **kw):
    d = MF.detect_and_fuse_moe(copy.deepcopy(dag), family, **kw)
    assert any(d["ops"][u]["op_type"] == "custom::moe_fused" for u in d["execution_order"])
    return d


@pytest.mark.parametrize("renorm", [True, False])
def test_the_stacked_path_leaves_a_closed_block_too(renorm):
    dag = qwen3vl_block(renorm=renorm)
    assert_closed(dag, _fused(dag))


def _cached(model, comp):
    p = cache_dir() / model / "components" / comp / "graph.json"
    if not p.exists():
        pytest.skip(f"{model} is not in the cache")
    return json.loads(p.read_text())


def _add_tail(dag, name, src, n=2):
    """`n` ops chained off tensor `src`, read by nobody, placed right after `src`'s producer."""
    at = next(i for i, u in enumerate(dag["execution_order"])
              if src in dag["ops"][u].get("output_tensor_ids", [])) + 1
    for i in range(n):
        uid, tin, tout = f"{name}.{i}", (src if i == 0 else f"{name}{i - 1}"), f"{name}{i}"
        dag["tensors"][tout] = dict(dag["tensors"][src], tensor_id=tout)
        dag["ops"][uid] = {"op_uid": uid, "op_type": "aten::neg", "input_tensor_ids": [tin],
                           "output_tensor_ids": [tout], "parent_module": "tail",
                           "attributes": {"args": [{"type": "tensor", "tensor_id": tin}], "kwargs": {}}}
        dag["execution_order"].insert(at + i, uid)
    return [f"{name}.{i}" for i in range(n)]


def test_a_dead_tail_two_ops_long_is_swept_whole():
    """deepseek-moe-16b-chat fuses through the per-expert path, the one these sweeps serve. A tail off
    the first router's top-k indices (a tensor the fusion removes: the input-side sweep) and a tail off
    the embedding (a tensor that stays: the output-side sweep) both go whole, not just their last op."""
    dag = _cached("deepseek-moe-16b-chat", "model")
    tails = (_add_tail(dag, "tail_removed", "aten.topk::0::out_1")
             + _add_tail(dag, "tail_kept", "aten.embedding::0::out_0"))
    d = _fused(dag, declared=True)
    assert not set(tails) & set(d["execution_order"]), set(tails) & set(d["execution_order"])
    assert_closed(dag, d)


@pytest.mark.parametrize("model,comp", [("deepseek-moe-16b-chat", "model"),
                                        ("Qwen3-Omni-30B-A3B-Instruct", "talker.model")])
def test_a_cached_moe_graph_is_closed_after_fusion(model, comp):
    dag = _cached(model, comp)
    assert_closed(dag, _fused(dag, declared=True))
