"""The granite-style MoE block fuses instead of replaying trace-time routing.

granitemoe traces topk BEFORE softmax, dispatches by sorted index through
split_with_sizes whose sizes are BAKED at trace time, and stores its experts
STACKED ([E, 2F, H] input_linear, [E, H, F] output_linear). Replayed as
traced, any prompt that routes differently than the trace dies at the baked
sizes — measured: "split_with_sizes expects split_sizes to sum exactly to
512 ... got [0, 6, 0, 23, ...]" (sum 184, the trace prompt's routing).

This test drives the REAL traced graph when the container is in the local
cache (machine-gated skip otherwise, same policy as the reshape-rung cells)
and asserts the rewrite: every router fused, zero split_with_sizes left,
one inserted softmax per layer (granite's softmax-after-topk equals
softmax-then-topk renormalized, exactly), producers before consumers.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from neurobrix.core.paths import cache_dir

_GRAPH = Path(cache_dir()) / "granite-3.1-1b-a400m-instruct" / "components" / \
    "model" / "graph.json"


@pytest.fixture(scope="module")
def granite_dag():
    if not _GRAPH.exists():
        pytest.skip("granite-3.1-1b-a400m-instruct is not in this machine's "
                    "cache — the pattern is proven where the trace lives")
    return json.loads(_GRAPH.read_text())


def _fused(dag):
    from neurobrix.core.runtime.graph.moe_fusion import detect_and_fuse_moe
    return detect_and_fuse_moe(copy.deepcopy(dag), family="llm")


def test_every_router_fuses_and_no_baked_split_survives(granite_dag):
    d2 = _fused(granite_dag)
    ops = d2["ops"]
    routers = sum(1 for o in granite_dag["ops"].values()
                  if o.get("op_type") == "aten::topk")
    fused = [o for o in ops.values()
             if o.get("op_type") == "custom::moe_fused"]
    splits = [o for o in ops.values()
              if o.get("op_type") == "aten::split_with_sizes"]
    softmaxes = [u for u in d2["execution_order"]
                 if u.startswith("moe_softmax::")]
    assert len(fused) == routers, (len(fused), routers)
    assert not splits, f"{len(splits)} baked splits survived the rewrite"
    assert len(softmaxes) == routers

    a = fused[0]["attributes"]
    st = a["stacked_experts"]
    assert a["norm_topk_prob"] is True
    assert a["top_k"] > 1 and a["num_experts"] > a["top_k"]
    assert st["ffn_dim"] > 0
    assert st["input_linear_tid"] != st["output_linear_tid"]


def test_the_rewritten_order_runs_producers_first(granite_dag):
    d2 = _fused(granite_dag)
    ops = d2["ops"]
    eo = d2["execution_order"]
    pos = {u: i for i, u in enumerate(eo)}
    prod = {}
    for u in eo:
        for t in ops[u].get("output_tensor_ids", []):
            prod[t] = u
    for u in eo:
        for t in ops[u].get("input_tensor_ids", []):
            pu = prod.get(t)
            assert pu is None or pos[pu] < pos[u], (
                f"{u} consumes {t} produced later by {pu}")
