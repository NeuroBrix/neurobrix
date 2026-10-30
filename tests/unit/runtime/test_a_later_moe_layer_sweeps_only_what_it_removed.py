"""After its first fusion, a MoE layer's dead-op sweep starts from what that layer removed, and kills the same ops.

The sweep visited every op of the order at every layer: 48 x 115 419 visits on Qwen3-30B-A3B, 2.8 M liveness checks,
8-11 s of one core per fusion, a fusion the plan and the load each run (2026-10-30). Once one whole sweep has reached
the fixed point, a later layer can only kill what lost a consumer to it. The fused graphs of the catalogue's eight MoE
models were compared whole, old code against new, declared and by family: identical (fuse_eq, 2026-10-30).

What would this file do if a later sweep missed an op? The fused graph differs from the one every layer sweeping whole
gives, and the equality fails (seen: a later sweep with no seed left 5 538 ops of deepseek-moe instead of 2 683)."""
from __future__ import annotations

import hashlib
import json

import pytest

import neurobrix.core.runtime.graph.moe_fusion as F
from neurobrix.core.optim.passes.normalize import graph_copy
from neurobrix.core.paths import cache_dir

MODEL = "deepseek-moe-16b-chat"      # 18 610 ops, per-expert layers (granite, smaller, takes the stacked matcher)


@pytest.fixture(scope="module")
def graph():
    path = cache_dir() / MODEL / "components" / "model" / "graph.json"
    if not path.is_file():
        pytest.skip(f"{MODEL} is not in this machine's cache")
    return json.loads(path.read_text())


def test_the_fused_graph_is_the_one_every_layer_sweeping_whole_gives(graph, monkeypatch):
    fused = F.detect_and_fuse_moe(graph_copy(graph), "declared", norm_topk_prob=True, declared=True)
    real = F._fuse_one_moe_layer
    calls = []
    monkeypatch.setattr(F, "_fuse_one_moe_layer",
                        lambda *a, **k: calls.append(k.get("whole_sweep")) or real(*a, **{**k, "whole_sweep": True}))
    every_whole = F.detect_and_fuse_moe(graph_copy(graph), "declared", norm_topk_prob=True, declared=True)
    assert calls[0] is True and len(calls) > 1 and not any(calls[1:]), "only the first fusion sweeps every op"
    digest = lambda g: hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()   # noqa: E731
    assert digest(fused) == digest(every_whole), (
        f"the sweep from what a layer removed left {len(fused['execution_order'])} ops, every layer sweeping whole "
        f"{len(every_whole['execution_order'])}")
