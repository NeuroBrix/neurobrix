"""A streamed LM stage is cut on the graph Prism cut, and its pieces compute what it computes whole.

The Mac counted 26 streamed runs refused while building the pieces (df2588e7), six models, both
triton engines:

    layer_streaming: the plan's segment boundaries are not in 'language_model's graph
    (1 of 14 op ids absent, e.g. 'custom.swiglu_fused::18')

Two causes, measured on this rack 2026-09-24:
  * Prism cuts `normalize_for_branch(graph)`, which folds silu+mul into `custom.swiglu_fused`;
    the strategy checked the base executor's graph, and a streamed base holds no weights and never
    compiles, so its graph is the one loaded. granite-speech at 4 096 MB: 16 boundary ids, 2 on a
    fused op, those 2 absent, 0 fused ops in the base graph. The numbering itself agrees (40/40,
    30/30, 40/40, 36/36 pairs on the four dense LMs once compiled) — it is WHICH graph, not how.
    And `triton_sequential` never fuses at all: its plan was cut on a graph it does not run.
  * Qwen3-Omni's thinker is a MoE LM under family `multimodal`: its flow declares it
    (`lm_config.num_experts > 1`, `set_moe_config`) and the runtime fuses its experts (12 132 ops
    -> 4 300); Prism's normalisation did not. (Qwen3-VL's stacked experts are not fused by that
    pass; its 48 absent of 192 were the first cause.) A presence check could not see this one:
    the 2-piece plan's boundaries sat on untouched ops. The plan now carries its declaration and
    the FINGERPRINT of the graph it cut; the strategy cuts with that declaration and refuses any
    other graph, at the cut and before the first run — the door these cells go through, in
    PRODUCTION ORDER (the flow declares after the pieces exist).

Two kinds of cell:
  * `test_every_plan_boundary_is_in_the_graph_the_strategy_cuts` — no weights: plan under the
    Mac's reading at the Mac's rungs, build the base executor the way the runtime builds it (load,
    plus the flow's MoE declaration from the same `lm_config` data), and build the pieces through
    the real `LayerStreamingStrategy`.
  * `test_the_pieces_compute_what_the_whole_computes` — the dish: granite-speech's LM (a SwiGLU
    stack, so a boundary CAN land on a fused op — the T5 of the first executed gate has none)
    whole vs its planned pieces, bit for bit, at the trace size (1, 23) and two others, one far.

Needs a card: `CUDA_VISIBLE_DEVICES=<n>` (the executors are built on it). Without one this FAILS.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "streamed_component_vs_whole.py"

# (model, component, mode, rung MB): the Mac's rows (df2588e7), both triton engines.
BOUNDARY_CELLS = [
    (m, c, mode, r)
    for m, c, rungs in [
        ("granite-speech-3.3-8b", "language_model", (4096, 6144, 12288)),
        ("Janus-Pro-7B", "language_model", (6144,)),
        ("GLM-4.1V-9B-Thinking", "model.language_model", (6144, 12288)),
        ("MiniCPM-o-4_5", "llm.model", (8192, 12288)),
        ("Qwen3-VL-30B-A3B-Thinking", "model.language_model", (11264, 12288)),
        # The MoE LM the runtime fuses on its flow's declaration (12 132 ops -> 4 300, 48
        # `custom::moe_fused`); Qwen3-VL's stacked experts are not fused by that pass (0 -> 0).
        ("Qwen3-Omni-30B-A3B-Instruct", "thinker.model", (16384,)),
    ]
    for r in rungs for mode in ("triton", "triton_sequential")
]

# Rows of the Mac's that this rack's idle-Mac reading (18 186 MB) does not reproduce, named:
#  * Qwen3-VL at 4 096: layer_streaming declines — `aten.bmm::3` reads 768.0 MB of weights on its
#    own, over the 753.8 MB a segment has for weights (the stacked experts); an op cannot be cut;
#  * Qwen3-Omni at 6 144 and 12 288 (it streams at 16 384, the cell above): the plan refuses before streaming — the KV check counts what a
#    plan keeps (fb8ab8f9) and the thinker's residents fill the rung. The Mac's rows predate it.

# (mode, batch, seq): the trace size (1, 23), 64 and far (301), in every engine where the whole
# runs. NOT batch: the whole LM fails at batch 2 in every engine before a piece exists — its
# causal mask `aten.where::0` is declared [23, 23], concrete, and evaluates to [b*s, b*s] (a
# frozen dim of the trace, queued for Forge; owed-proofs).
EXEC_CELLS = ([(m, 1, s) for m in ("triton", "triton_sequential") for s in (23, 64, 301)]
              + [("sequential", 1, 64), ("compiled", 1, 64)])


def _need_card():
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        pytest.fail("this cell builds executors on a card: set CUDA_VISIBLE_DEVICES=<n>. A "
                    "streaming gate that did not run is not a streaming gate that passed.",
                    pytrace=False)


def _plan(model, mode, rung):
    from neurobrix.core.prism import InputConfig, PrismSolver
    from neurobrix.nbx import NBXContainer
    from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, container_root, impose_rung,
                                                  pin_host, profile)
    mp = pytest.MonkeyPatch()
    try:
        pin_host(mp, 24576, 18186, "the Mac, idle")
        impose_rung(mp, rung)
        root = container_root(model)
        # The container's own default request: an LM stage beside an image generator (Janus)
        # cannot be planned without the image size its other components declare.
        dj_path = root / "runtime" / "defaults.json"
        dj = json.loads(dj_path.read_text()) if dj_path.is_file() else {}
        kw = {k: dj[k] for k in ("height", "width") if isinstance(dj.get(k), int)}
        p = PrismSolver().solve_smart(NBXContainer.load(str(root)), profile(APPLE_M4_PRO),
                                      InputConfig(batch_size=1, **kw), mode=mode)
    finally:
        mp.undo()
    return root, p


@pytest.mark.slow
@pytest.mark.parametrize("model,comp,mode,rung", BOUNDARY_CELLS,
                         ids=[f"{m}-{e}-{r}" for m, _, e, r in BOUNDARY_CELLS])
def test_every_plan_boundary_is_in_the_graph_the_strategy_cuts(model, comp, mode, rung):
    _need_card()
    from types import SimpleNamespace
    from neurobrix.core.prism.autodetect import load_default_profile
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    from neurobrix.core.strategies.base import StrategyContext
    from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy

    root, plan = _plan(model, mode, rung)
    bounds = (plan.layer_stream_plan or {}).get(comp)
    assert plan.strategy == "layer_streaming" and bounds, (
        f"{model} at {rung} MB planned {plan.strategy!r} without streaming {comp}: the Mac's row "
        f"is not reproduced, so this cell proves nothing")
    family = json.loads((root / "manifest.json").read_text())["family"]
    here = load_default_profile().devices[0]
    base = GraphExecutor(family=family, vendor=str(here.brand).split(".")[-1].lower(),
                         arch=str(here.architecture), device=here.get_device_string(),
                         dtype=plan.components[comp].dtype, mode=mode)
    base.load_graph(root / "components" / comp / "graph.json")
    # PRODUCTION ORDER. The vlm flows load the LM (which builds its pieces under this strategy)
    # BEFORE they declare its MoE (`set_moe_config`, core/triton flow/vlm.py) — so the pieces are
    # cut here with no declaration made, exactly as at runtime (the guardian, 2026-09-25: a first
    # form of this cell declared first and was green on an order production never runs).
    ctx = StrategyContext(strategy_name="layer_streaming", allocations={comp: (base.device, {})},
                          component_executors={comp: base},
                          runtime_package=SimpleNamespace(cache_path=root),
                          layer_segments={comp: bounds},
                          layer_graphs=dict(plan.layer_stream_graph),
                          layer_moe=dict(plan.layer_stream_moe))
    strategy = LayerStreamingStrategy(ctx, "layer_streaming")
    pieces = strategy._build_segment_executors(comp)
    assert len(pieces) == len(bounds) >= 2, (len(pieces), len(bounds))
    # Then the flow declares — as the vlm flow does after loading: `flow.vlm.lm_component` with
    # `lm_config.num_experts > 1`, read here from the container directly, not through Prism's
    # rule — and the strategy checks the cut again before the first run. A plan that did not
    # declare what the flow declares is refused here (the declared-MoE injection).
    topo = json.loads((root / "topology.json").read_text())
    lm_cfg = json.loads((root / "runtime" / "defaults.json").read_text()).get("lm_config") or {}
    if ((topo.get("flow") or {}).get("vlm") or {}).get("lm_component") == comp \
            and int(lm_cfg.get("num_experts") or 0) > 1:
        base.set_moe_config(norm_topk_prob=bool(lm_cfg["norm_topk_prob"]))
    strategy._graph_prism_cut(comp, base)


@pytest.mark.slow
@pytest.mark.parametrize("mode,batch,seq", EXEC_CELLS, ids=[f"{m}-b{b}-s{s}" for m, b, s in EXEC_CELLS])
def test_the_pieces_compute_what_the_whole_computes(mode, batch, seq, tmp_path):
    _need_card()
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [os.environ["NEUROBRIX_PYTHON"], str(TOOL), "granite-speech-3.3-8b", "language_model", mode,
         str(batch), str(seq), "18186", "4096", str(out)],
        cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        capture_output=True, text=True, timeout=3600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    report = json.loads(out.read_text())
    assert report["pieces"] >= 2, report
    # The size is the one asked for, not the trace's: a dim the harness could not bind from a
    # symbol keeps its traced extent, and the cell would then prove the trace size again.
    assert all(sh[1] == seq for sh in report["inputs"].values() if len(sh) >= 2), report["inputs"]
    assert all(o["whole_shape"][1] == seq for o in report["outputs"].values()), report["outputs"]
    for name, o in report["outputs"].items():
        assert not o.get("missing_from_pieces"), (name, o)
        assert o["whole_vs_whole_max_abs_diff"] == 0.0, (
            f"{name}: the WHOLE component is not reproducible run to run ({o})")
        assert o["bit_identical"], (f"{mode} b{batch} s{seq}: pieces differ from whole: {name} {o}")
