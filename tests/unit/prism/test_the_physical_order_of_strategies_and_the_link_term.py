"""The engine chooses right (docs/design/prism-fastest-plan-that-fits.md, the owner 2026-10-25 15:16).

(b) The ORDER is physical: a resident plan beats a tiled one beats a plan that streams its weights through the card
beats one that re-reads them per layer per token beats host compute — whatever the penalties inside a rank say.
Seen red before `STRATEGY_RANK`: `zero3` (base 100) and `op_level_tiling` (60) ranked under `lazy_sequential` (300)
and `component_placement_lazy` (400), and `layer_streaming` (50) under both; a zero3 penalty of 200 took it under
`cpu_execution`'s neighbourhood.
(c) The link term prices a device boundary by the profile's bandwidth, never by a PCIe literal: two spread plans over
NVLink (300 GB/s) and over the PCIe fallback (32 GB/s) differ by the bytes / bandwidth term; a pair the profile prices
nothing for refuses the estimate by name.
"""
from __future__ import annotations

import pytest

from types import SimpleNamespace

from neurobrix.core.prism import solver as S
from neurobrix.core.prism import time_estimate as TE
from neurobrix.core.prism.work import Work


class _NoTopology:
    topology = None
    # zero3's host side reads the profile's cores (no fallback since 2026-10-26); 4 = the old implicit default
    cpu = SimpleNamespace(cores=4, compute_threads=4)


def _score(name, devices=("cuda:0",)):
    s = S.PrismSolver.__new__(S.PrismSolver)
    allocs = {"a": devices[0], "b": devices[-1]}
    return s._score_strategy(name, allocs, _NoTopology())


def test_the_ranks_never_cross_whatever_the_penalties():
    order = ["single_gpu", "component_placement", "op_level_tiling", "zero3", "cpu_execution"]
    scores = [_score(n) for n in order]
    assert scores == sorted(scores, reverse=True), dict(zip(order, scores))
    # the host-link rank: zero3's PCIe penalty (200 on a profile whose reference bandwidth is PCIe) stays inside the
    # rank's window — today's score order among them (layer_streaming 50 over a penalised zero3) survives, the
    # estimate decides when the profile carries facts
    for n in ("zero3", "layer_streaming", "lazy_sequential", "component_placement_lazy"):
        assert S.RANK_BASE[4] - S.RANK_FLOOR_SPAN <= _score(n) < S.RANK_BASE[3] - S.RANK_FLOOR_SPAN, (n, _score(n))
    # D1 / D5 (the supervisor, 16:02) against the measurement (16:20): on one card lazy_sequential's plan IS the expert
    # store (Qwen3-30B: dense 2.3 GB resident, 128 experts a layer on the host, the gate's 174 s) and zero3 would stream
    # 56 GB per token (754 s estimated): the host-link strategies share ONE rank and the estimate decides among them
    assert S.STRATEGY_RANK["zero3"] == S.STRATEGY_RANK["lazy_sequential"] == S.STRATEGY_RANK["layer_streaming"]
    assert _score("lazy_sequential") > _score("cpu_execution") and _score("layer_streaming") > _score("cpu_execution")
    # D3: Sana_1600M_4Kpx_BF16 (weights 11.4 GB, activation 23 GB) took layer_streaming / lazy_sequential over tiling
    assert _score("op_level_tiling") > _score("lazy_sequential") and _score("op_level_tiling") > _score("layer_streaming")
    assert _score("op_level_tiling") > _score("zero3")
    # a zero3 penalty never takes it under the lazy rank
    assert _score("zero3") >= S.RANK_BASE[4] - S.RANK_FLOOR_SPAN > S.RANK_BASE[5] + S.RANK_FLOOR_SPAN


def test_the_estimate_decides_inside_a_rank_never_across():
    """D4: the profiles carry measured facts; inside the resident-spread rank the faster plan wins, and a slower
    resident plan still beats a faster streaming one."""
    est = {"block_scatter": 15.3, "pipeline_parallel": 9.0, "lazy_sequential": 15.3, "zero3": 754.1}
    order = S.order_candidates([(796.0, "block_scatter"), (700.0, "pipeline_parallel"), (301.0, "lazy_sequential"), (401.0, "zero3")], est)
    assert [n for _, n in order] == ["pipeline_parallel", "block_scatter", "lazy_sequential", "zero3"]   # Qwen3-30B on one card: lazy, not zero3
    # a tie within 1 % keeps today's order (the score): DeepSeek on four cards, pipeline_parallel 5.1 s = block_scatter 5.1 s
    tie = S.order_candidates([(817.0, "pipeline_parallel"), (802.0, "block_scatter")], {"pipeline_parallel": 5.1, "block_scatter": 5.1})
    assert [n for _, n in tie] == ["pipeline_parallel", "block_scatter"]
    # not estimated: after the estimated ones of its rank, never out of its rank
    order2 = S.order_candidates([(796.0, "block_scatter"), (700.0, "pipeline_parallel")], {"block_scatter": "not estimated: x", "pipeline_parallel": 9.0})
    assert [n for _, n in order2] == ["pipeline_parallel", "block_scatter"]
    # without any estimate the ranks still hold and the score orders inside them
    none = S.order_candidates([(301.0, "lazy_sequential"), (401.0, "zero3"), (1000.0, "single_gpu")], {})
    assert [n for _, n in none] == ["single_gpu", "zero3", "lazy_sequential"]


def test_a_declining_strategy_records_why_once():
    """D2: block_scatter, pipeline_parallel and component_placement declined mochi, Wan2.2-I2V-A14B and Qwen3-Omni on
    four cards with an empty refused(why) in the table."""
    s = S.PrismSolver.__new__(S.PrismSolver)
    assert s._decline("block_scatter", "vae whole (21214 MB with its activation) fits no card beside the scattered blocks") is None
    assert s._decline("block_scatter", "a later reason") is None
    assert s._declined == {"block_scatter": "vae whole (21214 MB with its activation) fits no card beside the scattered blocks"}


def test_the_tiling_rung_names_the_op_it_cannot_band():
    """D3: Sana 4K's transformer overflows at an op the TilingEngine has no band for; the rung says which."""
    class _C:
        name = "transformer"
        graph = {"ops": {"aten.bmm::7": {"op_type": "aten::bmm"}}}
    class _M:
        peak_op_uid = "aten.bmm::7"
    why = S.untiled_reason([("transformer", _M())], [_C()])
    assert "transformer: peak at aten.bmm::7 (aten::bmm)" in why and "no tiled implementation" in why


def test_a_strategy_without_a_rank_is_refused_not_scored():
    with pytest.raises(RuntimeError, match="STRATEGY_RANK"):
        _score("a_rung_nobody_placed")


def _facts():
    # the loader measured at the copy rate: a load costs what the link moves, as these figures were set before the
    # loader rates were measured (`weight_load_gbps`, 2026-10-27)
    rate = {"device": {"cold": 8.1, "warm": 8.1}, "host": {"cold": 8.1, "warm": 8.1}}
    return TE.DeviceFacts(matmul_flops={"float16": 42e12}, memory_bps=789e9, h2d_bps=8.1e9,
                          load_rates={"compiled": {"F16": rate}, "triton": {"F16": rate}})


def test_the_link_term_prices_a_boundary_by_the_profile_s_bandwidth():
    works = {"model": Work(flops=1, weight_bytes=0, activation_bytes=0, output_bytes=600 * 2**20)}
    uses = {"model": 10}
    allocs = {"model": "fgp:cuda:0,cuda:1"}
    nvlink = {("cuda:0", "cuda:1"): 300e9}
    pcie = {("cuda:0", "cuda:1"): 32e9}
    fast = TE.link_seconds("block_scatter", allocs, works, uses, nvlink)
    slow = TE.link_seconds("block_scatter", allocs, works, uses, pcie)
    assert fast == pytest.approx(10 * 600 * 2**20 / 300e9)
    assert slow == pytest.approx(10 * 600 * 2**20 / 32e9) and slow > fast
    # one device: nothing crosses
    assert TE.link_seconds("single_gpu", {"model": "cuda:0"}, works, uses, nvlink) == 0.0
    # two components on two devices: the producer's output crosses once per use of the consumer
    works2 = {"enc": Work(output_bytes=100 * 2**20), "dec": Work(output_bytes=5)}
    assert TE.link_seconds("component_placement", {"enc": "cuda:0", "dec": "cuda:1"}, works2, {"enc": 1, "dec": 4},
                           nvlink) == pytest.approx(4 * 100 * 2**20 / 300e9)


def test_a_pair_the_profile_does_not_price_refuses_by_name():
    works = {"model": Work(output_bytes=1)}
    with pytest.raises(TE.MissingFact, match="cuda:0 and cuda:2"):
        TE.link_seconds("block_scatter", {"model": "fgp:cuda:0,cuda:2"}, works, {"model": 1}, {("cuda:0", "cuda:1"): 1e9})
    with pytest.raises(TE.MissingFact, match="weight_sharding"):
        TE.link_seconds("weight_sharding", {"model": "tp:cuda:0,cuda:1"}, works, {"model": 1}, {("cuda:0", "cuda:1"): 1e9})


def test_plan_seconds_adds_the_link_term_when_links_are_given():
    works = {"model": Work(flops=0, weight_bytes=0, activation_bytes=0, output_bytes=300e9)}
    devices = {"cuda:0": _facts(), "cuda:1": _facts()}
    base = TE.plan_seconds("block_scatter", {"model": "fgp:cuda:0,cuda:1"}, works, {"model": 1}, devices, None, "float16", stored={"model": "float16"})
    with_link = TE.plan_seconds("block_scatter", {"model": "fgp:cuda:0,cuda:1"}, works, {"model": 1}, devices, None,
                                "float16", links={("cuda:0", "cuda:1"): 300e9}, stored={"model": "float16"})
    assert with_link == pytest.approx(base + 1.0)


def test_the_uses_follow_the_flow_s_names_as_the_executor_resolves_them():
    """GLM-4.1V-9B: the flow names `language_model`, the component is `model.language_model`; unresolved, the LM
    counted once and every streaming plan estimated 2.5 s (prism_choices_2026_10_25_after.tsv, 16:30)."""
    flow = {"order": ["model.visual", "model.language_model", "lm_head"],
            "generation": {"lm_component": "language_model", "head_component": "lm_head"}}
    uses = TE.uses_per_request(flow, None, 32, components=["model.visual", "model.language_model", "lm_head"])
    assert uses["model.language_model"] == ("token", 32) and uses["lm_head"] == ("token", 32)
    assert uses["model.visual"] == ("once", 1) and "language_model" not in uses
    # an exact name stays exact; an ambiguous suffix is left as named (nothing invented)
    assert TE.resolve_flow_name("lm_head", ["a.lm_head", "lm_head"]) == "lm_head"
    assert TE.resolve_flow_name("x", ["a.x", "b.x"]) == "x"


def test_the_tiling_rung_says_why_it_declines_sana_4k_on_16gb(monkeypatch):
    """D3 (the supervisor, 16:02 / 16:53), from the table: Sana_1600M_4Kpx_BF16 on a V100-16GB, weights 11.7 GB,
    the VAE's activations 23.7 GB. The rung declined in silence (op_level_tiling absent from the candidates, nothing in
    refused). Now every exit of `_try_op_level_tiling` records its reason: here the VAE holds 15 808 MB live before
    `aten.view::95` whatever its bands — an op-level tile cuts an op's own output, never what earlier ops hold — over
    the 11 885 MB the card leaves beside the weights. The next step is the region cut (activation architecture, step
    C), named, not pretended."""
    from neurobrix.core.prism.solver import ComponentMemory, DeviceState
    MB = 2**20
    s = S.PrismSolver.__new__(S.PrismSolver)
    s.oom_reserve_mb = 3072
    s._live_floor = {"vae": (15808 * MB, "aten.view::95")}
    s._neural_components = []
    monkeypatch.setattr(s, "_arena_over_allocation", lambda *a, **k: None)
    monkeypatch.setattr(s, "_flow_topology", lambda c: {})
    from neurobrix.core.flow import residency as _res
    monkeypatch.setattr(_res, "resident_beside", lambda topo, name, serve: [])   # Sana's flow releases the others
    vae = ComponentMemory("vae", 607 * MB, 23712 * MB, 1216 * MB, peak_op_uid="aten.pixel_shuffle::4", activation_profiled=True)
    tr = ComponentMemory("transformer", 6121 * MB, 979 * MB, 355 * MB, activation_profiled=True)
    te = ComponentMemory("text_encoder", 4986 * MB, 166 * MB, 258 * MB, activation_profiled=True)
    devs = [DeviceState(device_string="cuda:0", capacity_mb=15564.0)]
    out = s._try_op_level_tiling([("vae", vae), ("transformer", tr), ("text_encoder", te)], None, devs, {}, None, None)
    assert out is None
    why = s._declined["op_level_tiling"]
    assert "15808 MB are live before aten.view::95" in why and "never what earlier ops hold" in why, why
    # and on a 32 GB card nothing overflows the room a component runs in: a whole-component rung serves it
    s2 = S.PrismSolver.__new__(S.PrismSolver); s2.oom_reserve_mb = 3072; s2._live_floor = {}; s2._neural_components = []
    monkeypatch.setattr(s2, "_arena_over_allocation", lambda *a, **k: None)
    monkeypatch.setattr(s2, "_flow_topology", lambda c: {})
    assert s2._try_op_level_tiling([("vae", vae), ("transformer", tr), ("text_encoder", te)], None,
                                   [DeviceState(device_string="cuda:0", capacity_mb=31130.0)], {}, None, None) is None
    assert "no component's activation overflows" in s2._declined["op_level_tiling"]


def test_an_expert_store_s_per_token_traffic_is_priced_on_the_host_link():
    """D4 (the supervisor, 16:53): Qwen3-30B-A3B as an expert store (48 routed layers x 8 of 128 experts of 9 MB a token)
    estimated 15 s against 174 s measured — the experts a token reads were not priced. Now they cross the host link at
    every use for every non-resident strategy; a resident plan (the whole model on the cards) pays nothing."""
    per_token = 48 * 8 * 9 * 2**20
    tok = Work(flops=0, weight_bytes=0, activation_bytes=0, output_bytes=0, expert_bytes_per_use=per_token)
    devices = {"cuda:0": _facts()}
    lazy = TE.plan_seconds("lazy_sequential", {"model": "cuda:0"}, {"model": tok}, {"model": 32}, devices, None, "float16", stored={"model": "float16"})
    assert lazy == pytest.approx(32 * per_token / 8.1e9)
    resident = TE.plan_seconds("block_scatter", {"model": "fgp:cuda:0,cuda:1"}, {"model": tok}, {"model": 32},
                               {"cuda:0": _facts(), "cuda:1": _facts()}, None, "float16", stored={"model": "float16"})
    assert resident == 0.0
    # the prompt pass reads the experts its tokens touch, once
    prm = Work(expert_bytes_per_use=4 * 2**30)
    with_prompt = TE.plan_seconds("lazy_sequential", {"model": "cuda:0"}, {"model": tok}, {"model": 32}, devices, None,
                                  "float16", prompt={"model": prm}, stored={"model": "float16"})
    assert with_prompt == pytest.approx(lazy + 4 * 2**30 / 8.1e9)


def test_without_facts_a_lazy_plan_of_several_components_still_outranks_layer_streaming():
    """The Mac, 17:18: on a profile without measured facts the MoE LLMs (lazy_sequential = the expert store) fell to
    layer_streaming — the lazy penalty (20 points a component) had pushed lazy under layer_streaming inside the rank."""
    s = S.PrismSolver.__new__(S.PrismSolver)
    three = {"model": "cuda:0", "lm_head": "cuda:0", "vision": "cuda:0"}
    assert s._score_strategy("lazy_sequential", three, _NoTopology()) > s._score_strategy("layer_streaming", three, _NoTopology())
    assert s._score_strategy("component_placement_lazy", three, _NoTopology()) > s._score_strategy("layer_streaming", three, _NoTopology())
    # Janus-Pro-7B: 7 components — 140 points of lazy penalty took lazy_sequential (324) under layer_streaming (400)
    # on Metal (the Mac, 17:37). Without facts the base order of the rank holds whatever the component count.
    seven = {f"c{i}": "cuda:0" for i in range(7)}
    lazy, cpl = s._score_strategy("lazy_sequential", seven, _NoTopology()), s._score_strategy("component_placement_lazy", seven, _NoTopology())
    z3, ls = s._score_strategy("zero3", seven, _NoTopology()), s._score_strategy("layer_streaming", seven, _NoTopology())
    assert cpl > lazy > z3 > ls, (cpl, lazy, z3, ls)
    assert all(S.RANK_BASE[4] - S.RANK_FLOOR_SPAN <= v <= S.RANK_BASE[4] + S.RANK_FLOOR_SPAN for v in (cpl, lazy, z3, ls))
