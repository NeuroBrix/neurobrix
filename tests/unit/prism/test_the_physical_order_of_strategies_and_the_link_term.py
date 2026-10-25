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

from neurobrix.core.prism import solver as S
from neurobrix.core.prism import time_estimate as TE
from neurobrix.core.prism.work import Work


class _NoTopology:
    topology = None
    cpu = None


def _score(name, devices=("cuda:0",)):
    s = S.PrismSolver.__new__(S.PrismSolver)
    allocs = {"a": devices[0], "b": devices[-1]}
    return s._score_strategy(name, allocs, _NoTopology())


def test_the_ranks_never_cross_whatever_the_penalties():
    order = ["single_gpu", "component_placement", "op_level_tiling", "zero3", "layer_streaming", "cpu_execution"]
    scores = [_score(n) for n in order]
    assert scores == sorted(scores, reverse=True), dict(zip(order, scores))
    # inside rank 4 the per-use and per-phase reloads are ordered by the estimate at solve time; here, by score
    assert _score("lazy_sequential") > _score("layer_streaming")
    assert _score("op_level_tiling") > _score("lazy_sequential") > _score("zero3") >= S.RANK_BASE[4] - S.RANK_FLOOR_SPAN
    assert _score("zero3") > _score("layer_streaming") > _score("cpu_execution")


def test_a_strategy_without_a_rank_is_refused_not_scored():
    with pytest.raises(RuntimeError, match="STRATEGY_RANK"):
        _score("a_rung_nobody_placed")


def _facts():
    return TE.DeviceFacts(matmul_flops={"float16": 42e12}, memory_bps=789e9, h2d_bps=8.1e9)


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
    base = TE.plan_seconds("block_scatter", {"model": "fgp:cuda:0,cuda:1"}, works, {"model": 1}, devices, None, "float16")
    with_link = TE.plan_seconds("block_scatter", {"model": "fgp:cuda:0,cuda:1"}, works, {"model": 1}, devices, None,
                                "float16", links={("cuda:0", "cuda:1"): 300e9})
    assert with_link == pytest.approx(base + 1.0)
