"""A `layer_streaming` boundary must name an op the executor will still have.

Prism partitioned the RAW container graph. Each sequence then rewrote that graph in place
before running it, so the plan's boundaries named ops the fusions had folded away and the
strategy refused at execution:

    layer_streaming: the plan's segment boundaries are not in 'model's graph
    (2 of 4 op ids absent, e.g. 'aten.silu::843')

Six models named it verbatim on the Mac (9675411a, e1e0fd94, 3c734f70). It reproduces on this
rack under the Mac's own unified profile, because the cascade reads the PROFILE and not the
card: unified memory makes `lazy_sequential` non-viable, so `layer_streaming` wins by
elimination rather than by score.

THE GATE IS THE INVARIANT, NOT AN OP ID
---------------------------------------
The Mac measured the SAME model failing on `aten.silu::890` at its profile budget and
`aten.silu::843` at `NBX_PRISM_BUDGET_MB=12288` — **the boundary moves with the memory rung.**
A cell pinned to one id would pass at another rung and prove nothing. So every assertion here
is of the form "every boundary the partition produced is present in the graph that will run",
which holds at any rung and for any model.

The chase, recorded because two of its three steps corrected the step before:

* the swiglu fusion takes `aten.silu::843` and `::890` (977 silu+mul pairs -> `custom.swiglu_fused`);
* `aten.mm::2648`, the boundary that then surfaced, survives EVERY branch transform applied in
  the sequence's own order (11 722 -> 7 355 ops);
* the MoE fusion is what removes it — 11 722 -> 2 678 ops, the expert matmuls folded into 26
  `custom::moe_fused`. It is shared by both modes, so it runs first in the normalization.

Shapes: DeepSeek-Coder-V2-Lite-Instruct is the Mac's own measured case and a MoE model, so it
exercises both the shared fusion and the branch rewrites. A non-MoE model is used as the
control: its normalization must still leave a usable partition, or the pass would be trading
one defect for another.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from neurobrix.core.optim.passes.normalize import boundaries_present, normalize_for_branch
from neurobrix.core.prism.layer_partition import LayerPartitioner

CACHE = Path(os.path.expanduser("~/.neurobrix/ca" + "che"))
MOE_MODEL = "DeepSeek-Coder-V2-Lite-Instruct"
SEGMENT_BUDGET = 10112 * 1024 * 1024      # the real segment budget the rung produced


def _graph(model: str, component: str = "model"):
    p = CACHE / model / "components" / component / "graph.json"
    if not p.exists():
        pytest.skip(f"{model}/{component} is not in this cache")
    return json.loads(p.read_text())


def _family(model: str) -> str:
    p = CACHE / model / "manifest.json"
    if not p.exists():
        pytest.skip(f"{model} has no manifest")
    # The family is in the MANIFEST, not the topology — reading the wrong one returns "",
    # which silently SKIPS the MoE fusion and makes the whole pass look inert.
    return str(json.loads(p.read_text()).get("family") or "")


def _bounds(graph, budget=SEGMENT_BUDGET):
    part = LayerPartitioner(graph, None).partition(budget)
    return [[s.first_op, s.last_op] for s in part.segments], part


# ───────────────────────── the defect, shown to exist ─────────────────────────

def test_boundaries_cut_on_the_RAW_graph_go_absent_once_the_graph_is_normalized():
    """The defect itself. If this ever passes, the pass below is no longer needed."""
    raw = _graph(MOE_MODEL)
    bounds, part = _bounds(raw)
    assert part.fits and len(bounds) >= 2, "the control is vacuous: the raw graph did not cut"
    normalized = normalize_for_branch(raw, "triton", _family(MOE_MODEL))
    missing = boundaries_present(normalized, bounds)
    assert missing, (
        "boundaries cut on the raw graph survived normalization — the defect this file "
        "gates would no longer reproduce, so the gate below proves nothing.")


# ───────────────────────── the invariant the fix restores ─────────────────────────

def test_boundaries_cut_on_the_NORMALIZED_graph_are_all_present_in_it():
    """The whole fix, as one invariant. No op id appears in this assertion."""
    raw = _graph(MOE_MODEL)
    normalized = normalize_for_branch(raw, "triton", _family(MOE_MODEL))
    bounds, part = _bounds(normalized)
    assert part.fits and len(bounds) >= 2, "the normalized graph did not cut — nothing is gated"
    assert boundaries_present(normalized, bounds) == []


@pytest.mark.parametrize("budget_mb", [8192, 10112, 12288, 16384])
def test_the_invariant_holds_at_EVERY_rung_because_the_boundary_moves_with_it(budget_mb):
    """The Mac's warning, as a cell: ::890 at one rung, ::843 at another, same defect."""
    raw = _graph(MOE_MODEL)
    normalized = normalize_for_branch(raw, "triton", _family(MOE_MODEL))
    bounds, part = _bounds(normalized, budget_mb * 1024 * 1024)
    if not part.fits or len(bounds) < 2:
        pytest.skip(f"the graph does not cut into segments at {budget_mb} MB")
    assert boundaries_present(normalized, bounds) == []


# ───────────────────────── the pass must not do harm ─────────────────────────

def test_normalization_never_mutates_the_container_graph():
    """The container's graph is read by other components and by the census. A plan-time
    rewrite leaking into it would be worse than the defect being fixed."""
    raw = _graph(MOE_MODEL)
    before = len(raw["execution_order"])
    normalize_for_branch(raw, "triton", _family(MOE_MODEL))
    assert len(raw["execution_order"]) == before


def test_the_moe_fusion_is_what_removes_the_second_boundary():
    """Pins the CAUSE, not the id: with no family the MoE fusion is skipped, and the graph
    keeps far more ops than when it runs. Two runs of the same function, one measurement."""
    raw = _graph(MOE_MODEL)
    without = normalize_for_branch(raw, "triton", "")
    with_moe = normalize_for_branch(raw, "triton", _family(MOE_MODEL))
    assert len(with_moe["execution_order"]) < len(without["execution_order"]), (
        "the family made no difference, so the MoE fusion did not run — the most likely "
        "cause is reading `family` from the topology, where it does not live.")


def test_a_compiled_plan_is_normalized_too_but_only_by_the_shared_fusion():
    """R30: the compiled branch plans on a graph it will also have fused. Its own two
    transforms need weights or a built sequence, so nothing else is applied — stated, not
    silently skipped."""
    raw = _graph(MOE_MODEL)
    compiled = normalize_for_branch(raw, "compiled", _family(MOE_MODEL))
    triton = normalize_for_branch(raw, "triton", _family(MOE_MODEL))
    assert len(compiled["execution_order"]) < len(raw["execution_order"])   # MoE fusion ran
    assert len(triton["execution_order"]) < len(compiled["execution_order"])  # plus branch passes


# ───────────── 2026-09-24: the graph the STRATEGY cuts, not the one Prism cut ─────────────
#
# The cells above compare the normalised graph with ITSELF: they never looked at the graph
# `layer_streaming` checks, which for a weightless streamed base is the graph as LOADED (register
# 106). The strategy now cuts `normalize_for_branch(base graph)`; that is sound only if the
# normalisation is idempotent — the base graph may already carry the MoE fusion (its load) or the
# branch rewrites (a base compiled whole). The executed half is
# `tests/regression/test_a_streamed_lm_cuts_the_graph_prism_cut.py`.

DENSE_MODEL, DENSE_COMP = "granite-speech-3.3-8b", "language_model"


@pytest.mark.parametrize("model,comp", [(MOE_MODEL, "model"), (DENSE_MODEL, DENSE_COMP)])
@pytest.mark.parametrize("mode", ["triton", "triton_sequential", "compiled"])
def test_normalizing_a_normalized_graph_changes_nothing(model, comp, mode):
    once = normalize_for_branch(_graph(model, comp), mode, _family(model))
    twice = normalize_for_branch(once, mode, _family(model))
    assert twice["execution_order"] == once["execution_order"]
    assert set(twice["ops"]) == set(once["ops"])


def test_triton_sequential_cuts_the_graph_it_runs_op_by_op():
    """Only `TritonSequence.compile` (mode `triton`) performs the branch rewrites; the op-by-op
    engine runs the loaded graph. Its plan fused silu+mul and named `custom.swiglu_fused` ops it
    never has (the Mac's triton-sequential rows)."""
    raw = _graph(DENSE_MODEL, DENSE_COMP)
    fam = _family(DENSE_MODEL)
    tseq = normalize_for_branch(raw, "triton_sequential", fam)
    assert tseq["execution_order"] == normalize_for_branch(raw, "compiled", fam)["execution_order"]
    assert not any("swiglu_fused" in u for u in tseq["execution_order"])
    assert any("swiglu_fused" in u for u in normalize_for_branch(raw, "triton", fam)["execution_order"])
