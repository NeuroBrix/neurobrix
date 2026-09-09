"""The layer partitioner: boundaries from dataflow, and a budget it keeps.

The cascade's last rung streams one COMPONENT at a time, which is not fine
enough for a model that is one large component — measured 2026-09-09,
DeepSeek-Coder-V2-Lite is a single `model` component of 17777 MB of live
weights. This module supplies the missing grain.

What is tested here is what the rest will rely on:

  * cuts are found from the ORDER and the dataflow, never from a module name
  * every segment fits, and the PEAK RESIDENT it announces is under the
    budget — including the activations, which the first version added on top
    of a full budget and so promised 521.5 MB against 500
  * genuine impossibility refuses with its arithmetic instead of returning a
    partition that cannot be executed
"""

from __future__ import annotations

import pytest

from neurobrix.core.prism.layer_partition import (
    LayerPartitioner, Partition, tensor_bytes)


def _chain(n_blocks: int, weight_mb: int, act_mb: int = 1) -> dict:
    """A synthetic graph: `n_blocks` repeated blocks, each one weighted op.

    Deliberately carries NO `parent_module` and no recognisable naming: if
    the partitioner needed either, this graph would defeat it.
    """
    w_elems = weight_mb * 1024 * 1024 // 2          # bfloat16
    a_elems = act_mb * 1024 * 1024 // 2
    tensors, ops, order = {}, {}, []
    tensors["x0"] = {"shape": [a_elems], "dtype": "bfloat16", "is_parameter": False}
    prev = "x0"
    for i in range(n_blocks):
        wid, oid, out = f"p{i}", f"o{i}", f"x{i+1}"
        tensors[wid] = {"shape": [w_elems], "dtype": "bfloat16",
                        "is_parameter": True, "weight_name": f"w{i}"}
        tensors[out] = {"shape": [a_elems], "dtype": "bfloat16",
                        "is_parameter": False}
        ops[oid] = {"input_tensor_ids": [prev, wid], "output_tensor_ids": [out]}
        order.append(oid)
        prev = out
    return {"tensors": tensors, "ops": ops, "execution_order": order}


def test_cuts_come_from_dataflow_not_from_names():
    """Eight 10 MB blocks into a 35 MB budget."""
    g = _chain(8, weight_mb=10)
    part = LayerPartitioner(g).partition(35 * 1024 * 1024)
    assert part.fits, part.refusal
    assert len(part.segments) >= 3, "eight 10 MB blocks cannot be fewer than 3 pieces of 35 MB"
    assert sum(s.op_count for s in part.segments) == 8, "every op must land in exactly one segment"
    # contiguous and in order
    firsts = [s.first_op for s in part.segments]
    assert firsts == sorted(firsts, key=lambda o: g["execution_order"].index(o))


@pytest.mark.parametrize("budget_mb", [15, 21, 35, 64, 128])
def test_the_announced_peak_is_under_the_budget(budget_mb):
    """The number it promises is the number it holds.

    The first version sized segments against the WHOLE budget and then added
    the activation peak on top, announcing 521.5 MB against 500. A rung whose
    announced budget is not its executed budget is the defect this whole
    milestone is about, so it is a property test and not an example.
    """
    g = _chain(12, weight_mb=10, act_mb=2)
    part = LayerPartitioner(g).partition(budget_mb * 1024 * 1024)
    if not part.fits:
        return                      # refusing is allowed; over-promising is not
    assert part.peak_resident_bytes <= budget_mb * 1024 * 1024, (
        f"announced {part.peak_resident_mb:.1f} MB against a {budget_mb} MB budget")
    for s in part.segments:
        assert s.weight_bytes + part.peak_live_bytes <= budget_mb * 1024 * 1024


def test_an_op_wider_than_the_budget_refuses_with_its_arithmetic():
    """A cut cannot run half an op."""
    g = _chain(4, weight_mb=40)
    part = LayerPartitioner(g).partition(20 * 1024 * 1024)
    assert not part.fits
    assert "on its own" in part.refusal
    assert "40.0 MB" in part.refusal, part.refusal
    assert part.segments == []


def test_activations_alone_over_budget_refuses_and_says_cutting_cannot_help():
    g = _chain(4, weight_mb=1, act_mb=64)
    part = LayerPartitioner(g).partition(8 * 1024 * 1024)
    assert not part.fits
    assert "Cutting" in part.refusal or "cutting" in part.refusal


def test_weight_sizes_from_the_index_win_over_the_graph():
    """The index records what is STORED, which is what a load costs."""
    g = _chain(4, weight_mb=10)
    # tell the partitioner each weight is really 1 MB
    sizes = {f"w{i}": 1024 * 1024 for i in range(4)}
    part = LayerPartitioner(g, sizes).partition(8 * 1024 * 1024)
    assert part.fits, part.refusal
    assert len(part.segments) == 1, "four 1 MB weights fit one 8 MB segment"


def test_tensor_bytes_refuses_to_guess():
    assert tensor_bytes({"shape": [4], "dtype": "float32"}) == 16
    assert tensor_bytes({"shape": [4], "dtype": "who knows"}) is None
    assert tensor_bytes({"shape": ["s0", 4], "dtype": "float32"}) is None


def test_an_empty_graph_is_a_partition_with_nothing_in_it():
    part = LayerPartitioner({}).partition(1024)
    assert part.fits
    assert part.segments == []
    assert part.peak_resident_bytes == 0


# ---------------------------------------------------------------------------
# The segments, as executable graphs
# ---------------------------------------------------------------------------

from neurobrix.core.prism.layer_partition import build_segment_graph


def test_segments_reassemble_the_component_exactly():
    """Every op once, in order, across the segment graphs."""
    g = _chain(9, weight_mb=10)
    g["output_tensor_ids"] = ["x9"]
    part = LayerPartitioner(g).partition(35 * 1024 * 1024)
    assert part.fits

    seen = []
    for seg in part.segments:
        sub = build_segment_graph(g, seg)
        seen.extend(sub["execution_order"])
    assert seen == g["execution_order"], "the segments must tile the order exactly"


def test_a_segment_carries_only_the_weights_its_ops_read():
    g = _chain(9, weight_mb=10)
    part = LayerPartitioner(g).partition(35 * 1024 * 1024)
    for seg in part.segments:
        sub = build_segment_graph(g, seg)
        params = {tid for tid, t in sub["tensors"].items() if t.get("is_parameter")}
        names = {sub["tensors"][tid].get("weight_name") for tid in params}
        assert names == seg.weight_names, (
            "the graph a segment executes and the weights it was budgeted "
            "for must be the same set")


def test_the_seam_is_declared_not_assumed():
    """A segment's inputs are what earlier ops produced and it reads."""
    g = _chain(6, weight_mb=10)
    g["output_tensor_ids"] = ["x6"]
    part = LayerPartitioner(g).partition(25 * 1024 * 1024)
    assert len(part.segments) > 1, "need at least one seam to test one"

    subs = [build_segment_graph(g, s) for s in part.segments]
    # A seam tensor is aliased so the executor can bind it: the id carries
    # `input::` and the NAME the caller passes is the original tensor id.
    assert "input::x0" in subs[0]["input_tensor_ids"]
    assert "x0" in subs[0]["segment_input_names"]
    for sub in subs:
        assert all(t.startswith("input::") for t in sub["input_tensor_ids"]), (
            "every declared input must be bindable by the executor, which "
            "finds them by that prefix and by nothing else")
        assert sub["segment_input_names"] == [t[7:] for t in sub["input_tensor_ids"]]

    # Every segment reads only what EARLIER segments published, or what the
    # component itself was given. Restricting it to the IMMEDIATELY previous
    # segment is too strict and measurably wrong: on DeepSeek's real graph
    # the second segment reads `input::position_ids`, a declared component
    # input with no producer, which no segment publishes because nothing
    # computes it. A tensor may also skip a segment entirely.
    component_inputs = set(g.get("input_tensor_ids") or []) | {"x0"}
    published = set(component_inputs)
    for sub in subs:
        # compare in the caller's vocabulary: the names, not the aliases
        unmet = set(sub["segment_input_names"]) - published
        assert not unmet, (
            f"segment {sub['segment_index']} reads {sorted(unmet)}, which "
            f"neither an earlier segment publishes nor the component "
            f"receives")
        published |= set(sub["output_tensor_ids"])

    # the last publishes the component's output
    assert "x6" in subs[-1]["output_tensor_ids"]


def test_a_single_segment_component_is_the_component():
    g = _chain(3, weight_mb=1)
    g["output_tensor_ids"] = ["x3"]
    part = LayerPartitioner(g).partition(64 * 1024 * 1024)
    assert len(part.segments) == 1
    sub = build_segment_graph(g, part.segments[0])
    assert sub["execution_order"] == g["execution_order"]
    assert sub["output_tensor_ids"] == ["x3"]


# ---------------------------------------------------------------------------
# The rung: what it would choose, and why it loses when it should
# ---------------------------------------------------------------------------

def test_the_rung_scores_below_every_whole_component_strategy():
    """Inertia, stated where it is enforced.

    `layer_streaming` is never gated on a vendor, a device count or a memory
    size. It simply scores below every strategy that keeps a component whole,
    so on a machine where one of those is viable it loses and is never
    chosen. This pins that ordering: if someone raises its score above zero3
    or single_gpu, a big card starts streaming per layer for no reason.
    """
    import re
    from pathlib import Path

    src = Path("src/neurobrix/core/prism/solver.py").read_text()
    scores = dict(re.findall(r'"([a-z0-9_]+)":\s*(\d+)(?:\.\d+)?,', src))
    assert "layer_streaming" in scores, "the rung must carry a score"
    layer = int(scores["layer_streaming"])

    for whole in ("single_gpu", "zero3", "lazy_sequential"):
        if whole in scores:
            assert layer < int(scores[whole]), (
                f"layer_streaming ({layer}) must score below {whole} "
                f"({scores[whole]}): a machine that can hold a component "
                f"whole must never stream it")

    for host in ("cpu_execution", "cpu_streaming"):
        if host in scores:
            assert layer > int(scores[host]), (
                f"layer_streaming ({layer}) must score above {host} "
                f"({scores[host]}): streaming on the accelerator beats "
                f"moving the whole model to the host")


def test_the_alias_does_not_damage_the_graph_it_was_cut_from():
    """A partition must leave its source intact.

    Ops are shared dicts between the full graph and a segment, so rewriting
    an op's inputs in place would corrupt the component for every later
    segment and for anyone else holding the graph.
    """
    g = _chain(6, weight_mb=10)
    before = {oid: list(op["input_tensor_ids"]) for oid, op in g["ops"].items()}
    part = LayerPartitioner(g).partition(25 * 1024 * 1024)
    for seg in part.segments:
        build_segment_graph(g, seg)
    after = {oid: list(op["input_tensor_ids"]) for oid, op in g["ops"].items()}
    assert before == after, "building a segment rewrote the source graph"


def test_a_consuming_op_reads_the_alias_not_the_original():
    """The rewrite has to reach the ops, or the binding is decorative."""
    g = _chain(6, weight_mb=10)
    part = LayerPartitioner(g).partition(25 * 1024 * 1024)
    subs = [build_segment_graph(g, s) for s in part.segments]
    later = subs[1]
    aliased = {t[7:]: t for t in later["input_tensor_ids"]}
    first_op = later["ops"][later["execution_order"][0]]
    for tid in first_op["input_tensor_ids"]:
        assert tid not in aliased, (
            f"op still reads {tid!r}, which nothing will bind; it should read "
            f"{aliased.get(tid)!r}")


def test_the_alias_carries_the_shape_and_dtype_of_what_it_replaces():
    g = _chain(6, weight_mb=10, act_mb=3)
    part = LayerPartitioner(g).partition(25 * 1024 * 1024)
    subs = [build_segment_graph(g, s) for s in part.segments]
    for sub in subs:
        for tid in sub["input_tensor_ids"]:
            alias = sub["tensors"][tid]
            src_id = alias["seam_alias_of"] if "seam_alias_of" in alias else None
            if src_id is None:
                continue
            assert alias["shape"] == g["tensors"][src_id]["shape"]
            assert alias["dtype"] == g["tensors"][src_id]["dtype"]
            assert alias["is_input"] is True
            assert alias["producer_op_uid"] is None
