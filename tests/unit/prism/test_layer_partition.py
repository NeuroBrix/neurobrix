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
