"""A piece reserves the activations alive WHILE its widest op runs — its inputs at their last use
plus its outputs (`LayerPartitioner.op_peak_curve`, the profiler's rule) — not the set alive after
it (`live_activation_curve`, what a cut carries).

Allegro's transformer at its derived request on a 16 GB V100 (guidance batch 2): reserving the
after-op maximum priced 7 012 MB of activations where the profiler prices 12 541 — the FFN's gelu
reads [2, 79 200, 9 216] fp32 and writes as much, 5 569 MB each, alive together — and the Triton
run died at aten.gelu::16 asking 5 569 MB with 11 050 MB live (2026-10-05).

Injection (seen red, then restored green): `_greedy_on` reserving `max(curve)` again (its `peaks`
argument ignored) -> the reserve is one FFN tensor, not two, and a budget under the in-op peak
"fits".
"""
from __future__ import annotations

from tests.unit.prism._graph import as_graph

from neurobrix.core.prism.layer_partition import LayerPartitioner

MB = 1024 * 1024


def _ffn(n_blocks: int, weight_mb: int = 10, wide_mb: int = 40, narrow_mb: int = 2) -> dict:
    """`n_blocks` of: up-projection (weighted, narrow -> wide), activation (wide -> wide, no
    weight), down-projection (weighted, wide -> narrow). Names carry nothing."""
    tensors, ops, order = {}, {}, []

    def act(tid, mb):
        tensors[tid] = {"shape": [mb * MB // 4], "dtype": "float32", "is_parameter": False}

    def param(tid, mb):
        tensors[tid] = {"shape": [mb * MB // 4], "dtype": "float32", "is_parameter": True,
                        "weight_name": tid}

    act("x0", narrow_mb)
    prev = "x0"
    for i in range(n_blocks):
        param(f"wu{i}", weight_mb), param(f"wd{i}", weight_mb)
        act(f"h{i}", wide_mb), act(f"g{i}", wide_mb), act(f"x{i + 1}", narrow_mb)
        for uid, ins, out in ((f"up{i}", [prev, f"wu{i}"], f"h{i}"), (f"act{i}", [f"h{i}"], f"g{i}"),
                              (f"down{i}", [f"g{i}", f"wd{i}"], f"x{i + 1}")):
            ops[uid] = {"op_type": "aten::mm", "input_tensor_ids": ins, "output_tensor_ids": [out]}
            order.append(uid)
        prev = f"x{i + 1}"
    return {"tensors": tensors, "ops": ops, "execution_order": order,
            "output_tensor_ids": [prev]}


def test_the_reserve_is_the_activation_ops_input_and_output_together():
    lp = LayerPartitioner(as_graph(_ffn(4)))
    after, during = lp.live_activation_curve(), lp.op_peak_curve()
    i = lp.order.index("act1")
    assert during[i] >= 80 * MB, during[i]                 # h and g alive while it runs
    assert max(after) < 80 * MB, max(after)                # no CUT ever carries both
    part = lp.partition(200 * MB)
    assert part.fits, part.refusal
    assert part.peak_live_bytes == max(during), (part.peak_live_bytes, max(during))


def test_a_budget_under_the_in_op_peak_does_not_fit_on_the_after_op_figure():
    """Weights of the widest piece (two projections) + the after-op figure fit in 72 MB; the in-op
    figure does not: the honest partition refuses or announces a peak under the budget that
    includes the in-op reserve."""
    lp = LayerPartitioner(as_graph(_ffn(4)))
    budget = 72 * MB
    part = lp.partition(budget)
    if part.fits:
        assert part.peak_live_bytes >= max(lp.op_peak_curve())
        assert part.peak_resident_bytes <= budget
        for s in part.segments:
            assert s.weight_bytes + max(lp.op_peak_curve()) <= budget, s


def test_a_view_allocates_nothing_while_it_runs():
    """`aten::_unsafe_view` of a 40 MB product: its output IS the product's storage, so the in-op
    figure at the view is the live set before it — not the product twice (Ming-Lite-Omni-1.5's
    vision tower on the busy Mac read 9 878 MB at such a view of a 4 420 MB bmm output).

    Injection (seen red, then restored green): the `profiler.VIEW_OP_TYPES` test in `_walk_liveness`
    removed -> the view's in-op figure is 80 MB."""
    g = _ffn(1)
    g["tensors"]["v0"] = dict(g["tensors"]["h0"])
    g["ops"]["view0"] = {"op_type": "aten::_unsafe_view", "input_tensor_ids": ["h0"],
                         "output_tensor_ids": ["v0"]}
    g["ops"]["act0"]["input_tensor_ids"] = ["v0"]
    g["execution_order"].insert(1, "view0")
    lp = LayerPartitioner(as_graph(g))
    during, after = lp.op_peak_curve(), lp.live_activation_curve()
    i = lp.order.index("view0")
    assert during[i] == after[i - 1], (during[i] / MB, after[i - 1] / MB)
    assert during[i] < 80 * MB
    assert during[lp.order.index("act0")] >= 80 * MB          # the activation still holds both


def _peak_at_a_view() -> dict:
    """One FFN block whose up-projection's 40 MB product is read through an `_unsafe_view` by an
    activation that writes a narrow tensor: counted as a second buffer, the view is the peak."""
    g = _ffn(1)
    g["tensors"]["v0"] = dict(g["tensors"]["h0"])
    g["tensors"]["g0"] = dict(g["tensors"]["x0"])
    g["ops"]["view0"] = {"op_type": "aten::_unsafe_view", "input_tensor_ids": ["h0"],
                         "output_tensor_ids": ["v0"]}
    g["ops"]["act0"]["input_tensor_ids"] = ["v0"]
    g["execution_order"].insert(1, "view0")
    return as_graph(g)


def test_the_placement_estimate_and_the_partitioner_hold_one_figure_for_a_graph():
    """ONE view rule (`profiler.VIEW_OP_TYPES`) for every walk of a graph's activations: the
    placement estimate's peak (`ActivationProfiler.estimate_peak_memory`) is the partitioner's
    in-op maximum (`LayerPartitioner.op_peak_curve`), and the overflow scan that feeds op-level
    tiling does not flag a view for the bytes it never allocates.

    Injections (seen red, then restored green — STATE.md of the op_tiler campaign, 2026-10-05):
      A. the profiler's peak back to `current_bytes` at a view -> the two figures differ;
      B. the overflow scan counting a view's output -> the view is flagged."""
    from neurobrix.core.prism.profiler import ActivationProfiler
    g = _peak_at_a_view()
    lp = LayerPartitioner(as_graph(g))
    figure = max(lp.op_peak_curve())
    threshold = 60 * MB
    ap = ActivationProfiler(g).estimate_peak_memory(vram_per_gpu_bytes=int(threshold / 0.85) + 1)
    assert ap.peak_bytes == figure, (ap.peak_bytes / MB, figure / MB)
    assert figure < 80 * MB                                   # not the product twice
    flagged = [o[0] for o in ap.overflow_ops]
    assert "view0" not in flagged, flagged
