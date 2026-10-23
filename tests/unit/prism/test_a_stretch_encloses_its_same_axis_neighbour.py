"""A new stretch on an axis is not walled by an existing stretch on the SAME axis: a seam may enclose it whole, and
the two run as one slicing — what passed between them never exists whole (HAT-L at 2048 px on 16 GB: the stretch
making the overlapping-window attention's unfolded K and V stopped at the attention's own stretch and handed them out
whole, 15 840 MB, 2026-10-22). A seam inside an existing stretch is never taken.

Injection (seen red, then restored green, 2026-10-22): existing stretches walls again (every region in `taken`) -> red."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from neurobrix.core.prism.chunked_region import plan_region, region_carriers
from neurobrix.core.prism.layer_partition import LayerPartitioner, _Region

_spec = importlib.util.spec_from_file_location(
    "stretch_case", Path(__file__).with_name("test_a_stretch_over_the_rung_runs_in_slices_of_its_token_axis.py"))
_case = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_case)


def test_a_new_stretch_encloses_the_neighbour_on_its_axis():
    seq = 4096
    sizes = {w: _case.D * 4 for w in _case.WEIGHTS}
    p = LayerPartitioner(_case._graph(), sizes, symbol_map={"s0": _case.B, "s1": seq},
                         compute_dtype_bytes=4, token_split=True)
    ax = p._axis("s1", 1)
    order = p.order
    hot = order.index("aten.bmm::1")                    # y = q @ kv: the op the new stretch is built around
    left = order.index("aten.mul::1")                   # q = h * wq: an existing stretch just before it
    plan, why = plan_region(ax, left, left, protected=p.protected,
                            carriers=region_carriers(p._dag, order[left:left + 1], p._op_symbols()))
    assert plan is not None, why
    existing = _Region(axis=ax, plan=plan, weight_names=set())
    existing.set_slice(1, p._region_peak(ax, plan, 1), p._symbol_map)
    eff, _ = p._effective_curve([existing])
    got, why = p._new_region(ax, hot, eff, 1 << 62, [existing], absorb=True, host_ok=False)
    assert got is not None, why
    # the seam before the existing stretch holds less than any after it: the new stretch starts there, enclosing it
    assert got.first <= left <= got.last, (order[got.first], order[got.last], order[left])
