"""A stretch of local spatial ops runs in slices of its axis, each widened by the stretch's halo (workstream 2).

`chunked_region.TokenAxis` gives a same-padded convolution a halo (`Verdict.halo` = dilation * (kernel - 1) / 2 rows of
its input) and carries nearest upsampling / pixel shuffle as a constant scale of the axis; `plan_region` sums the
receptive field in units of the axis symbol exactly and rounds it up once (`RegionPlan.margin`); the executor
(`chunked_piece`) reads [a - margin, b + margin] clipped at the axis' true ends and keeps [a, b) — NIST's exact
tile-based inference (halo = half the receptive field, zones of responsibility). Sana 4K's VAE decoder: no mixing op
along its height, its last stage a margin of 1 latent row.

What this test would do if the code were wrong: a margin one short — the rows next to each interior cut read the
convolutions' zero padding instead of their neighbours: red at the comparison (a large error at the cut rows); a
planner that let a contraction into a halo stretch — red at the refusal row.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism import chunked_region as CR

S = {"type": "symbol", "id": "s1", "trace": 7}


def _mul(n):
    return {"type": "mul", "left": S, "right": n, "trace": 7 * n}


def _t(dims, shape, param=False):
    d = {"shape": shape, "dtype": "float32", "symbolic_shape": {"dims": dims, "concrete": shape}}
    if param:
        d["is_parameter"] = True
    return d


def _T(tid):
    return {"type": "tensor", "tensor_id": tid}


def _conv(uid, x, w, out):
    return uid, {"op_type": "aten::convolution",
                 "attributes": {"args": [_T(x), _T(w), {"type": "scalar", "value": None},
                                         {"type": "list", "value": [1, 1]}, {"type": "list", "value": [1, 1]},
                                         {"type": "list", "value": [1, 1]}, {"type": "scalar", "value": False},
                                         {"type": "list", "value": [0, 0]}, {"type": "scalar", "value": 1}]},
                 "input_tensor_ids": [x, w], "output_tensor_ids": [out]}


def _graph():
    """conv3x3 -> silu -> upsample x2 -> conv3x3 -> conv3x3 along the height symbol s1."""
    tensors = {"input::x": _t([1, 4, S, 8], [1, 4, 7, 8]),
               "t::c1": _t([1, 4, S, 8], [1, 4, 7, 8]), "t::s": _t([1, 4, S, 8], [1, 4, 7, 8]),
               "t::u": _t([1, 4, _mul(2), 16], [1, 4, 14, 16]), "t::c2": _t([1, 4, _mul(2), 16], [1, 4, 14, 16]),
               "t::y": _t([1, 4, _mul(2), 16], [1, 4, 14, 16])}
    for w in ("w1", "w2", "w3"):
        tensors[f"param::{w}"] = _t([4, 4, 3, 3], [4, 4, 3, 3], True)
    ops = dict([_conv("conv1", "input::x", "param::w1", "t::c1"),
                ("silu", {"op_type": "aten::silu", "attributes": {"args": [_T("t::c1")]},
                          "input_tensor_ids": ["t::c1"], "output_tensor_ids": ["t::s"]}),
                ("up", {"op_type": "aten::upsample_nearest2d",
                        "attributes": {"args": [_T("t::s"), {"type": "list", "value": [_mul(2), 16]}]},
                        "input_tensor_ids": ["t::s"], "output_tensor_ids": ["t::u"]}),
                _conv("conv2", "t::u", "param::w2", "t::c2"), _conv("conv3", "t::c2", "param::w3", "t::y")])
    return {"tensors": tensors, "ops": ops, "execution_order": ["conv1", "silu", "up", "conv2", "conv3"],
            "input_tensor_ids": ["input::x"], "output_tensor_ids": ["t::y"]}


def test_the_halo_stretch_is_planned_with_its_receptive_field():
    ax = CR.TokenAxis(_graph(), "s1")
    assert [v.kind for v in ax.verdicts] == [CR.SLICE] * 5
    assert [v.halo for v in ax.verdicts] == [1, 0, 0, 1, 1]
    plan, why = CR.plan_region(ax, 0, 4)
    assert plan is not None, why
    assert plan.margin == 2                       # 1 row at scale 1 + 1/2 + 1/2 at scale 2


def _run_sliced(torch, x, ws, margin, total, count, ax):
    import torch.nn.functional as F
    f = lambda v: F.conv2d(F.conv2d(F.interpolate(F.silu(F.conv2d(v, ws[0], padding=1)), scale_factor=2,
                                                  mode="nearest"), ws[1], padding=1), ws[2], padding=1)
    lay_in, lay_out = ax.layout["input::x"], ax.layout["t::y"]
    out = None
    for start, length in CR.chunk_extents(total, count):
        a = max(0, start - margin)
        n = min(total, start + length + margin) - a
        part = f(CR.take_slice(x, lay_in, total, 1, a, n))
        part = CR.take_slice(part, lay_out, n, 2, start - a, length)
        if out is None:
            out = part.new_empty(CR.whole_shape(part.shape, lay_out, total, length))
        CR.put_slice(out, part, lay_out, total, 2, start, length)
    return out, f(x)


def test_slices_with_the_halo_equal_the_whole_and_one_row_short_does_not():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    total = 40
    x = torch.randn(1, 4, total, 8, dtype=torch.float64)
    ws = [torch.randn(4, 4, 3, 3, dtype=torch.float64) for _ in range(3)]
    ax = CR.TokenAxis(_graph(), "s1")
    plan, _ = CR.plan_region(ax, 0, 4)
    sliced, whole = _run_sliced(torch, x, ws, plan.margin, total, 5, ax)
    assert torch.allclose(sliced, whole, rtol=0, atol=1e-10)
    short, _ = _run_sliced(torch, x, ws, plan.margin - 1, total, 5, ax)
    assert (short - whole).abs().max() > 1e-3     # the cut rows read the padding


def test_a_halo_stretch_never_holds_a_contraction():
    """The planner's rule itself: the same stretch with its middle op a contraction (the k^T v of a linear attention)
    is refused — its widened slices overlap, and their partial sums would count the margin rows twice."""
    ax = CR.TokenAxis(_graph(), "s1")
    assert CR.plan_region(ax, 0, 4)[0] is not None
    ax.verdicts[1] = CR.Verdict(CR.CONTRACT)
    plan, why = CR.plan_region(ax, 0, 4)
    assert plan is None and "contraction" in why
