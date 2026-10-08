"""A dim computed from VALUES is classed over its envelope, never read at the trace's draw.

Kokoro's duration predictor sums its predicted per-phoneme durations into a frame count
(`repeat_interleave` over the rounded, clamped durations), and every dim downstream holds the
trace's 34 as a literal; the run lets the actual frame count win, so it formed baddbmm
(640, 240, 96) on this rack and (640, 160, 55) on the Mac where the table held only 34 frames
(the supervisor's decision, 2026-10-04 17:23). The derivation now finds such an axis in the graph,
bounds it by interval arithmetic over the ops producing its repeats (the duration projection's
50-wide sigmoid sum: [1, 50] frames per phoneme, the vendor's `max_dur` 50), carries it through
every dim downstream (a transposed convolution and a scaled upsample by their own arithmetic) and
derives the cone at every key class of the axis.

What this test would do if the code were wrong: a derivation reading the trace's extent only
(the bisection removed) forms one frame class — the exhaustive comparison and the 240-frame class
fail; a cone rule that dropped the transposed convolution's arithmetic leaves the 2F convolution at
the trace — the exhaustive comparison fails; an unbounded repeat read as a number instead of
refused passes the refusal case silently — it fails.

Shapes: a synthetic predictor at 23 phonemes traced to 34 frames, the Kokoro geometry (640
channels, a stride-2 depthwise transposed convolution, a 2.0 upsample); derived on a committed
V100 profile's ladders against the exhaustive enumeration of every extent in the envelope.
"""
from __future__ import annotations

import collections
import copy
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

import derived_census as D  # noqa: E402
from neurobrix.core.prism import runtime_widths as RW  # noqa: E402
from neurobrix.kernels import census as _census  # noqa: E402

MODEL, COMP = "synthetic-value-axis", "predictor"
S1 = {"type": "symbol", "id": "s1", "trace": 23}


def _t(shape, dtype="float32", sym=None, param=False):
    m = {"shape": list(shape), "dtype": dtype}
    if sym is not None:
        m["symbolic_shape"] = {"dims": sym, "concrete": list(shape)}
    if param:
        m["is_parameter"] = True
    return m


def _op(uid, kind, ins, outs, args, **attrs):
    return {"op_uid": uid, "op_type": kind, "input_tensor_ids": ins, "output_tensor_ids": outs,
            "attributes": {"args": args, "kwargs": {}, **attrs}}


T_ = lambda tid: {"type": "tensor", "tensor_id": tid}
S_ = lambda v: {"type": "scalar", "value": v}
L_ = lambda v: {"type": "list", "value": v}


def graph():
    """The predictor's value chain and cone, as Forge records them: durations [1, s1, 50] ->
    sigmoid -> sum(-1) -> /1.0 -> round -> clamp(min 1) -> int64 -> squeeze -> repeat_interleave
    [34]; the alignment [1, s1, 34] (eq against the phoneme index); bmm with the encoding
    [1, 640, s1]; a stride-2 transposed depthwise convolution to 68; a 2.0 upsample to 68; a
    convolution over the upsampled frames."""
    t = {
        "input::dur": _t([1, 23, 50], sym=[1, S1, 50]),
        "input::enc": _t([1, 640, 23], sym=[1, 640, S1]),
        "input::idx": _t([23, 1], "int64", sym=[S1, 1]),
        "param::up.w": _t([640, 1, 3], param=True), "param::up.b": _t([640], param=True),
        "param::c.w": _t([256, 640, 3], param=True), "param::c.b": _t([256], param=True),
    }
    ops, order = {}, []

    def add(uid, kind, ins, outs_shapes, args, **attrs):
        outs = []
        for i, (shp, dt) in enumerate(outs_shapes):
            tid = f"{uid}::out_{i}"
            # Forge's annotation: the phoneme axis (23 at the trace) is the symbol s1 everywhere
            t[tid] = _t(shp, dt, sym=[S1 if d == 23 else d for d in shp])
            outs.append(tid)
        ops[uid] = _op(uid, kind, ins, outs, args, **attrs)
        order.append(uid)
        return outs[0]
    x = add("aten.sigmoid::0", "aten::sigmoid", ["input::dur"], [([1, 23, 50], "float32")], [T_("input::dur")])
    x = add("aten.sum::0", "aten::sum", [x], [([1, 23], "float32")], [T_(x), L_([-1])], dim=[-1])
    x = add("aten.div::0", "aten::div", [x], [([1, 23], "float32")], [T_(x), S_(1.0)])
    x = add("aten.round::0", "aten::round", [x], [([1, 23], "float32")], [T_(x)])
    x = add("aten.clamp::0", "aten::clamp", [x], [([1, 23], "float32")], [T_(x), S_(1)], min=1)
    x = add("aten._to_copy::0", "aten::_to_copy", [x], [([1, 23], "int64")], [T_(x)])
    x = add("aten.squeeze::0", "aten::squeeze", [x], [([23], "int64")], [T_(x), S_(0)], dim=0)
    f = add("aten.repeat_interleave::0", "aten::repeat_interleave", [x], [([34], "int64")], [T_(x)])
    f = add("aten.unsqueeze::0", "aten::unsqueeze", [f], [([1, 34], "int64")], [T_(f), S_(0)])
    a = add("aten.eq::0", "aten::eq", ["input::idx", f], [([23, 34], "bool")], [T_("input::idx"), T_(f)])
    a = add("aten._to_copy::1", "aten::_to_copy", [a], [([23, 34], "float32")], [T_(a)])
    a = add("aten.unsqueeze::1", "aten::unsqueeze", [a], [([1, 23, 34], "float32")], [T_(a), S_(0)])
    a = add("aten.expand::0", "aten::expand", [a], [([1, 23, 34], "float32")], [T_(a), L_([1, S1, 34])])
    y = add("aten.bmm::0", "aten::bmm", ["input::enc", a], [([1, 640, 34], "float32")], [T_("input::enc"), T_(a)])
    z = add("aten.convolution::0", "aten::convolution", [y, "param::up.w", "param::up.b"],
            [([1, 640, 68], "float32")],
            [T_(y), T_("param::up.w"), T_("param::up.b"), L_([2]), L_([1]), L_([1]), S_(True), L_([1]), S_(640)],
            stride=[2], padding=[1], dilation=[1], transposed=True, output_padding=[1], groups=640)
    u = add("aten.upsample_nearest1d::0", "aten::upsample_nearest1d", [y], [([1, 640, 68], "float32")],
            [T_(y), L_([68]), S_(2.0)])
    s = add("aten.add::0", "aten::add", [z, u], [([1, 640, 68], "float32")], [T_(z), T_(u)])
    c = add("aten.convolution::1", "aten::convolution", [s, "param::c.w", "param::c.b"],
            [([1, 256, 68], "float32")],
            [T_(s), T_("param::c.w"), T_("param::c.b"), L_([1]), L_([1]), L_([1]), S_(False), L_([0]), S_(1)],
            stride=[1], padding=[1], dilation=[1], transposed=False, output_padding=[0], groups=1)
    return {"tensors": t, "ops": ops, "execution_order": order,
            "input_tensor_ids": ["input::dur", "input::enc", "input::idx"], "output_tensor_ids": [c],
            "symbolic_context": {"symbols": {"s1": {"name": "seq_len", "trace_value": 23,
                                                    "source": "input::dur::dim_1"}}}}


class _Contract:
    fp32_op_uids = frozenset()


@pytest.fixture
def synthetic(monkeypatch):
    """The synthetic graph served as the component's runtime graph; the plan-time contract empty
    and every tensor at its traced dtype (fp32 throughout) — what is under test is the axis."""
    _census._bind_target("c4140-4xv100-16GB-nvlink", None)     # a committed V100 profile: its ladders

    def install(g):
        for cache in (D._RUNTIME_GRAPHS, D._VALUE_AXES, D._INERT_AXES):
            cache.pop((MODEL, COMP), None)
        D._RUNTIME_GRAPHS[(MODEL, COMP)] = g
        return g
    monkeypatch.setattr(D, "_contract", lambda *a, **k: _Contract())
    monkeypatch.setattr(RW, "runtime_dtypes",
                        lambda g, *a, **k: {tid: m["dtype"] for tid, m in g["tensors"].items()})
    yield install
    for cache in (D._RUNTIME_GRAPHS, D._VALUE_AXES, D._INERT_AXES):
        cache.pop((MODEL, COMP), None)


def _derive(L, unhandled):
    return D.derive_component(MODEL, COMP, "float32", "triton", {"s1": L}, False, 0, 1, 1, unhandled)


def test_the_axis_is_found_bounded_and_carried(synthetic):
    g = synthetic(graph())
    (grp,) = D.value_axes(MODEL, COMP)
    (ax,) = grp
    assert ax.uid == "aten.repeat_interleave::0" and ax.trace == 34 and ax.refused is None
    shape = D._shape_fn(g, COMP, {"s1": 94}, {}, {})
    assert ax.extent_range(g, shape) == (94, 94 * 50)          # [1, 50] frames per phoneme
    assert {"aten.bmm::0", "aten.convolution::0", "aten.upsample_nearest1d::0",
            "aten.convolution::1"} <= ax.cone_ops
    assert ax.carry["aten.bmm::0::out_0"][2](229) == 229
    assert ax.carry["aten.convolution::0::out_0"][2](229) == 458     # (F-1)*2 - 2 + 2 + 1 + 1
    assert ax.carry["aten.upsample_nearest1d::0::out_0"][2](229) == 458
    assert ax.carry["aten.convolution::1::out_0"][2](229) == 458
    assert ax.outputs == ["aten.convolution::1::out_0"]


@pytest.mark.parametrize("L", [23, 94])
def test_the_bisection_meets_every_class_the_envelope_holds(synthetic, tl_dot_gemms, L):
    """The derived keys equal the exhaustive derivation at EVERY extent of the envelope."""
    g = synthetic(graph())
    u = collections.Counter()
    got = set(_derive(L, u))
    assert not D.unplaced(u), u
    (grp,) = D.value_axes(MODEL, COMP)
    lo, hi = grp[0].extent_range(g, D._shape_fn(g, COMP, {"s1": L}, {}, {}))
    want = set()
    for n in range(lo, hi + 1):
        want |= set(D._derive_at(MODEL, COMP, "float32", "triton", {"s1": L}, False, 0, 1, 1,
                                 collections.Counter(), axis_values={grp[0].uid: n}, axes=grp))
    assert got == want, (len(got), len(want))
    bmm = {k[1] for uid, _q, k in got if uid == "aten.bmm::0"}
    assert 240 in bmm                                    # the run's 229 frames: the 240 class
    assert (34 in bmm) == (lo <= 34)                     # the trace's draw only where the envelope holds it


def test_the_run_extents_of_kokoro_land_in_a_derived_class(synthetic, tl_dot_gemms):
    """(640, 240, 96) at 94 phonemes (this rack's census request) and (640, 160, 55) at 55 (the
    Mac's VALIDATE request) are derived classes."""
    synthetic(graph())
    keys = {(L, k[:3]) for L in (94, 55) for uid, _q, k in _derive(L, collections.Counter())
            if uid == "aten.bmm::0"}
    assert (94, (640, 240, 96)) in keys and (55, (640, 160, 55)) in keys


def test_the_matrix_unit_places_the_gemms_keyless_and_the_derivation_says_so(synthetic):
    """The V100 profile as committed (its m8n8k4 unit, fp32 split): the fp32 bmm runs on the unit with no
    autotune key. The derivation forms none for it and records it as PLACED (`KEYLESS aten::bmm`), never as
    unhandled: a model of such ops alone is proven keyless, not refused. The fp32 convolution keeps its key
    (the unit's implicit GEMM takes native operands only, `launch_keys.matrix_unit_native`)."""
    synthetic(graph())
    u = collections.Counter()
    got = {uid for uid, _q, _k in _derive(94, u)}
    assert "aten.bmm::0" not in got and "aten.convolution::1" in got, got
    assert u[f"{D.KEYLESS}aten::bmm"] > 0, u
    assert not D.unplaced(u), u


def test_an_unbounded_axis_is_refused_by_name(synthetic):
    g = graph()
    g["ops"]["aten.sigmoid::0"]["op_type"] = "aten::exp"          # no interval rule bounds exp
    synthetic(g)
    u = collections.Counter()
    _derive(94, u)
    assert any("value-derived axis not classed" in w and "aten.sigmoid::0 (aten::exp)" in w for w in u), u


def test_an_extent_no_rule_explains_is_refused_by_name(synthetic):
    g = graph()
    y = "aten.bmm::0::out_0"
    g["tensors"]["aten.view::9::out_0"] = _t([1, 21760])
    g["ops"]["aten.view::9"] = _op("aten.view::9", "aten::view", [y], ["aten.view::9::out_0"],
                                   [T_(y), L_([1, 21760])])
    g["execution_order"].append("aten.view::9")
    synthetic(g)
    u = collections.Counter()
    _derive(94, u)
    assert any("aten.view::9 (aten::view): output dim 1 (21760)" in w for w in u), u


def test_a_given_output_size_or_a_constant_is_not_value_derived(synthetic):
    g = graph()
    g["ops"]["aten.repeat_interleave::0"]["attributes"]["kwargs"]["output_size"] = 34
    synthetic(g)
    assert D.value_axes(MODEL, COMP) == []
    g2 = copy.deepcopy(graph())
    g2["tensors"]["input::dur"]["is_parameter"] = True               # durations stored, not computed
    synthetic(g2)
    assert D.value_axes(MODEL, COMP) == []


def test_an_axis_no_kernel_reads_is_inert_not_refused(synthetic):
    """A row count only indexed downstream (a vision tower's cu_seqlens from its grid, traced at 1)
    moves no key: listed inert, never refused — a refusal there would unwrite a whole model for an
    axis that forms nothing."""
    g = graph()
    keep = g["execution_order"].index("aten.unsqueeze::0")
    for uid in g["execution_order"][keep:]:
        g["ops"].pop(uid)
    g["execution_order"] = g["execution_order"][:keep]
    g["tensors"]["aten.repeat_interleave::0::out_0"]["shape"] = [1]
    g["output_tensor_ids"] = []
    synthetic(g)
    assert D.value_axes(MODEL, COMP) == []
    (ax,) = D._INERT_AXES[(MODEL, COMP)]
    assert ax.inert and ax.refused is None
    u = collections.Counter()
    _derive(94, u)
    assert not D.unplaced(u), u


def test_an_extent_carried_through_padding_and_slicing_but_reaching_no_kernel_is_inert(synthetic):
    """A vision tower's cu_seqlens: the row count padded by one at the front, sliced back, its
    VALUES bucketized into an index that gathers the encoding a matmul reads — the values reach a
    kernel, the extent none. The padding and the slice carry the extent (each checked at the
    trace); the axis is inert, and nothing is refused."""
    g = graph()
    keep = g["execution_order"].index("aten.unsqueeze::0")
    for uid in g["execution_order"][keep:]:
        g["ops"].pop(uid)
    g["execution_order"] = g["execution_order"][:keep]
    g["output_tensor_ids"] = []
    t, ops, order = g["tensors"], g["ops"], g["execution_order"]

    def add(uid, kind, ins, shape, args, dtype="float32", **attrs):
        t[f"{uid}::out_0"] = _t(shape, dtype)
        ops[uid] = _op(uid, kind, ins, [f"{uid}::out_0"], args, **attrs)
        order.append(uid)
        return f"{uid}::out_0"
    r = "aten.repeat_interleave::0::out_0"
    p = add("aten.constant_pad_nd::0", "aten::constant_pad_nd", [r], [35], [T_(r), L_([1, 0]), S_(0)], "int64")
    s = add("aten.slice::0", "aten::slice", [p], [34], [T_(p), S_(0), S_(1), S_(2 ** 63 - 1), S_(1)], "int64")
    b = add("aten.bucketize::0", "aten::bucketize", ["input::idx", s], [23, 1],
            [T_("input::idx"), T_(s)], "int64")
    t["param::m"] = _t([1, 23, 8], param=True)
    sq = add("aten.view::0", "aten::view", [b], [23], [T_(b), L_([23])], "int64")
    x = add("aten.index_select::0", "aten::index_select", ["input::enc", sq], [1, 640, 23],
            [T_("input::enc"), S_(2), T_(sq)])
    add("aten.bmm::0", "aten::bmm", [x, "param::m"], [1, 640, 8], [T_(x), T_("param::m")])
    synthetic(g)
    assert D.value_axes(MODEL, COMP) == []
    (ax,) = D._INERT_AXES[(MODEL, COMP)]
    assert ax.inert and ax.refused is None
    assert ax.carry[p][0](229) == 230 and ax.carry[s][0](229) == 229
    u = collections.Counter()
    _derive(94, u)
    assert not D.unplaced(u), u
