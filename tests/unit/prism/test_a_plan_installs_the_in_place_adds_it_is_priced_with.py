"""The figure a strategy is gated on is the figure the plan executes under — the in-place adds.

`PrismSolver._compute_memory` prices a component's large residual adds IN PLACE whenever it has
candidates (`_identify_inplace_add_candidates_static`), including when no op overflows — and the
strategy gates compare that figure. `_detect_op_level_tiling_pairs` installed nothing for a
component with no overflowing op, so the runtime ran those adds out of place: a third buffer the
gate never saw. Measured on the Mac, 2026-09-28: Sana_1600M_4Kpx_BF16 triton at 3072x4096 placed
single_gpu on a 6 144 MB tiled activation, no `[OpLevelTiling]` line in the log, aten.add::86 run
full size (3 x 3.2 GB), an 18 157 MB footprint and an out-of-memory in the decode.

What each test would do if the code were wrong: without the installation the first test finds no
plan for the component (RED, seen); a detector that installed in-place adds regardless of the
estimate's own threshold would make the second test find a plan (RED).

SIZE IS THE REQUEST'S, NOT THE TRACE'S (2026-10-12). Sana_1600M_4Kpx_BF16's VAE retraced at its
own 7x10 latent (Forge 4730493) carries 224x320 traced extents at its merge adds; the detectors read
the DAG's traced `output_shapes`, every add fell under the 1 GiB threshold, and the 3072x4096 plan
priced and installed none of them (15 360 -> 18 432 MB). The third test holds the request's sizing:
SEEN RED with `_request_shape_of` returning None (the traced extents) — no candidate, no plan.
"""
from __future__ import annotations

from types import SimpleNamespace

from neurobrix.core.prism.solver import PrismSolver
from neurobrix.core.prism.profiler import InputConfig

GiB = 1024 ** 3


def _graph(h: int, w: int) -> dict:
    """x -> conv-free stand-in: two producers of one (1,128,h,w) fp16 tensor each, joined by a residual add."""
    shape = [1, 128, h, w]
    t = lambda tid: {"shape": shape, "dtype": "float16"}  # noqa: E731
    return {
        "tensors": {"in_a": t("in_a"), "in_b": t("in_b"), "a": t("a"), "b": t("b"), "out": t("out")},
        "input_tensor_ids": ["in_a", "in_b"],
        "output_tensor_ids": ["out"],
        "execution_order": ["aten.relu::0", "aten.relu::1", "aten.add::0"],
        "ops": {
            "aten.relu::0": {"op_uid": "aten.relu::0", "op_type": "aten::relu", "input_tensor_ids": ["in_a"],
                             "output_tensor_ids": ["a"], "input_shapes": [shape], "output_shapes": [shape],
                             "attributes": {"args": []}},
            "aten.relu::1": {"op_uid": "aten.relu::1", "op_type": "aten::relu", "input_tensor_ids": ["in_b"],
                             "output_tensor_ids": ["b"], "input_shapes": [shape], "output_shapes": [shape],
                             "attributes": {"args": []}},
            "aten.add::0": {"op_uid": "aten.add::0", "op_type": "aten::add", "input_tensor_ids": ["a", "b"],
                            "output_tensor_ids": ["out"], "input_shapes": [shape, shape], "output_shapes": [shape],
                            "attributes": {"args": []}},
        },
    }


def _detect(graph: dict, card_mb: int = 32768):
    solver = PrismSolver()
    comp = SimpleNamespace(name="vae", graph=graph)
    dev = SimpleNamespace(memory_mb=card_mb, get_device_string=lambda: "cuda:0")
    profile = SimpleNamespace(devices=[dev])
    alloc = SimpleNamespace(device="cuda:0", memory_mb=300, shard_map={})
    return solver._detect_op_level_tiling_pairs(
        container=None, components=[comp], allocations={"vae": alloc}, profile=profile,
        input_config=InputConfig(batch_size=1, height=1024, width=1024), target_dtype_str="float16")


def test_an_add_priced_in_place_is_installed_in_place_when_nothing_overflows():
    g = _graph(4096, 4096)                 # 4 GiB per tensor in fp16: 12 GiB out of place, far under 0.85 x 32 GiB
    priced = PrismSolver()._identify_inplace_add_candidates_static(g)
    assert priced == [("aten.add::0", 0)], priced          # the estimate prices it in place
    plans = _detect(g)
    assert "vae" in plans, "the estimate priced the add in place, so the plan must carry it"
    assert plans["vae"].inplace_adds == priced
    assert not plans["vae"].fusion_pairs and not plans["vae"].tiled_ops and not plans["vae"].residual_chains


def test_an_add_the_estimate_prices_out_of_place_installs_nothing():
    g = _graph(256, 256)                   # 32 MiB fp32-equivalent: under the estimate's 1 GiB in-place threshold
    assert PrismSolver()._identify_inplace_add_candidates_static(g) == []
    assert _detect(g) == {}


def _symbolic_graph(trace_h: int, trace_w: int, scale: int) -> dict:
    """The same residual add, its spatial dims SYMBOLIC (latent height/width x scale), traced small."""
    def dim(sym, trace):
        return {"type": "mul", "left": {"type": "symbol", "id": sym, "trace": trace}, "right": scale,
                "trace": trace * scale}
    shape = [1, 128, trace_h * scale, trace_w * scale]
    sym = {"dims": [{"type": "symbol", "id": "s0", "trace": 1}, 128, dim("s1", trace_h), dim("s2", trace_w)],
           "concrete": shape}
    g = _graph(shape[2], shape[3])
    for t in g["tensors"].values():
        t["symbolic_shape"] = sym
    g["symbolic_context"] = {"symbols": {
        "s0": {"name": "batch", "trace_value": 1},
        "s1": {"name": "height", "trace_value": trace_h},
        "s2": {"name": "width", "trace_value": trace_w}}}
    return g


def test_an_add_is_sized_at_the_request_not_at_the_trace():
    g = _symbolic_graph(7, 10, 32)         # traced at 224x320: 36 MiB fp32-equivalent, under 1 GiB
    request = InputConfig(batch_size=1, height=3072, width=4096, dtype="float16", vae_scale=32)
    solver = PrismSolver()
    assert solver._identify_inplace_add_candidates_static(g) == []          # the trace's answer
    priced = solver._identify_inplace_add_candidates_static(g, input_config=request)
    assert priced == [("aten.add::0", 0)], priced                           # 6 GiB at 3072x4096
    comp = SimpleNamespace(name="vae", graph=g)
    dev = SimpleNamespace(memory_mb=32768, get_device_string=lambda: "cuda:0")
    alloc = SimpleNamespace(device="cuda:0", memory_mb=300, shard_map={})
    plans = solver._detect_op_level_tiling_pairs(
        container=None, components=[comp], allocations={"vae": alloc},
        profile=SimpleNamespace(devices=[dev]), input_config=request, target_dtype_str="float16")
    assert "vae" in plans and plans["vae"].inplace_adds == priced, plans


# THE RUNTIME SITE (2026-10-12). `OpLevelTilingEngine.register_into_graph_executor` re-detected the
# in-place adds from the DAG whenever a plan carried none — at the TRACE's `output_shapes`, the
# same class the two tests above hold at the plan. Prism now hands its request-sized decision over
# with EVERY plan it emits (also the overflow / residual-chain branch, which handed nothing over),
# and the runtime refuses a plan that carries no decision, by name. What each test would do if the
# code were wrong: a plan built without the handover would carry `inplace_adds is None` (RED in the
# fourth test, SEEN); a runtime that re-detected instead of refusing would install the add at the
# trace's extents and not raise (RED in the fifth, SEEN).

def test_a_plan_from_the_chain_branch_carries_the_request_sized_adds(monkeypatch):
    g = _symbolic_graph(7, 10, 32)
    request = InputConfig(batch_size=1, height=3072, width=4096, dtype="float16", vae_scale=32)
    solver = PrismSolver()
    spec = {"fork_uid": "aten.relu::0", "merge_uid": "aten.add::0", "chain_uids": ["aten.relu::1"],
            "spatial_axis": 2, "halo": 0, "shape": [1, 128, 3072, 4096], "bytes_fp32": 6 * GiB}
    monkeypatch.setattr(PrismSolver, "_identify_residual_chain_specs", lambda self, graph, ic=None: [spec])
    comp = SimpleNamespace(name="vae", graph=g)
    dev = SimpleNamespace(memory_mb=32768, get_device_string=lambda: "cuda:0")
    alloc = SimpleNamespace(device="cuda:0", memory_mb=300, shard_map={})
    plans = solver._detect_op_level_tiling_pairs(
        container=None, components=[comp], allocations={"vae": alloc},
        profile=SimpleNamespace(devices=[dev]), input_config=request, target_dtype_str="float16")
    assert "vae" in plans and plans["vae"].residual_chains, plans       # the chain branch ran
    assert plans["vae"].inplace_adds == [("aten.add::0", 0)], plans["vae"].inplace_adds


def test_the_runtime_refuses_a_plan_without_the_in_place_decision_by_name():
    import pytest
    from neurobrix.core.module.tiling_engine import OpLevelTilingEngine, OpLevelTilingPlan
    g = _graph(4096, 4096)                 # an add the trace extents alone would put in place

    class _Exec:
        mode = "compiled"
        _dag = g
        _weights = {}

        def register_op_uid_interceptors(self, interceptors, groups=(), planned=()):
            self.registered = dict(interceptors)
            return len(interceptors)

    plan = OpLevelTilingPlan("vae")
    plan.add_tiled_op("aten.relu::0", "aten::convolution", 2)   # a non-empty plan, no add decision
    assert plan.inplace_adds is None
    ex = _Exec()
    with pytest.raises(RuntimeError, match=r"'vae': the plan carries no in-place-add decision"):
        OpLevelTilingEngine(plan).register_into_graph_executor(ex)
    assert plan.inplace_adds is None and not hasattr(ex, "registered")
