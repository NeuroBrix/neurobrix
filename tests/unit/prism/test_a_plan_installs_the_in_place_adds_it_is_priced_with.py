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
