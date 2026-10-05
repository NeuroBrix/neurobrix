"""Every op Prism's plan tiles runs tiled, or the run is refused by name (ZERO FALLBACK).

The budget a plan is accepted under counts each planned op banded (`OpLevelTilingPlan`: fusion
pairs, standalone tiled ops, rank-5 conv chunks, residual chains). `register_into_graph_executor`
used to hand the executor only the planned ops it had wired (`planned=[u for u in planned if u in
interceptors]`) and to log a tiled op of a type it had no implementation for at debug level: the
op then ran whole — the allocation the plan was cut to avoid — and nothing said so. A residual
chain left inactive (a mode without the chain wrapper, the NBX_TRITON_CHAIN_WRAPPER=0 kill
switch) vanished the same way.

Injections (seen red, then restored green — STATE.md of the op_tiler campaign, 2026-10-05):
  A. the `unwired` refusal removed -> the residual-chain test red (nothing raised).
  B. the tiled-op `else` branch back to a debug log -> the unknown-type test red.
"""
from __future__ import annotations

import pytest

from neurobrix.core.module.tiling_engine import OpLevelTilingEngine, OpLevelTilingPlan

SHAPE = [1, 8, 16, 16]


class _Executor:
    def __init__(self, mode):
        self.mode = mode
        self.registered = None
        self._op_uid_interceptors = {}
        self._dag = {"ops": {
            "conv": {"op_uid": "conv", "op_type": "aten::convolution",
                     "input_tensor_ids": ["x", "w"], "output_tensor_ids": ["t"],
                     "input_shapes": [SHAPE, [8, 8, 3, 3]], "output_shapes": [SHAPE]},
            "conv2": {"op_uid": "conv2", "op_type": "aten::convolution",
                      "input_tensor_ids": ["t", "w"], "output_tensor_ids": ["y"],
                      "input_shapes": [SHAPE, [8, 8, 3, 3]], "output_shapes": [SHAPE]},
        }, "execution_order": ["conv", "conv2"], "output_tensor_ids": ["y"], "tensors": {}}

    def register_op_uid_interceptors(self, interceptors, groups=(), planned=()):
        self.registered = (sorted(interceptors), list(planned))
        return len(interceptors)


def test_a_planned_tiled_conv_reaches_the_executor_as_planned():
    plan = OpLevelTilingPlan("vae")
    plan.add_tiled_op("conv", "aten::convolution", 4)
    ex = _Executor("compiled")
    OpLevelTilingEngine(plan).register_into_graph_executor(ex)
    assert "conv" in ex.registered[0] and ex.registered[1] == ["conv"], ex.registered


def test_a_planned_op_of_a_type_without_a_tiled_implementation_is_refused_by_name():
    plan = OpLevelTilingPlan("vae")
    plan.add_tiled_op("conv", "aten::cudnn_convolution", 4)
    with pytest.raises(RuntimeError, match=r"conv \(aten::cudnn_convolution\).*no tiled implementation"):
        OpLevelTilingEngine(plan).register_into_graph_executor(_Executor("compiled"))


def test_a_planned_residual_chain_the_mode_does_not_run_is_refused_by_name(monkeypatch):
    monkeypatch.setenv("NBX_TRITON_CHAIN_WRAPPER", "0")
    plan = OpLevelTilingPlan("vae")
    plan.residual_chains = [{"fork_uid": "conv", "merge_uid": "conv2", "chain_uids": []}]
    ex = _Executor("triton")
    with pytest.raises(RuntimeError, match=r"2 op\(s\) no interceptor runs in mode 'triton'.*conv.*residual chains"):
        OpLevelTilingEngine(plan).register_into_graph_executor(ex)
    assert ex.registered is None
