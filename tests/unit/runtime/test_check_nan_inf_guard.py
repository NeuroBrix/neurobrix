"""P-ZERO-FALLBACK-SWEEP — op-output NaN/Inf guard contract.

Pins the corrected contract of `GraphExecutor._check_nan_inf`:
  - NaN is NEVER legitimate → raises (any op);
  - +Inf in a compute output is an overflow, never legitimate → raises;
  - -Inf is legitimate across the zoo (additive attention masks,
    log-underflow of a true zero, -inf padding) and is NOT distinguishable
    from a corrupting -inf after arithmetic propagation → NOT raised
    per-op. Where -inf genuinely signals corruption — the iterative-flow
    loop state — it is caught unambiguously by the always-on
    step-boundary isfinite state gate (engine audit #2 2026-07-05);
  - METADATA-class ops stay exempt by design (views alias storage already
    checked at its producing compute op; aten::empty legitimately exposes
    uninitialised memory).

This mirrors the historical NaN + positive-Inf contract: a per-op -inf
raise false-positives on legitimately masked attention (many in-zoo
models trace decomposed softmax with an additive -inf bias) and on
in-graph log-underflow (aten::log appears in 13 zoo graphs).

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_check_nan_inf_guard.py
"""
from __future__ import annotations

import pytest
import torch

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
from neurobrix.core.runtime.graph_executor import GraphExecutor
from neurobrix.kernels.classification import OpExecution


class _Ctx:
    def __init__(self, op_outputs):
        self.op_outputs = op_outputs


class _Stub:
    """Minimal surface _check_nan_inf reads (called unbound)."""

    def __init__(self, op_outputs):
        self._ctx = _Ctx(op_outputs)


def _check(op_uid, op_type, out, exec_type=OpExecution.TRITON):
    stub = _Stub({op_uid: [out]})
    GraphExecutor._check_nan_inf(stub, op_uid, op_type, exec_type)


def test_clean_output_passes():
    _check("aten.add::1", "aten::add", torch.randn(4, 4))


def test_nan_raises():
    out = torch.randn(4)
    out[0] = float("nan")
    with pytest.raises(RuntimeError, match="NaN"):
        _check("aten.add::1", "aten::add", out)


def test_pos_inf_raises():
    out = torch.randn(4)
    out[0] = float("inf")
    with pytest.raises(RuntimeError, match="Pos-Inf"):
        _check("aten.add::1", "aten::add", out)


def test_neg_inf_does_not_raise_per_op():
    """-inf is legitimate (masks / log-underflow) and is left to the
    step-boundary state gate; the per-op guard must NOT false-positive."""
    out = torch.randn(4)
    out[0] = float("-inf")
    # log-underflow and an additive-mask propagation both reach here as
    # plain compute ops; neither may raise.
    _check("aten.log::1", "aten::log", out)
    _check("aten.add::1", "aten::add", out)


def test_mask_neg_inf_is_legitimate():
    """-inf from any op (mask materialisation or otherwise) never raises."""
    mask = torch.full((4, 4), float("-inf"))
    for op_type in ("aten::full", "aten::masked_fill", "aten::where", "aten::add"):
        _check("op::1", op_type, mask)


def test_pos_inf_raises_even_for_mask_ops():
    """+Inf is overflow and never legitimate, mask op or not."""
    out = torch.full((4,), float("inf"))
    with pytest.raises(RuntimeError, match="Pos-Inf"):
        _check("op::1", "aten::masked_fill", out)


def test_nan_still_raises_for_mask_ops():
    """NaN has no exemption — not even for mask ops."""
    out = torch.full((4,), float("nan"))
    with pytest.raises(RuntimeError, match="NaN"):
        _check("op::1", "aten::masked_fill", out)


def test_metadata_exec_type_exempt():
    out = torch.full((4,), float("nan"))
    _check("aten.view::1", "aten::view", out,
           exec_type=OpExecution.METADATA)


def test_message_names_op():
    out = torch.full((4,), float("nan"))
    with pytest.raises(RuntimeError) as excinfo:
        _check("aten.softmax::7", "aten::softmax", out)
    msg = str(excinfo.value)
    assert "aten.softmax::7" in msg and "aten::softmax" in msg
