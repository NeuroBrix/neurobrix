"""P-ZERO-FALLBACK-SWEEP — op-output NaN/Inf guard blind spots.

Pins the MED finding of engine audit #2 (2026-07-05) on
`GraphExecutor._check_nan_inf`:
  - the historical guard missed -inf entirely (`has_pos_inf` only), so
    -inf saturation (e.g. fp16 log-underflow) sailed through;
  - ±inf IS legitimate for mask-materialising ops (full(-inf),
    masked_fill(-inf), where(mask, -inf, x)) — those are allowlisted
    via `_INF_LEGITIMATE_OPS`; NaN has NO allowlist;
  - METADATA-class ops stay exempt by design (views alias storage
    already checked at its producing compute op; aten::empty
    legitimately exposes uninitialised memory) — the always-on
    step-boundary state gate provides the composite coverage.

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

    _INF_LEGITIMATE_OPS = GraphExecutor._INF_LEGITIMATE_OPS

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
    with pytest.raises(RuntimeError, match=r"\+inf=1"):
        _check("aten.add::1", "aten::add", out)


def test_neg_inf_raises_blind_spot_closed():
    """The historical guard only flagged +inf; -inf must raise too."""
    out = torch.randn(4)
    out[0] = float("-inf")
    with pytest.raises(RuntimeError, match=r"-inf=1"):
        _check("aten.log::1", "aten::log", out)


def test_mask_op_inf_is_legitimate():
    """±inf from mask-materialising ops is by-construction legitimate."""
    mask = torch.full((4, 4), float("-inf"))
    for op_type in ("aten::full", "aten::masked_fill", "aten::where"):
        _check("op::1", op_type, mask)


def test_mask_op_nan_still_raises():
    """NaN has no allowlist — not even for mask ops."""
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
