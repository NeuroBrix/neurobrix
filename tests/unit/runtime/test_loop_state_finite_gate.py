"""P-ZERO-FALLBACK-SWEEP — always-on diffusion loop-state NaN/Inf gate.

Pins the MED finding of engine audit #2 (2026-07-05): the only NaN/Inf
check on the diffusion loop state was DEBUG-gated and never raised
(print-and-continue). The fix is an always-on `isfinite` gate at the
step boundary — one scan per step on the state tensor, never per-op —
that RAISES with op/step context, mirrored separately in both engines
(R30; the triton mirror is NBXTensor + Triton kernels only, R33).

The gate must catch NaN, +inf AND -inf (the -inf blind spot of the
historical guards), and must NOT fire on finite, integer, or non-tensor
state.

CPU coverage: the compiled gate runs fully on CPU torch tensors; the
triton gate's raise/skip logic is driven through monkeypatched kernel
wrappers (the isfinite/all Triton kernels themselves are exercised by
the GPU gates).

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_loop_state_finite_gate.py
"""
from __future__ import annotations

import math

import pytest
import torch

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
from neurobrix.core.flow.iterative_process import (
    _gate_loop_state_finite as _gate_compiled,
)
from neurobrix.triton.flow.iterative_process import (
    _gate_loop_state_finite as _gate_triton,
)
from neurobrix.kernels.nbx_tensor import NBXDtype, NBXTensor


# ── compiled engine gate (full CPU coverage) ─────────────────────────

def test_compiled_finite_state_passes():
    _gate_compiled(torch.randn(2, 4, 8, 8), 0, 999.0, "model.transformer")


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")],
                         ids=["nan", "pos_inf", "neg_inf"])
def test_compiled_nonfinite_state_raises_with_context(poison):
    state = torch.randn(2, 4, 8, 8)
    state[0, 0, 0, 0] = poison
    with pytest.raises(RuntimeError) as excinfo:
        _gate_compiled(state, 5, torch.tensor(981.0), "model.transformer")
    msg = str(excinfo.value)
    assert "step 6" in msg                 # step context (1-based)
    assert "model.transformer" in msg      # op/component context
    assert "981" in msg                    # timestep context
    assert "ZERO FALLBACK" in msg


def test_compiled_gate_skips_non_float_and_non_tensor():
    _gate_compiled(torch.ones(4, dtype=torch.int64), 0, 0.0, "c")
    _gate_compiled(None, 0, 0.0, "c")
    _gate_compiled("not a tensor", 0, 0.0, "c")


def test_compiled_gate_counts_offenders_in_message():
    state = torch.zeros(8)
    state[1] = float("nan")
    state[2] = float("-inf")
    with pytest.raises(RuntimeError) as excinfo:
        _gate_compiled(state, 0, 1.0, "c")
    msg = str(excinfo.value)
    assert "1 NaN" in msg and "1 Inf" in msg


# ── triton engine gate (kernel wrappers monkeypatched) ───────────────

def _nbx_stub(dtype: NBXDtype, shape=(2, 4)) -> NBXTensor:
    """Real-type NBXTensor stub without GPU allocation: bypass __init__
    and set only the attributes the gate reads (isinstance check stays
    honest; __del__ is disarmed via owns_data=False)."""
    t = NBXTensor.__new__(NBXTensor)
    t._dtype = dtype
    t._shape = tuple(shape)
    t._owns_data = False
    t._data_ptr = 0
    t._device = "cuda"
    return t


class _ScalarBool:
    def __init__(self, v: bool):
        self._v = v

    def item(self):
        return self._v


def test_triton_finite_state_passes(monkeypatch):
    import neurobrix.kernels.wrappers as w
    monkeypatch.setattr(w, "isfinite_wrapper", lambda x: x)
    monkeypatch.setattr(w, "all_wrapper", lambda x: _ScalarBool(True))
    _gate_triton(_nbx_stub(NBXDtype.float32), 0, 999.0, "model.transformer")


def test_triton_nonfinite_state_raises_with_context(monkeypatch):
    import neurobrix.kernels.wrappers as w
    monkeypatch.setattr(w, "isfinite_wrapper", lambda x: x)
    monkeypatch.setattr(w, "all_wrapper", lambda x: _ScalarBool(False))
    with pytest.raises(RuntimeError) as excinfo:
        _gate_triton(_nbx_stub(NBXDtype.float16), 5, 981.0,
                     "model.transformer")
    msg = str(excinfo.value)
    assert "step 6" in msg
    assert "model.transformer" in msg
    assert "981" in msg
    assert "ZERO FALLBACK" in msg


def test_triton_gate_skips_non_float_and_non_nbx(monkeypatch):
    import neurobrix.kernels.wrappers as w

    def _must_not_run(x):  # pragma: no cover - failure path
        raise AssertionError("kernel wrapper must not be invoked")

    monkeypatch.setattr(w, "isfinite_wrapper", _must_not_run)
    monkeypatch.setattr(w, "all_wrapper", _must_not_run)
    _gate_triton(_nbx_stub(NBXDtype.int64), 0, 0.0, "c")
    _gate_triton(None, 0, 0.0, "c")
    _gate_triton(math.pi, 0, 0.0, "c")
