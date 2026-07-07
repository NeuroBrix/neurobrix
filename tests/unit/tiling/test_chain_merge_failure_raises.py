"""P-ZERO-FALLBACK-SWEEP — band-streamed chain merge failure must RAISE.

Pins the CRITICAL finding of engine audit #2 (2026-07-05): a failed
band-streamed residual-chain merge in the op-level tiling engine was
caught, WARNED, and the UN-MERGED base tensor was substituted as the
chain result — the run continued to a silently wrong final output.

Doctrine (ZERO FALLBACK): the merge failure is a crash with an
actionable message naming the chain and the op, never a substitution.

The test drives the real unified node interceptor built by
`OpLevelTilingEngine.register_into_graph_executor` on a minimal fake
GraphExecutor (compiled mode, CPU torch tensors only — no GPU), with the
chain wrapper monkeypatched to fail, and asserts:
  - the interceptor raises RuntimeError naming the chain,
  - the un-merged base tensor is NOT registered as the chain result.

Run: PYTHONPATH=src python -m pytest tests/unit/tiling/test_chain_merge_failure_raises.py
"""
from __future__ import annotations

import pytest
import torch

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
import neurobrix.kernels.ops.residual_chain as _rc
from neurobrix.core.module.tiling_engine import (
    OpLevelTilingEngine,
    OpLevelTilingPlan,
)


FORK_UID = "aten.add::10"
CHAIN_UID = "aten.conv2d::11"
MERGE_UID = "aten.add::20"


class _FakeGraphExecutor:
    """Minimal surface register_into_graph_executor touches."""

    mode = "compiled"

    def __init__(self):
        self._dag = {"ops": {}, "execution_order": [], "output_tensor_ids": []}
        self._weights = {}
        self.registered = {}

    def register_op_uid_interceptors(self, interceptors):
        self.registered.update(interceptors)


def _make_engine_with_chain():
    plan = OpLevelTilingPlan("test_component")
    plan.add_residual_chain({
        "fork_uid": FORK_UID,
        "merge_uid": MERGE_UID,
        "chain_uids": [CHAIN_UID],
        "tile_factor": 4,
        "spatial_axis": 2,
        "halo": 2,
        "shape": [1, 3, 8, 8],
    })
    return OpLevelTilingEngine(plan)


def test_failed_chain_merge_raises_and_does_not_substitute(monkeypatch):
    # Chain weights resolve (truthy) so the interceptor takes the merge
    # path; the merge itself fails — the recurring failure class the
    # audit names (e.g. "Pointer argument cannot be accessed from
    # Triton (cpu tensor?)").
    monkeypatch.setattr(_rc, "resolve_chain_weights",
                        lambda spec, dag, weights: {"w0": object()})

    def _boom(*args, **kwargs):
        raise ValueError("injected merge failure")

    monkeypatch.setattr(_rc, "band_streamed_chain_torch", _boom)

    engine = _make_engine_with_chain()
    ge = _FakeGraphExecutor()
    engine.register_into_graph_executor(ge)
    assert FORK_UID in ge.registered and CHAIN_UID in ge.registered

    # Fork op executes: natural add + stash the pending chain value.
    a = torch.ones(1, 3, 8, 8)
    b = torch.ones(1, 3, 8, 8)
    fork_out = ge.registered[FORK_UID](a, b)
    assert torch.equal(fork_out, a + b)
    assert len(engine._pending_chain) == 1
    cid = next(iter(engine._pending_chain))

    # First chain intermediate executes: the deferred merge runs and
    # fails → MUST raise, naming the chain — never substitute the
    # un-merged base tensor and continue.
    with pytest.raises(RuntimeError) as excinfo:
        ge.registered[CHAIN_UID](torch.zeros(1, 3, 8, 8))

    msg = str(excinfo.value)
    assert "Band-streamed residual-chain merge failed" in msg
    assert cid in msg
    assert "injected merge failure" in msg
    # The poison substitution is gone: nothing was registered as the
    # chain result.
    assert cid not in engine._chain_registry


def test_failed_weight_resolution_raises_and_does_not_substitute(monkeypatch):
    """Sibling path of the merge failure: resolve_chain_weights returns
    {} exactly when a required chain weight cannot be resolved. The
    chain intermediates are intercepted (never execute natively), so the
    old `_registry[cid] = t_base_pending` silently DROPPED the whole
    residual branch."""
    monkeypatch.setattr(_rc, "resolve_chain_weights",
                        lambda spec, dag, weights: {})

    engine = _make_engine_with_chain()
    ge = _FakeGraphExecutor()
    engine.register_into_graph_executor(ge)

    ge.registered[FORK_UID](torch.ones(1, 3, 8, 8), torch.ones(1, 3, 8, 8))
    cid = next(iter(engine._pending_chain))

    with pytest.raises(RuntimeError) as excinfo:
        ge.registered[CHAIN_UID](torch.zeros(1, 3, 8, 8))

    msg = str(excinfo.value)
    assert "weight resolution failed" in msg
    assert cid in msg
    assert cid not in engine._chain_registry


def test_successful_chain_merge_still_registers_result(monkeypatch):
    """Control: the fix must not break the healthy merge path."""
    monkeypatch.setattr(_rc, "resolve_chain_weights",
                        lambda spec, dag, weights: {"w0": object()})
    merged = torch.full((1, 3, 8, 8), 7.0)
    monkeypatch.setattr(_rc, "band_streamed_chain_torch",
                        lambda *a, **k: merged)

    engine = _make_engine_with_chain()
    ge = _FakeGraphExecutor()
    engine.register_into_graph_executor(ge)

    ge.registered[FORK_UID](torch.ones(1, 3, 8, 8), torch.ones(1, 3, 8, 8))
    sentinel = ge.registered[CHAIN_UID](torch.zeros(1, 3, 8, 8))
    assert sentinel is not None  # ChainSentinel referencing the registry
    cid = next(iter(engine._chain_registry))
    assert torch.equal(engine._chain_registry[cid], merged)
