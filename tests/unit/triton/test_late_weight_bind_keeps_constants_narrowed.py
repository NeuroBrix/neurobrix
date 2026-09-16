"""The lazy weight bind must not undo the seq-dependent narrowing.

Contract, written in the engine's own words (`sequence.py`, the late-bind
branch): "the skip must never be observable on the op-by-op path."

It WAS observable. `_run_triton_compiled` defers `bind_weights` when
`would_replay()` says yes, so the order inverts:

    eager : bind_weights -> bind_inputs -> bind_symbols -> narrow -> run
    lazy  : bind_inputs -> bind_symbols -> narrow -> run -> bind_weights

and `bind_weights` both clears `_seq_constant_originals` and rebinds every
weight slot to its FULL-SIZE tensor. It must do that — the cached originals are
views into the previous bind's arena — but on the late path it lands AFTER the
narrowing and silently discards it. The ops then execute against full-size
seq-dependent constants: no exception, no wrong shape anywhere the launcher can
see, just wrong values.

Measured end to end on swin2SR-classical-sr-x2-64 (Apple M4 Pro, triton-ext):

    NBX_LAZY_BIND=1 (default), before  std 9.201   image washed to the model's
                                                   own mean, judged wrong by eye
    NBX_LAZY_BIND=0                    std 103.123 correct
    NBX_LAZY_BIND=1 (default), after   BYTE-IDENTICAL to the NBX_LAZY_BIND=0 arm

Identical kernels and identical launch order on both arms (2408 launches, same
counts) — only the constants differed. This is a TRUNK defect, not a Metal one:
any model with seq-dependent constants meets it when the lazy bind skips the
eager bind and replay then declines.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from neurobrix.triton.sequence import TritonSequence


class _Resolver:
    def __init__(self, val): self._val = val
    def get(self, _sym): return self._val


def _seq(monkeypatch, weight, runtime_len):
    """A TritonSequence carrying only what these two methods touch."""
    s = object.__new__(TritonSequence)
    s._arena = {0: weight}
    s._tid_to_slot = {"t0": 0}
    s._weight_ids = ["t0"]
    s.dag = {"tensors": {"t0": {"weight_name": "constant_T_pos_bias"}}}
    s._pretranspose_weights = set()
    s._seq_constant_originals = {}
    s._seq_dependent_constants = [(0, 0, "s0", weight.shape[0])]
    s._symbol_resolver = _Resolver(runtime_len)
    monkeypatch.setattr(s, "_execute_const_fold", lambda w: None, raising=False)
    monkeypatch.setattr(s, "compute_op_devices", lambda: None, raising=False)
    return s


def _w(n):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    return NBXTensor.from_numpy(np.arange(n * 2, dtype=np.float32).reshape(n, 2))


def test_a_late_bind_undoes_the_narrowing_unless_it_is_redone(monkeypatch):
    full, runtime = 8, 3
    s = _seq(monkeypatch, _w(full), runtime)

    s.update_seq_dependent_constants()
    assert s._arena[0].shape[0] == runtime, "precondition: the constant narrows"

    # what the late bind does, on its own
    s.bind_weights({"constant_T_pos_bias": _w(full)})
    assert s._arena[0].shape[0] == full, (
        "this is the defect: bind_weights rebinds the full-size constant and "
        "clears the narrowing cache")
    assert s._seq_constant_originals == {}

    # and the repair the late-bind branch must perform
    s.update_seq_dependent_constants()
    assert s._arena[0].shape[0] == runtime, (
        "re-narrowing after a late bind must restore the runtime extent")


def test_the_late_bind_branch_renarrows():
    """Guard the call site: the end-to-end proof is the swin2SR artefact, but
    nothing else keeps this ordering honest if someone edits `run`."""
    src = inspect.getsource(TritonSequence.run)
    late = src.split("self.bind_weights(_parked)")
    assert len(late) == 2, "the late-bind call moved; re-read this test"
    assert "update_seq_dependent_constants()" in late[1], (
        "the late bind must be followed by update_seq_dependent_constants(): "
        "bind_weights rebinds full-size constants and clears the narrowing "
        "cache, so without it the lazy path silently computes on un-narrowed "
        "seq-dependent constants")
