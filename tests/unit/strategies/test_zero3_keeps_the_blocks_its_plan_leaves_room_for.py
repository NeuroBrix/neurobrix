"""zero3 keeps resident, across passes, the weights its plan leaves room for — and streams the rest.

The ratchet moved every block host->card on every pass and released them all at the pass end.
An autoregressive decode runs one pass per token, so it streamed the WHOLE component per token
while the card stayed nearly empty: Janus-Pro-7B on a 16 GB V100, lazy_sequential with its
language model on zero3, moved 12.4 GB over PCIe per image token (2.64 s a step, 576 steps,
1 520 s against a 1 200 s cap), the ratchet's peak 2 152 MB of the card (2026-10-09).

Prism now prices the room (`ComponentAllocation.resident_weight_mb`, `_zero3_resident_budgets`)
and the ratchet keeps that much resident: the non-block weights first, then the blocks in
execution order. These cells drive the ratchet with a recorder in place of the device moves and
pin: a budget of 0 is the old ratchet, unchanged; a budget keeps exactly the blocks it covers,
never evicts them, streams only the others, and binds the resident non-block weights again at
every pass start (both engines re-bind every slot from the host weights at the start of a run —
without it the final norm met its host weight on the second decode step, `custom.rms_norm::60`
"cuda:0 and cpu", the first try of this fix).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from neurobrix.core.strategies.zero3 import Zero3Strategy

MIB = 1024 * 1024
N_BLOCKS = 4
OPS_PER_BLOCK = 2


def _blocks():
    blocks = {-1: {"first_op": 0, "last_op": N_BLOCKS * OPS_PER_BLOCK - 1,
                   "weight_tensor_ids": ["param::norm.weight"]}}
    for b in range(N_BLOCKS):
        blocks[b] = {"first_op": b * OPS_PER_BLOCK, "last_op": b * OPS_PER_BLOCK + OPS_PER_BLOCK - 1,
                     "weight_tensor_ids": [f"param::layers.{b}.w"]}
    return blocks


class _Seq:
    def __init__(self):
        self._blocks = _blocks()

    def get_op_blocks(self):
        return self._blocks

    def mark_cpu_weighted_ops_for_transfer(self, dev):
        return 0

    def override_weightless_op_devices(self, dev):
        pass


class _Exec:
    def __init__(self):
        self._compiled_seq = _Seq()
        self._triton_seq = None
        # 1 MiB of host weight per block and for the non-block weights
        self._weights = {"norm.weight": torch.zeros(MIB // 4)}
        for b in range(N_BLOCKS):
            self._weights[f"layers.{b}.w"] = torch.zeros(MIB // 4)


class _Recording(Zero3Strategy):
    """The ratchet's decisions, with the device moves recorded instead of made."""

    def _prefetch_block(self, state, bidx, is_triton):
        if bidx in state["gpu_cache"] or bidx not in state["blocks"]:
            return
        self.log.append(("fetch", bidx))
        state["gpu_cache"][bidx] = {"moved": True}

    def _wait_for_block(self, state, bidx, is_triton):
        pass

    def _install_block_on_arena(self, state, bidx):
        self.log.append(("bind", bidx))

    def _evict_block(self, state, bidx, is_triton):
        if bidx in state["gpu_cache"]:
            self.log.append(("evict", bidx))
            state["gpu_cache"].pop(bidx)


def _strategy(resident_mb):
    s = object.__new__(_Recording)
    s.exec_device = "cuda:0"
    s._ratchet = {}
    s._installed = {}
    s.log = []
    s.context = SimpleNamespace(allocations={"lm": {"device": "zero3:cuda:0", "strategy": "zero3",
                                                    "resident_weight_mb": float(resident_mb)}})
    return s


def _passes(s, n, monkeypatch):
    # no stream on a test host: the ratchet's synchronous path
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no device")))
    ex = _Exec()
    s._install("lm", ex)
    logs = []
    for _ in range(n):
        start = len(s.log)
        for op in range(N_BLOCKS * OPS_PER_BLOCK):
            ex._persistent_pre_op_callback(op, None)
        ex._post_run_hook()
        logs.append(s.log[start:])
    return logs, s._ratchet["lm"]


def _of(log, kind):
    return [b for k, b in log if k == kind]


def test_a_zero_budget_is_the_old_ratchet(monkeypatch):
    logs, state = _passes(_strategy(0), 3, monkeypatch)
    assert state["resident"] == set()
    for log in logs:
        assert sorted(_of(log, "fetch")) == list(range(N_BLOCKS)), log
        assert sorted(_of(log, "evict")) == list(range(N_BLOCKS)), log
    assert state["gpu_cache"] == {}


def test_a_budget_keeps_the_blocks_it_covers_and_streams_the_rest(monkeypatch):
    # 3 MiB: the non-block weights, then blocks 0 and 1; blocks 2 and 3 stream
    logs, state = _passes(_strategy(3), 3, monkeypatch)
    assert state["resident"] == {-1, 0, 1}
    assert state["streamed"] == [2, 3]
    # the first pass moves everything once; the later passes only the streamed blocks
    assert sorted(_of(logs[0], "fetch")) == [-1, 0, 1, 2, 3]
    for log in logs[1:]:
        assert sorted(_of(log, "fetch")) == [2, 3], log
    for log in logs:
        assert sorted(_of(log, "evict")) == [2, 3], log
    assert sorted(state["gpu_cache"]) == [-1, 0, 1]


def test_the_streamed_prefetch_starts_while_the_resident_blocks_run(monkeypatch):
    """Entering block 0 already starts block 2 — the transfer overlaps the resident blocks'
    compute instead of waiting for the block before it."""
    logs, _ = _passes(_strategy(3), 2, monkeypatch)
    later = logs[1]
    first_bind = later.index(("bind", 0))
    assert later[first_bind + 1] == ("fetch", 2), later


def test_the_resident_non_block_weights_are_bound_again_every_pass(monkeypatch):
    """The run's own `bind_weights` puts the host weights back in every slot; the ratchet binds
    the resident non-block weights before the pass's first op, every pass."""
    logs, _ = _passes(_strategy(3), 3, monkeypatch)
    # the first pass binds them where the ratchet is built, before its first op
    assert logs[0][:2] == [("fetch", -1), ("bind", -1)], logs[0]
    for log in logs[1:]:
        assert log[0] == ("bind", -1), log


def test_a_budget_below_the_non_block_weights_still_keeps_the_blocks_it_covers(monkeypatch):
    s = _strategy(1)
    s._ratchet = {}
    ex = _Exec()
    ex._weights["norm.weight"] = torch.zeros(2 * MIB // 4)   # 2 MiB, over the 1 MiB budget
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no device")))
    s._install("lm", ex)
    ex._persistent_pre_op_callback(0, None)
    assert s._ratchet["lm"]["resident"] == {0}


@pytest.mark.parametrize("budget_mb", [0, 2, 5])
def test_nothing_resident_is_left_on_the_card_after_uninstall(monkeypatch, budget_mb):
    s = _strategy(budget_mb)
    _passes(s, 2, monkeypatch)
    state = s._ratchet["lm"]
    s._uninstall("lm")
    assert state["gpu_cache"] == {}
