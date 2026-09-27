"""zero3 primes every component it installs on, blocks or none.

The ratchet moves transformer blocks host->card two at a time; everything
outside a block (embedding, final norm, a head) reaches the card through ONE
priming sweep that flips its host-weighted ops to the transfer path. The
sweep ran only once a ratchet state existed, and `_build_ratchet_state`
returned None for a component with no block structure — so a head whose four
weights all sit in block -1 kept them on the host, and its first GEMM died:

    Failed at op aten.addmm::1: ... mat1 is on cuda:0, different from other
    tensors on cpu

(Janus-Pro-7B native, zero3 on a 16 GB V100, `gen_head`, 2026-09-27; the
language model beside it, which has blocks, was primed and ran.)
"""
from __future__ import annotations

from neurobrix.core.strategies.zero3 import Zero3Strategy


class _Seq:
    def __init__(self, blocks):
        self._blocks = blocks
        self.flipped_on = None
        self.weightless_on = None

    def get_op_blocks(self):
        return self._blocks

    def mark_cpu_weighted_ops_for_transfer(self, dev):
        self.flipped_on = dev
        return 4

    def override_weightless_op_devices(self, dev):
        self.weightless_on = dev


class _Exec:
    def __init__(self, seq):
        self._compiled_seq = seq
        self._triton_seq = None
        self._weights = {}


def _strategy():
    s = object.__new__(Zero3Strategy)
    s.exec_device = "cuda:0"
    s._ratchet = {}
    s._installed = {}
    return s


def _first_op(strategy, name, executor):
    strategy._install(name, executor)
    executor._persistent_pre_op_callback(0, None)


def test_a_head_with_only_block_minus_one_is_primed():
    seq = _Seq({-1: {"first_op": 0, "last_op": 3, "weight_tensor_ids": ["param::head.weight"]}})
    s = _strategy()
    _first_op(s, "gen_head", _Exec(seq))
    assert seq.flipped_on is not None, "the head's host weights were never flipped to the transfer path"
    assert s._ratchet["gen_head"]["primed"] is True


def test_a_component_with_no_blocks_at_all_is_primed():
    seq = _Seq({})
    s = _strategy()
    _first_op(s, "aligner", _Exec(seq))
    assert seq.flipped_on is not None


def test_a_prime_only_state_moves_no_block_and_resets_clean():
    seq = _Seq({-1: {"first_op": 0, "last_op": 3, "weight_tensor_ids": []}})
    s = _strategy()
    ex = _Exec(seq)
    _first_op(s, "gen_head", ex)
    for i in range(1, 4):
        ex._persistent_pre_op_callback(i, None)
    assert s._ratchet["gen_head"]["gpu_cache"] == {}
    ex._post_run_hook()
    assert s._ratchet["gen_head"]["current_block"] == -1


def test_no_sequence_yet_is_retried_not_primed():
    """Before the compiled sequence exists there is nothing to sweep; the
    callback returns and tries again at the next op."""
    s = _strategy()
    ex = _Exec(None)
    ex._compiled_seq = None
    _first_op(s, "gen_head", ex)
    assert "gen_head" not in s._ratchet
