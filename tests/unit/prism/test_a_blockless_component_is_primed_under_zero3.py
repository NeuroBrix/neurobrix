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


# ── the flipped op computes on the execution device, in both engines ──
#
# SANA-Video's transformer under lazy_sequential+zero3 (32 GB V100, 2026-09-27):
# `param::scale_shift` -> unsqueeze -> unsqueeze -> add(time embedding). The first
# unsqueeze is the flipped op and its ONLY tensor is the host weight; the compiled
# priming set its device to the weight's (CPU), the slow path found no CUDA argument
# to prefer, the view was made on the host, and `aten.add::352` met it on the card.

class _Op:
    def __init__(self, weight_slots):
        self.weight_input_slots = weight_slots
        self.device = None
        self.needs_transfer = False
        self.device_idx = None


class _HostTensor:
    def __init__(self, device):
        self.device = device
        self._device = device.type if hasattr(device, "type") else device


def test_the_compiled_flip_targets_the_execution_device():
    import torch
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence

    seq = object.__new__(CompiledSequence)
    weight_only = _Op([0])
    seq._arena = [_HostTensor(torch.device("cpu"))]
    seq._ops = [weight_only, _Op([])]
    exec_dev = torch.device("cuda:0")
    assert CompiledSequence.mark_cpu_weighted_ops_for_transfer(seq, exec_dev) == 1
    assert weight_only.needs_transfer is True
    assert weight_only.device == exec_dev, (
        f"a flipped op computes on the execution device, not on its weight's "
        f"{weight_only.device} — an op whose only tensor is the weight has no "
        f"other argument to take the device from")


def test_the_triton_mirror_already_targets_the_execution_device():
    from neurobrix.triton.sequence import TritonSequence

    seq = object.__new__(TritonSequence)
    weight_only = _Op([0])
    seq._arena = [_HostTensor("cpu")]
    seq._ops = [weight_only]
    assert TritonSequence.mark_cpu_weighted_ops_for_transfer(seq, 0) == 1
    assert weight_only.device_idx == 0 and weight_only.needs_transfer is True
