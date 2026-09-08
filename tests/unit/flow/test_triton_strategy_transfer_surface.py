"""The triton strategies' transfer surface: no silent no-op, no wrong card left current.

Three defects lived here, and each of them reads as success:

  a device string the surface did not recognise fell through and returned the tensor
  UNCHANGED — a transfer that moved nothing, which on a backend whose devices are not
  spelled `cuda` is every transfer of every request;

  a cross-card move went through `NBXTensor.to_cuda`, whose D2D memcpy is enqueued on the
  TARGET card's stream and does not wait for the SOURCE card's — a peer copy issued right
  after a component's kernels reads a buffer they may still be writing;

  `synchronize_device` selected the named card and never put back the one it found, so every
  later allocation and launch that does not name a device landed on that card.
"""
from __future__ import annotations

import pytest

from neurobrix.core.strategies.triton.base import TritonStrategy, _cuda_index


class _Fake:
    """An NBXTensor as this surface duck-types it."""

    def __init__(self, idx=0):
        self._device = "cuda"
        self._device_idx = idx
        self.moved_to_cpu = False

    def to_cpu(self):
        self.moved_to_cpu = True
        return self

    def to_cuda(self, idx=0):                     # must NOT be the path a peer move takes
        raise AssertionError("a peer move must go through the shared cross-device helper")


def test_an_unknown_device_is_refused_not_returned_untouched():
    for name in ("mps", "metal", "hip:0", "rocm", "gpu:1", ""):
        with pytest.raises(ValueError, match="cannot transfer"):
            _cuda_index(name)
    with pytest.raises(ValueError, match="cannot transfer"):
        TritonStrategy.transfer_tensor(None, _Fake(), "mps")


def test_cuda_names_map_to_their_card():
    assert _cuda_index("cuda") == 0
    assert _cuda_index("cuda:0") == 0
    assert _cuda_index("cuda:3") == 3


def test_a_peer_move_goes_through_the_shared_helper(monkeypatch):
    seen = {}
    import neurobrix.triton.device_transfer as dt
    monkeypatch.setattr(dt, "transfer_tensor",
                        lambda t, idx: seen.setdefault("call", (t, idx)) or t)
    t = _Fake(idx=0)
    TritonStrategy.transfer_tensor(None, t, "cuda:2")     # to_cuda would raise
    assert seen["call"][1] == 2


def test_cpu_and_zero3_are_named_cases():
    t = _Fake()
    TritonStrategy.transfer_tensor(None, t, "cpu")
    assert t.moved_to_cpu
    u = _Fake()
    assert TritonStrategy.transfer_tensor(None, u, "zero3:cuda:0") is u
    assert not u.moved_to_cpu


def test_a_non_tensor_is_returned_unchanged():
    assert TritonStrategy.transfer_tensor(None, 7, "cuda:1") == 7


def test_synchronize_puts_back_the_card_it_found(monkeypatch):
    import neurobrix.kernels.nbx_tensor as nt
    state = {"cur": 0, "synced": []}
    monkeypatch.setattr(nt.DeviceAllocator, "get_device", staticmethod(lambda: state["cur"]))
    monkeypatch.setattr(nt.DeviceAllocator, "set_device",
                        staticmethod(lambda i: state.__setitem__("cur", i)))
    monkeypatch.setattr(nt.DeviceAllocator, "sync_device",
                        staticmethod(lambda: state["synced"].append(state["cur"])))

    TritonStrategy.synchronize_device(None, "cuda:2")
    assert state["synced"] == [2], "waited on the wrong card"
    assert state["cur"] == 0, "left the current card on the one it synced"

    TritonStrategy.synchronize_device(None, "cuda:0")
    assert state["synced"] == [2, 0]
    assert state["cur"] == 0
