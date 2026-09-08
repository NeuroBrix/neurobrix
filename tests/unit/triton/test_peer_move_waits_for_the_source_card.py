"""A move BETWEEN cards waits for the card that produced the data.

`NBXTensor.to_cuda` issues a device-to-device memcpy after selecting the TARGET card, so the
copy is queued on the target's legacy stream — which is not ordered against the SOURCE card's.
A peer copy issued right after a component's kernels therefore reads a buffer those kernels may
still be writing: wrong values, no error, nothing in a log.

The engine already had the answer: `triton.device_transfer.transfer_tensor` waits on the source,
enables the peer link, materialises a non-dense window and carries the strides. The op-by-op
paths have used it since the DeepSeek-Coder-V2-Lite and Qwen3-Omni faults; the strategies'
component hand-off went on calling `to_cuda` directly.
"""
from __future__ import annotations

from neurobrix.core.strategies.triton.base import TritonStrategy


class _Fake:
    def __init__(self, device="cuda", idx=0):
        self._device = device
        self._device_idx = idx
        self.to_cuda_called_with = None

    def to_cpu(self):
        return self

    def to_cuda(self, idx=0):
        self.to_cuda_called_with = idx
        return self


def test_a_cross_card_move_goes_through_the_shared_helper(monkeypatch):
    seen = {}
    import neurobrix.triton.device_transfer as dt
    monkeypatch.setattr(dt, "transfer_tensor",
                        lambda t, idx: seen.setdefault("peer", idx) or t)
    t = _Fake(idx=0)
    TritonStrategy.transfer_tensor(None, t, "cuda:2")
    assert seen.get("peer") == 2, "a peer move did not take the path that waits for the source"
    assert t.to_cuda_called_with is None, "it took `to_cuda`, which does not wait"


def test_a_same_card_move_and_a_host_move_stay_on_the_plain_path(monkeypatch):
    import neurobrix.triton.device_transfer as dt
    monkeypatch.setattr(dt, "transfer_tensor",
                        lambda t, idx: (_ for _ in ()).throw(AssertionError("not a peer move")))
    same = _Fake(idx=1)
    TritonStrategy.transfer_tensor(None, same, "cuda:1")
    assert same.to_cuda_called_with == 1

    from_host = _Fake(device="cpu", idx=0)
    TritonStrategy.transfer_tensor(None, from_host, "cuda:3")
    assert from_host.to_cuda_called_with == 3, "a host-to-card move is not a peer copy"


def test_every_accelerator_prefix_still_reaches_a_card(monkeypatch):
    """The Mac's rule, unchanged: hip, xpu, mps and the rest are GPUs Prism can name."""
    import neurobrix.triton.device_transfer as dt
    monkeypatch.setattr(dt, "transfer_tensor", lambda t, idx: t)
    for name, want in (("hip:1", 1), ("mps:0", 0), ("xpu:2", 2), ("cuda", 0)):
        t = _Fake(device="cpu")
        TritonStrategy.transfer_tensor(None, t, name)
        assert t.to_cuda_called_with == want, f"{name} did not reach a card"
