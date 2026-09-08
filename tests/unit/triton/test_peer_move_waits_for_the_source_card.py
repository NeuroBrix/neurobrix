"""One cross-card copy path, and it always waits for the card that produced the data.

A device-to-device memcpy is issued after selecting the DESTINATION, so it is queued on the
destination's legacy stream — which is not ordered against the SOURCE's. A peer copy issued
right after a component's kernels therefore reads a buffer those kernels may still be writing:
wrong values, no error, nothing in a log. The wait used to live in one caller
(`triton.device_transfer.transfer_tensor`) while another (`NBXTensor.to_cuda`, which the
strategies' component hand-off called) had none.

It now lives in the primitive, which is the only place the copy is written, so no caller can
reach a copy that skips it. What this pins is the ORDER of the allocator calls.
"""
from __future__ import annotations

import pytest

from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXTensor


class _View:
    """An NBXTensor as `to_cuda` reads one — dense, empty, on card 0."""

    _device = "cuda"
    _device_idx = 0
    _shape = (4,)
    _strides = (1,)
    _numel = 4
    _nbytes = 0                      # so the copy itself is skipped; the ORDER is the subject
    _dtype = "float16"

    is_dense_window = NBXTensor.is_dense_window

    def data_ptr(self):
        return 0x1000


@pytest.fixture
def trace(monkeypatch):
    calls = []
    monkeypatch.setattr(DeviceAllocator, "get_device", staticmethod(lambda: 3))
    monkeypatch.setattr(DeviceAllocator, "set_device",
                        staticmethod(lambda i: calls.append(("set", i))))
    monkeypatch.setattr(DeviceAllocator, "sync_device",
                        staticmethod(lambda: calls.append(("sync",))))
    monkeypatch.setattr(DeviceAllocator, "ensure_peer_access",
                        staticmethod(lambda s, d: calls.append(("peer", s, d))))
    monkeypatch.setattr(DeviceAllocator, "malloc_cuda",
                        staticmethod(lambda n, dev=None: calls.append(("malloc", n)) or 0))
    monkeypatch.setattr(NBXTensor, "__init__",
                        lambda self, *a, **k: setattr(self, "_owns_data", False))
    return calls


def test_a_card_to_card_copy_waits_on_the_source_and_puts_the_card_back(trace):
    NBXTensor.to_cuda(_View(), 2)
    assert ("peer", 0, 2) in trace, "the direct card-to-card link was not enabled"
    sync_at = trace.index(("sync",))
    assert trace[sync_at - 1] == ("set", 0), "the wait was not on the SOURCE card"
    assert trace[sync_at + 1] == ("set", 3), "the card that was current was not put back"
    malloc_at = next(i for i, c in enumerate(trace) if c[0] == "malloc")
    assert sync_at < malloc_at, "the destination was allocated before the source was waited on"
    assert trace[malloc_at - 1] == ("set", 2)


def test_a_host_to_card_copy_waits_on_nothing(trace):
    host = _View()
    host._device = "cpu"
    NBXTensor.to_cuda(host, 1)
    assert not any(c[0] == "sync" for c in trace), "a host copy has no source card to wait on"
    assert not any(c[0] == "peer" for c in trace)


def test_a_tensor_already_on_the_card_is_returned_as_is(trace):
    v = _View()
    assert NBXTensor.to_cuda(v, 0) is v
    assert trace == []


def test_the_dense_window_rule_is_the_primitive_s_own():
    """A view whose strided span exceeds its numel cannot be moved by a flat memcpy with its
    strides carried over; the primitive decides that, not each caller."""
    v = _View()
    assert v.is_dense_window()
    v._shape, v._strides, v._numel = (2, 2), (16, 1), 4      # interior narrow: span 18 > 4
    assert not v.is_dense_window()
