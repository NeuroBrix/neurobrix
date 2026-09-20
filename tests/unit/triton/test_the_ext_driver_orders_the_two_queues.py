"""Two command queues, one device, shared buffers — nothing orders them for free.

NeuroBrix enqueues its device-to-device copies as blits on ITS queue and returns
to the host immediately. `metal_device._blit` states the premise that makes this
safe: "a blit on the same queue as the kernels is ordered by the GPU and waits
for nothing." triton-ext breaks it — its kernels run on its OWN queue — and Metal
orders command buffers only WITHIN a queue; across queues an explicit event is
required (Apple: a fence cannot synchronize untracked resources accessed from
separate queues).

Measured on swin2SR-classical-sr-x2-64 (M4 Pro, triton-ext, 2026-09-17), same
conv, same single sweep:

    no drain   input nan=63181 inf=356 absmax=3.39e+38  (63537 scattered runs)
    drained    input nan=0     inf=0   absmax=1.492
    sequential input nan=0     inf=0   absmax=1.492

and end to end, from a CLEARED autotune cache, both runs exit 0:

    no drain   std   9.2009505442844  unique 115  washed out, judged by eye
    drained    std 103.25420290707218 unique 256  correct, judged by eye

The ORDER is the whole content of the fix: draining after the dispatch is what
the driver already did, and it is not enough. So the test asserts the sequence,
not merely that a drain exists.

No GPU: the dispatch and both queues are stubbed.
"""
from __future__ import annotations

import pytest

from neurobrix.triton import triton_ext_driver as D


def test_neurobrix_queue_is_drained_BEFORE_the_foreign_dispatch(monkeypatch):
    events: list[str] = []

    # the backend module itself is only consulted for its scalar layout; stubbing
    # it keeps this a pure ordering test that runs without triton-ext installed
    monkeypatch.setattr(D, "_ext", lambda: type("X", (), {
        "_compute_scalar_layout": staticmethod(lambda tys: (4, [0])),
        "_SETBYTES_LIMIT": 4096})())
    monkeypatch.setattr(D, "_nbx_queue_drain", lambda: events.append("drain-nbx"))
    monkeypatch.setattr(D, "_native", lambda: type(
        "N", (), {"synchronize": staticmethod(lambda: events.append("sync-ext"))})())
    monkeypatch.setattr(D, "_buffer_for", lambda addr, ty: f"buf@{addr:#x}")

    def _fn(*args, **kwargs):
        events.append("dispatch")

    D.TritonExtDriver.instance().launch(
        _fn, grid=(2, 1, 1), block=(32, 1, 1), shared=0, stream=0,
        params=[("ptr", 0x1000), ("i32", 7)], names=["x_ptr", "n"],
        types=["*fp32", "i32"])

    assert events == ["drain-nbx", "dispatch", "sync-ext"], events
    assert events.index("drain-nbx") < events.index("dispatch"), (
        "NeuroBrix's queue must be drained BEFORE the foreign queue reads its "
        "buffers; draining only afterwards is the defect this test pins")


def test_the_drain_waits_on_neurobrix_own_queue():
    """It must be NeuroBrix's queue, not the backend's — the backend's own
    synchronize was already there and did not prevent the corruption."""
    import inspect
    src = inspect.getsource(D._nbx_queue_drain)
    assert "metal_device" in src and "sync" in src, src
