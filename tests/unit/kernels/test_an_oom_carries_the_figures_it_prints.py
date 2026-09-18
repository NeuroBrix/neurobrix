"""The allocator's refusal carries its numbers, not only a sentence about them.

`DeviceAllocator.malloc_cuda` already computes the requested size, the live set,
the pool's cached bytes and the driver's free figure — it needs all four to write
its message. They were then discarded into an f-string, so a caller that wanted to
RESHAPE the work rather than give up had to parse them back out of English.

Those four numbers are the input the adaptive-memory controller needs to re-enter
the placement cascade at the op-level tiling rung with the free figure at the
moment of failure, instead of the estimate made before the run began
(`docs/reference/adaptive-memory-a-runtime-controller.md`, addition 3). A
controller built on a regex over this message would break the first time the
message is reworded, and the message is a diagnostic, not an interface.

Two things are asserted, and the second matters as much as the first: the fields
are present, AND they agree with the text. A carried number that drifts from the
printed one is worse than no carried number, because both look authoritative.

Injection that turns this red: drop the keyword arguments from the `raise
DeviceOOMError(...)` in `malloc_cuda` and keep the message.
"""

from __future__ import annotations

import re

import pytest

from neurobrix.kernels.nbx_tensor import DeviceAllocator, DeviceOOMError


def _cuda_total_bytes():
    """Through torch, not a bare `ctypes.CDLL("libcudart.so")`.

    This runs at MODULE scope, and a bare CDLL of the SONAME resolves
    `libcudart.so.12` to the system runtime; torch's `libc10_cuda.so` then binds
    to that copy instead of the one torch ships, and every later module in the
    session that imports torch fails to import. The sibling cell
    `test_a_finalizer_does_not_import_at_shutdown` did exactly that on
    2026-09-18 and cost eleven collection errors in this directory — each of
    which passed when run alone, which is what makes the shape hard to see.

    Same question, same answer, no second CUDA runtime in the process.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return 0


_TOTAL = _cuda_total_bytes()


@pytest.mark.skipif(_TOTAL == 0, reason="needs a CUDA device")
def test_the_refusal_carries_requested_live_cached_and_driver_free():
    # Four times the card's total: refused by the driver immediately, so this
    # allocates nothing and puts no pressure on anything else using the card.
    want = _TOTAL * 4
    with pytest.raises(DeviceOOMError) as e:
        DeviceAllocator.malloc_cuda(want)
    oom = e.value

    assert oom.requested == want
    assert oom.device_idx is not None
    for field in ("live", "pool_cached", "pool_blocks"):
        assert getattr(oom, field) is not None, f"{field} was not carried"
        assert getattr(oom, field) >= 0

    text = str(oom)
    assert f"for {want} bytes" in text

    # The carried figures must be the ones printed. The message rounds to whole
    # MB, so each is compared at that resolution.
    def _printed(label):
        m = re.search(rf"{label}=(\d+)MB", text)
        return int(m.group(1)) if m else None

    live, cached = oom.live, oom.pool_cached
    assert live is not None and cached is not None
    assert _printed("live_tracked") == round(live / 1024 / 1024)
    assert _printed("pool_cached") == round(cached / 1024 / 1024)
    free = oom.driver_free
    if free is not None:
        assert _printed("driver_free") == round(free / 1024 / 1024)
        assert oom.driver_total is not None and oom.driver_total > 0
        # The shortfall is what a reshape has to close, and for a request four
        # times the card it is most of the request.
        assert oom.shortfall == want - free > 0


@pytest.mark.skipif(_TOTAL == 0, reason="needs a CUDA device")
def test_an_unanswerable_driver_reads_as_none_and_never_as_zero():
    """`0 bytes free` is a legitimate reading and a controller would act on it.

    "The runtime would not tell me" must therefore be a different value, or the
    controller cannot distinguish a full card from a broken query — and it would
    refuse to reshape in exactly the case that needs reshaping most.

    The unanswerable path is FORCED here rather than waited for. On this rack the
    driver always answers, so a version of this cell that simply allocated and
    checked `driver_total > 0` would assert the branch that is trivially true and
    never once execute the branch it is named after — green whatever the code did.
    So `cudaMemGetInfo` is hidden from the runtime handle for the duration, which
    is exactly the condition the `except`/`hasattr` fallback in `malloc_cuda`
    exists for.

    Injection that turns it red: pass `driver_total=driver_total` instead of
    `driver_total or None` in `malloc_cuda`, so an unanswerable query arrives as 0.
    Measured in that state: `driver_total=0 driver_free=0 shortfall=70368744177664`
    — a controller reading it would believe the card had nothing free and say so.
    """
    import neurobrix.kernels.nbx_tensor as M

    real_runtime = M._gpu_runtime

    class _NoMemInfo:
        """The runtime handle, minus the one symbol under test."""

        def __init__(self, rt):
            self._rt = rt

        def __getattr__(self, name):
            if name == "cudaMemGetInfo":
                raise AttributeError(name)
            return getattr(self._rt, name)

    M._gpu_runtime = lambda *a, **k: _NoMemInfo(real_runtime())
    try:
        with pytest.raises(DeviceOOMError) as e:
            DeviceAllocator.malloc_cuda(_TOTAL * 4)
    finally:
        M._gpu_runtime = real_runtime

    oom = e.value
    assert oom.driver_total is None, (
        f"an unanswerable driver came back as {oom.driver_total!r}, which a "
        f"controller cannot tell from a real reading")
    assert oom.driver_free is None
    assert oom.shortfall is None, (
        "a shortfall was computed against a free figure that does not exist")
    # What the allocator DOES know is still carried.
    assert oom.requested == _TOTAL * 4 and oom.live is not None
