"""A GPU address a kernel captures is valid exactly as long as its wrap lives.

`_buffer_for` builds a NEW `metal_native.wrap` per launch, so an address a
kernel stores into a table (`tl.cast(ptr, tl.int64, bitcast=True)`) names a
buffer that is released before the next launch runs — the read comes back
ZEROS, nothing raised. That is the mixture-of-experts idiom, and it is why
`execute_moe_fused` refuses today.

`pinned_addresses` is the lifetime as CODE: inside the scope every launch
reuses one wrap per (address, element type), so the captured address keeps
naming the same storage.

The bare path is asserted at the MECHANISM (a fresh wrap object per launch),
not at the data. Measured 2026-09-20 while writing this file: with no scope the
table read came back 0/256 in one ordering and 256/256 in another — Metal can
hand a re-wrap of the same region the same GPU address it just released, so the
stale read sometimes accidentally works. A behaviour that is an accident cannot
be pinned by asserting either of its outcomes; what CAN be pinned is that the
wrap is transient without the scope and stable inside it, which is exactly the
difference the scope exists to make.
"""
from __future__ import annotations

import numpy as np
import pytest

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from neurobrix.kernels.nbx_tensor import NBXTensor  # noqa: E402
from neurobrix.kernels.launcher import launch  # noqa: E402


def _metal_or_skip():
    try:
        from neurobrix.kernels.nbx_tensor import _detect_gpu_backend
        if _detect_gpu_backend() != "metal":
            pytest.skip("the transient-wrap lifetime is this driver's; "
                        "other drivers bind their own way")
        from neurobrix.triton import triton_ext_driver as drv
        return drv
    except Exception:
        pytest.skip("no Metal backend the engine can resolve")


N, BLOCK = 256, 256


@triton.jit
def _capture(src_ptr, tab_ptr, i):
    tl.store(tab_ptr + i, tl.cast(src_ptr, tl.int64, bitcast=True))


@triton.jit
def _read_through(tab_ptr, out_ptr, n, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    m = off < n
    src = tl.cast(tl.load(tab_ptr), tl.pointer_type(tl.float32), bitcast=True)
    tl.store(out_ptr + off, tl.load(src + off, mask=m, other=0.0), mask=m)


def _capture_then_read(src):
    tab = NBXTensor.from_numpy(np.zeros(1, dtype=np.int64))
    launch(_capture, (1,), src, tab, 0)
    out = NBXTensor.from_numpy(np.zeros(N, dtype=np.float32))
    launch(_read_through, (1,), tab, out, N, BLOCK=BLOCK)
    return out.numpy()


def test_without_a_scope_every_launch_gets_a_fresh_wrap():
    """The failure's MECHANISM, asserted where it is deterministic.

    Two binds of the same pointer outside any scope must be two objects: the
    first launch's wrap is released before the second runs, which is why a
    captured address is undefined there. If this ever returns one object, a
    cache has appeared under the driver and `pinned_addresses` needs
    re-justifying against whatever now holds it."""
    drv = _metal_or_skip()
    src = NBXTensor.from_numpy(np.arange(1, N + 1, dtype=np.float32))
    a = drv._buffer_for(src.data_ptr(), "*fp32")
    b = drv._buffer_for(src.data_ptr(), "*fp32")
    assert a is not b, (
        "the driver returned the SAME wrap for two binds outside any scope")


def test_inside_a_scope_one_wrap_serves_every_bind():
    """The mechanism, inside: same pointer, same scope, ONE wrap object."""
    drv = _metal_or_skip()
    src = NBXTensor.from_numpy(np.arange(1, N + 1, dtype=np.float32))
    with drv.pinned_addresses(src):
        a = drv._buffer_for(src.data_ptr(), "*fp32")
        b = drv._buffer_for(src.data_ptr(), "*fp32")
        assert a is b, "two binds inside one scope must reuse the pinned wrap"
    assert not drv._PIN_COUNTS


def test_inside_the_scope_the_captured_address_reads_the_data():
    drv = _metal_or_skip()
    want = np.arange(1, N + 1, dtype=np.float32)
    src = NBXTensor.from_numpy(want)
    with drv.pinned_addresses(src):
        got = _capture_then_read(src)
    assert int((got != want).sum()) == 0, (
        f"inside pinned_addresses the table read "
        f"{int((got != want).sum())}/{N} elements wrong")


def test_the_pin_does_not_outlive_its_scope():
    """After exit the driver is back to transient wraps — nothing leaks."""
    drv = _metal_or_skip()
    want = np.arange(1, N + 1, dtype=np.float32)
    src = NBXTensor.from_numpy(want)
    with drv.pinned_addresses(src):
        pass
    assert not drv._PINNED_WRAPS and not drv._PIN_COUNTS, (
        f"the scope exited but left pins behind: "
        f"{list(drv._PINNED_WRAPS)} / {dict(drv._PIN_COUNTS)}")
    a = drv._buffer_for(src.data_ptr(), "*fp32")
    b = drv._buffer_for(src.data_ptr(), "*fp32")
    assert a is not b, (
        "after the scope exited the driver still serves one wrap for two "
        "binds — the pin leaked past its lifetime")


def test_nested_scopes_hold_until_the_last_exit():
    drv = _metal_or_skip()
    want = np.arange(1, N + 1, dtype=np.float32)
    src = NBXTensor.from_numpy(want)
    with drv.pinned_addresses(src):
        with drv.pinned_addresses(src):
            pass
        # the outer scope still holds the pin
        got = _capture_then_read(src)
    assert int((got != want).sum()) == 0, (
        "an inner scope's exit dropped a pin the outer scope still holds")
    assert not drv._PIN_COUNTS


def test_an_empty_scope_refuses():
    drv = _metal_or_skip()
    with pytest.raises(RuntimeError, match="nothing to pin"):
        drv.pinned_addresses()
