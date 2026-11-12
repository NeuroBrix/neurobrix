"""A GPU address a kernel captures stays valid for every launch inside a `pinned_addresses` scope.

A kernel stores the bits of a pointer it was handed (`tl.cast(ptr, tl.int64, bitcast=True)`) into a table and a later
kernel reads through it: the mixture-of-experts idiom. The address is the allocation's own buffer address plus the
offset (nbxtensor binds its own buffers: `backend/metal_dispatch.py`), valid as long as the allocation lives; the scope
holds the tensors, so the allocator cannot free and reissue their storage under a table.

Until 2026-11-12 a launch bound a FRESH wrap of the memory each time, so a captured address named a buffer released
before the next launch, and the scope's job was to keep one wrap alive; the tests of that wrap are retired with it.
"""
from __future__ import annotations

import numpy as np
import pytest

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from nbxtensor.tensor import NBXTensor  # noqa: E402
from nbxtensor.backend.launcher import launch  # noqa: E402


def _metal_or_skip():
    try:
        from nbxtensor.backend.runtime import gpu_backend
        if gpu_backend() != "metal":
            pytest.skip("the captured-address lifetime is this driver's; "
                        "other drivers bind their own way")
        from nbxtensor.backend import triton_ext_driver as drv
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
    """After exit nothing is held, and an address asked for outside a scope is refused by name."""
    drv = _metal_or_skip()
    want = np.arange(1, N + 1, dtype=np.float32)
    src = NBXTensor.from_numpy(want)
    with drv.pinned_addresses(src):
        pass
    assert not drv._PIN_COUNTS, f"the scope exited but left pins behind: {dict(drv._PIN_COUNTS)}"
    with pytest.raises(RuntimeError, match="not under any pinned_addresses scope"):
        drv.pinned_gpu_address(src.data_ptr())


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


def test_the_address_flags_of_a_compilation_reach_its_kernel_call():
    """triton-ext's compiler records whether a kernel exposes addresses (stores a pointer's bits) or reads
    through them, and its own launcher passes both flags to every call: Metal keeps resident only the buffers
    the encoder names. Our launch passed neither. They travel with the compilation's trailing object."""
    _metal_or_skip()
    from nbxtensor.backend.launcher import prepare
    src = NBXTensor.from_numpy(np.arange(1, N + 1, dtype=np.float32))
    tab = NBXTensor.from_numpy(np.zeros(1, dtype=np.int64))
    out = NBXTensor.from_numpy(np.zeros(N, dtype=np.float32))
    capture, _ = prepare(_capture, (src, tab, 0), {})
    read, _ = prepare(_read_through, (tab, out, N), {"BLOCK": BLOCK})
    assert capture.trailing is not None and capture.trailing.exposes_addresses, capture.trailing
    assert read.trailing is not None and read.trailing.reads_addresses, read.trailing
