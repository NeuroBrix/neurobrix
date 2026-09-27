"""A kernel argument that is a strided VIEW is screened on exactly its elements — never seated
unscreened, never restored from a stale copy of the span from its pointer.

The Mac's engine fact (its report, 2026-09-26; its gate RED at bd39b288): `_writable_buffers`
answered None to a non-contiguous argument and `_screen_configs` seated the FASTEST AMONG n
CANDIDATES THAT NOTHING VERIFIED. The span from a view's pointer covers other tensors' bytes (the
eight-test corruption of the earlier skip). Now a view carries its layout; its covering extent is
copied in ONE transfer, its elements gathered on the host in logical order, and on restore the
saved elements are scattered into a FRESH read of the extent so the bytes between them go back as
they are. Over the screening budget a view is refused by name. Seen RED on main 2739c0ee.

A first design (uncommitted) copied one byte range per contiguous run — one memcpy per ELEMENT for
a transposed operand — and sat 65 minutes inside `_snapshot_ranges` in the unit suite (py-spy,
2026-09-26 09:23 UTC). The last cell pins the transfer count so that cannot come back.
Pure Python, duck-typed arguments, no device: a bytearray plays the device memory.
"""
from __future__ import annotations

import ctypes
import time
from pathlib import Path

import numpy as np
import pytest

from neurobrix.kernels import launcher as L

REPO = Path(__file__).resolve().parents[3]


class _Dtype:
    name = "float16"


class _View:
    """A duck-typed tensor: data_ptr, _nbytes, is_contiguous, shape, stride (elements), element_size."""

    def __init__(self, addr, shape, strides, itemsize=2, contiguous=False):
        self._addr, self.shape, self._strides, self._itemsize, self._contig = addr, shape, strides, itemsize, contiguous
        n = 1
        for s in shape:
            n *= s
        self._nbytes = n * itemsize
        self.dtype = _Dtype()

    def data_ptr(self):
        return self._addr

    def is_contiguous(self):
        return self._contig

    def stride(self, dim=None):
        return self._strides if dim is None else self._strides[dim]

    def element_size(self):
        return self._itemsize


@pytest.fixture
def device(monkeypatch):
    """A fake device: `mem` is the allocation at address `base`; every transfer is counted."""
    class Dev:
        base = 1 << 20
        mem = bytearray()
        transfers = 0

    def memcpy(dst, src, nbytes, kind=3):
        Dev.transfers += 1
        if kind == 2:
            ctypes.memmove(dst, (ctypes.c_char * nbytes).from_buffer(Dev.mem, src - Dev.base), nbytes)
        elif kind == 1:
            Dev.mem[dst - Dev.base:dst - Dev.base + nbytes] = ctypes.string_at(src, nbytes)
        else:
            raise AssertionError(kind)

    from neurobrix.kernels import nbx_tensor as T
    monkeypatch.setattr(T.DeviceAllocator, "memcpy", staticmethod(memcpy))
    return Dev


def test_the_extent_of_a_row_slice_a_transpose_and_an_empty_view():
    assert L._view_extent((4, 3), (16, 2), 2) == (0, 3 * 16 + 2 * 2 + 2)
    assert L._view_extent((3, 4), (2, 16), 2) == (0, 2 * 2 + 3 * 16 + 2)
    assert L._view_extent((4,), (-2,), 2) == (-6, 8)
    assert L._view_extent((0, 5), (10, 2), 2) == (0, 0)


def test_a_strided_view_is_a_screened_buffer_charged_its_extent():
    (b,) = L._writable_buffers([_View(1000, (4, 3), (8, 1))])
    assert b.address == 1000 and b.dtype == "float16"
    assert b.view == (0, (4, 3), (16, 2), 2) and b.nbytes == 54
    (c,) = L._writable_buffers([_View(2000, (4, 8), (8, 1), contiguous=True)])
    assert c.view is None and c.nbytes == 64


def test_snapshot_is_the_logical_tensor_and_restore_leaves_the_gaps_as_they_are(device):
    device.mem = bytearray(range(64))                  # a [4, 8] fp16 allocation, 64 bytes
    view = _View(device.base, (4, 3), (8, 1))          # the first three columns of every row
    buffers = L._writable_buffers([view])
    (before,) = L._snapshot(buffers)
    assert before == bytes(device.mem[0:6] + device.mem[16:22] + device.mem[32:38] + device.mem[48:54])
    for i in range(64):                                # a candidate scribbles over the allocation
        device.mem[i] = 0xEE
    device.mem[10] = 0x42                              # ...and something else, in a gap, changes too
    L._restore(buffers, [before])
    for r in range(4):
        assert bytes(device.mem[r * 16:r * 16 + 6]) == bytes(range(r * 16, r * 16 + 6))   # elements back
    assert device.mem[10] == 0x42                                                      # the gap as it IS
    assert bytes(device.mem[6:10]) == b"\xee" * 4 and bytes(device.mem[54:64]) == b"\xee" * 10


def test_a_transposed_operand_is_gathered_in_logical_order(device):
    a = np.arange(12, dtype=np.float16).reshape(3, 4)  # stored row-major; the view is a.T
    device.mem = bytearray(a.tobytes())
    view = _View(device.base, (4, 3), (1, 4))          # a.T: shape (4, 3), strides (1, 4)
    (shot,) = L._snapshot(L._writable_buffers([view]))
    assert np.frombuffer(shot, dtype=np.float16).reshape(4, 3).tolist() == a.T.tolist()


def test_a_large_transposed_operand_costs_one_transfer_per_snapshot_and_two_per_restore(device):
    n = 2048                                           # a [2048, 2048] fp16 operand, transposed: 4.2 M elements
    device.mem = bytearray(np.arange(n * n, dtype=np.uint16).tobytes())
    view = _View(device.base, (n, n), (1, n))
    buffers = L._writable_buffers([view])
    device.transfers = 0
    t0 = time.perf_counter()
    shots = L._snapshot(buffers)
    snap_transfers = device.transfers
    L._restore(buffers, shots)
    elapsed = time.perf_counter() - t0
    assert snap_transfers == 1 and device.transfers == 3, device.transfers
    assert elapsed < 10.0, f"snapshot + restore of one {n}x{n} view took {elapsed:.1f} s on the host"


def test_the_screen_no_longer_seats_a_strided_view_unscreened():
    src = (REPO / "src/neurobrix/kernels/launcher.py").read_text()
    body = src.split("def _screen_configs(", 1)[1]
    assert "a strided view among the arguments, which the screen cannot snapshot" not in src
    assert "buffers is None" not in body
    assert "strided views the row-windowed screen cannot" in body
