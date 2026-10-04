"""`NBXTensor.to_cuda` / `to_cuda_async` / `to_cpu` copy a view's OWN elements, whatever
its layout.

The copy is one memcpy of `nbytes` from `data_ptr()`. That is the view's bytes
only when its elements tile one dense span (contiguous, or a permutation of it —
a transpose). A narrow on an inner axis has gaps between its rows: the gate half
of a stacked expert slab W_in[e][:, 0:F] out of [H, 2F] — the per-expert views
the triton MoE promotes to the card when a weight is still on the host at the op
(`execute_moe_fused`, the zero3 slow path). Before 2026-10-04 only an EXPAND view
was materialised first; this one was copied as the first H*F contiguous elements
of the slab and kept its strides — silent garbage. `to_cpu` had the sibling defect:
it copied nbytes from data_ptr into a row-major host buffer, so a transpose crossed
as its storage order read row-major (values transposed in silence).

What this test would do if the code were wrong: with the old condition the copied
values are the slab's leading bytes, not the half — the element-by-element compare
fails. No card: the device allocator is replaced by host memory for the duration of
a subprocess, so the bytes the copy moved are read back exactly as they landed.
"""
from __future__ import annotations

import os
import subprocess
import sys

SCRIPT = r'''
import ctypes, os
import numpy as np
import neurobrix.kernels.nbx_tensor as NT
from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype

keep = []
def malloc(nbytes, dev_idx=None):
    b = ctypes.create_string_buffer(max(nbytes, 1)); keep.append(b); return ctypes.addressof(b)
def memcpy(dst, src, nbytes, kind=3):
    ctypes.memmove(dst, src, nbytes)
NT.DeviceAllocator.malloc_cuda = staticmethod(malloc)
NT.DeviceAllocator.memcpy = staticmethod(memcpy)
NT.DeviceAllocator.set_device = staticmethod(lambda *a, **k: None)
NT.DeviceAllocator.free_cuda = staticmethod(lambda *a, **k: None)
NT.DeviceAllocator.memcpy_async = staticmethod(lambda dst, src, nbytes, kind=3, stream=0: ctypes.memmove(dst, src, nbytes))
NT.DeviceAllocator.malloc_host_pinned = staticmethod(malloc)

E, H, F = 3, 5, 4
src = np.arange(E * H * 2 * F, dtype=np.float32).reshape(E, H, 2 * F)
slab = NBXTensor.empty_cpu((E, H, 2 * F), NBXDtype.float32)
ctypes.memmove(slab.data_ptr(), src.ctypes.data, src.nbytes)

def read(t):
    """The device copy's elements, read through its own strides."""
    out = np.empty(tuple(t.shape), dtype=np.float32)
    base = t.data_ptr()
    for idx in np.ndindex(*out.shape):
        off = sum(i * s for i, s in zip(idx, t._strides))
        out[idx] = ctypes.c_float.from_address(base + 4 * off).value
    return out

cases = {
    "up half, inner-axis narrow": (slab.select(0, 1).narrow(1, F, F), src[1][:, F:]),
    "gate half transposed (the reader's view)": (slab.select(0, 2).narrow(1, 0, F).t(), src[2][:, :F].T),
    "transpose of a dense expert (dense, kept as is)": (slab.select(0, 0).t(), src[0].T),
    "contiguous expert": (slab.select(0, 1), src[1]),
}
for name, (view, want) in cases.items():
    got = read(view.to_cuda(0))
    assert np.array_equal(got, want), (name, got, want)
    got = read(view.to_cuda_async(0))
    assert np.array_equal(got, want), ("async", name, got, want)
    # to_cpu: the host re-pack (pinned) of a host view, the D2H's sibling copy
    got = read(view.to_cpu(pinned=True))
    assert np.array_equal(got, want), ("to_cpu", name, got, want)
    print("OK", name)
assert slab.select(0, 0).t().spans_densely()
assert not slab.select(0, 1).narrow(1, F, F).spans_densely()
print("ALL OK")
import sys; sys.stdout.flush()
os._exit(0)          # the host buffers posing as device memory are not freed through the allocator
'''


def test_a_non_dense_view_crosses_as_its_own_elements():
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")})
    r = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-2500:])
    assert "ALL OK" in r.stdout, r.stdout[-1500:]
