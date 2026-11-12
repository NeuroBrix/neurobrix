"""A host-resident stacked expert slab crosses to the device ONCE and WHOLE when the op's caller fetches it through
`ops.moe_fused.StackedSlabPromotion`, which then says the call's weights live for this call only; and a call whose
weights live for it alone never enters — nor is served from — the pointer-table cache
(`memory.pointer_tables.tables_for_call`): a reused address meeting a stale table reads another tensor's memory (zeros
on Metal, in silence).

What this test would do if the code were wrong: a promotion that copies per view, or not at all, or forgets to say
`per_call` — RED on the transfers or the flag; a resident slab promoted — RED; a per-call table cached, or a per-call
call served the cached table — RED on the cache. No card: host memory stands in for the device allocator, in a
subprocess (the fake addresses must never reach the real allocator's free).
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("triton")


_SCRIPT = r"""
import ctypes, os, sys
import numpy as np
import nbxtensor.memory.allocator as A
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.tensor import NBXTensor
calls, keep = [], []
def malloc(nbytes, dev_idx=None):
    b = ctypes.create_string_buffer(max(nbytes, 1)); keep.append(b); return ctypes.addressof(b)
def memcpy(dst, src, nbytes, kind=3):
    calls.append(nbytes); ctypes.memmove(dst, src, nbytes)
A.DeviceAllocator.malloc_cuda = staticmethod(malloc)
A.DeviceAllocator.memcpy = staticmethod(memcpy)
A.DeviceAllocator.set_device = staticmethod(lambda *a, **k: None)
A.DeviceAllocator.ensure_triton_device = staticmethod(lambda *a, **k: None)
A.DeviceAllocator.free_cuda = staticmethod(lambda *a, **k: None)
from nbxtensor.ops.moe_fused import StackedSlabPromotion
import nbxtensor.memory.pointer_tables as PT
E, H, F = 4, 6, 3

def host(shape):
    t = NBXTensor.empty_cpu(shape, NBXDtype.float16)
    a = np.arange(int(np.prod(shape)), dtype=np.float16).reshape(shape)
    ctypes.memmove(t.data_ptr(), a.ctypes.data, a.nbytes)
    return t

# 1. the promotion: a whole slab, one transfer, its bytes, and it says so
slab, out = host((E, H, 2 * F)), host((E, F, H))
promo = StackedSlabPromotion({"in": slab, "out": out}.get, 0)
assert not promo.per_call
d_in, d_out = promo("in"), promo("out")
assert promo.per_call, "a promoted slab must mark the call's weights per-call"
assert d_in._device == "cuda" and d_out._device == "cuda", "a slab stayed on the host"
assert calls == [slab.nbytes(), out.nbytes()], ("one whole transfer per slab", calls)
assert (d_in.shape, d_out.shape) == ((E, H, 2 * F), (E, F, H)), (d_in.shape, d_out.shape)
assert ctypes.string_at(d_in.data_ptr(), slab.nbytes()) == ctypes.string_at(slab.data_ptr(), slab.nbytes())
resident = StackedSlabPromotion({"in": d_in, "none": None}.get, 0)
assert resident("in") is d_in and resident("none") is None and not resident.per_call, "a resident slab is not per-call"
assert len(calls) == 2, ("a resident slab crossed again", calls)
print("PROMOTED OK")

# 2. the table rule: per-call weights never enter (nor read) the pointer-table cache
built = []
PT.build_ptr_tables = lambda *a: built.append(1) or object()
ws = [NBXTensor.empty((F, H), "float16", 0) for _ in range(E)]
fp = PT._ptr_cache_fingerprint(ws, ws, ws, E)
t1 = PT.tables_for_call(ws, ws, ws, E, True, 0)
assert PT._ptr_cache_get(fp) is None, "a per-call table entered the cache"
t2 = PT.tables_for_call(ws, ws, ws, E, False, 0)
assert PT._ptr_cache_get(fp) is t2 and len(built) == 2, "a resident table is cached"
assert PT.tables_for_call(ws, ws, ws, E, False, 0) is t2 and len(built) == 2, "a resident call is served the cache"
t3 = PT.tables_for_call(ws, ws, ws, E, True, 0)
assert t3 is not t2 and len(built) == 3, "a per-call call must not be served a cached table"
print("TABLES OK"); sys.stdout.flush(); os._exit(0)
"""


def test_a_promoted_slab_crosses_once_and_its_table_is_never_cached():
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")})
    r = subprocess.run([sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-2500:])
    for line in ("PROMOTED OK", "TABLES OK"):
        assert line in r.stdout, (line, r.stdout[-1500:])
