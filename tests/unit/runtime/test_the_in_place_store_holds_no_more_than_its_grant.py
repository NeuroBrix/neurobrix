"""The in-place expert store (unified memory) never holds more of the experts' pages than the slots Prism granted it,
counting EVERY holder: the store's own bindings (`DeviceAllocator.bind_mapped`, one Metal buffer per projection) and the
whole-allocation wraps the cached pointer tables pin (`pointer_tables.build_ptr_tables` -> `pinned_addresses`,
`triton_ext_driver._RESIDENT_WRAPS`). A pin is a second Metal buffer over the same pages and keeps them resident as long
as the table lives in its 256-entry LRU, so a pin that outlives its binding, or more pins than bound experts, holds
pages the grant does not count.

Why: DeepSeek-Coder-V2-Lite triton on the M4 Pro (2026-11-08, RUN 81) wired 2.7 -> 16.2 GB in 46 s against a 614-slot
(10 131 MB) grant, while the driver's pinned-wrap ledger grew to 2 370 entries. This gate says whether the store and its
tables stay inside the grant at the op level, on real mapped pages, through the engine's own MoE call.

What this test does if the code is wrong: red when a pinned wrap survives the unbinding of its range, when the bound set
exceeds the slot count, or when the bytes held by bindings and pins together exceed n_slots experts. Seen red on main
caafe74c8 at the third decode call: "6 pinned wrap(s) over pages no binding holds any more (18 pins, 12 bindings, 0 cached
tables)" — the half-precision SIMT decode band (`execute_moe_fused` STEP 2c, 1d5da7cfd 2026-10-28) returned without
exiting its per-call table's pins, so every routed expert kept a wrap after the store unbound it.
"""
from __future__ import annotations

import ctypes
import mmap
import os

import numpy as np
import pytest

from neurobrix.core.runtime.expert_residency import ExpertResidency

E, TOP_K, H, F, LAYERS, N_SLOTS, CALLS = 16, 2, 128, 128, 3, 4, 48


def _metal():
    import nbxtensor.backend.runtime as rt
    return rt.gpu_backend() == "metal"


def _host_experts(tmp_path, layer, rng):
    """E (gate, up, down) fp16 [F, H] host views of ONE file mapping, packed back to back (no page alignment: a
    projection shares pages with its neighbours, as in a safetensors shard)."""
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.tensor import NBXTensor
    mats = [(rng.standard_normal((F, H)) * 0.1).astype(np.float16) for _ in range(3 * E)]
    path = tmp_path / f"shard{layer}.bin"
    path.write_bytes(b"".join(m.tobytes() for m in mats) + b"\0" * 64)
    with open(path, "rb") as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    base = ctypes.addressof(ctypes.c_char.from_buffer(m))
    nb = F * H * 2
    views = [NBXTensor.from_raw(base + i * nb, (F, H), NBXDtype.float16, "cpu", owns_data=False, base=m)
             for i in range(3 * E)]
    return m, views[0::3], views[1::3], views[2::3]


def _held(spans):
    """(bound ranges, pinned ranges) over this test's mappings `spans` [(start, end)], as {page start: bytes}, read from
    the driver's own tables. Only this test's pages: a pin another test left in the same process is not this store's."""
    from nbxtensor.backend.metal_device import runtime, _MappedView
    from nbxtensor.backend import triton_ext_driver as drv
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    rt = runtime()
    page = os.sysconf("SC_PAGESIZE")
    ours = lambda a: any(s // page * page <= a < e for s, e in spans)
    bound = {s: e[2] for s, e in rt._host_buffers.items() if isinstance(e[1], _MappedView) and ours(s)}
    pinned = {}
    for b in [b for b in drv._RESIDENT_WRAPS if ours(b)]:
        hit = DA.containing(b)
        pinned[b] = hit[1] if hit is not None else -1          # -1: a pin whose range nobody records any more
    return bound, pinned


def _union_bytes(ranges):
    spans = sorted((s, s + n) for s, n in ranges.items() if n > 0)
    total, cur_s, cur_e = 0, None, None
    for s, e in spans:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + ((cur_e - cur_s) if cur_e is not None else 0)


@pytest.mark.skipif(not _metal(), reason="the in-place store binds file mappings on unified memory only")
def test_bindings_and_pinned_tables_together_stay_inside_the_grant(tmp_path):
    from nbxtensor.tensor import NBXTensor
    from neurobrix.triton.expert_store import NBXExpertStore
    from neurobrix.triton.moe import execute_moe_fused
    from neurobrix.kernels.autotune_certify import _bind_hardware_profile
    import nbxtensor.memory.pointer_tables as pt
    _bind_hardware_profile()
    rng = np.random.default_rng(7)
    layers = [_host_experts(tmp_path, i, rng) for i in range(LAYERS)]
    spans = [(lay[1][0].data_ptr(), lay[3][-1].data_ptr() + F * H * 2) for lay in layers]
    page = os.sysconf("SC_PAGESIZE")
    proj = -(-(F * H * 2) // page) * page + page               # a projection's pages, at most one page of overlap
    grant = N_SLOTS * 3 * proj
    store = NBXExpertStore(ExpertResidency(N_SLOTS), 0, (layers[0][1][0], layers[0][2][0], layers[0][3][0]),
                           in_place=True)
    worst = (0, None)
    try:
        for call in range(CALLS):
            li = call % LAYERS
            _m, g, u, d = layers[li]
            h = (rng.standard_normal((1, H)) * 0.5).astype(np.float16)
            logits = rng.standard_normal((1, E)).astype(np.float32)
            execute_moe_fused(NBXTensor.from_numpy(logits), NBXTensor.from_numpy(h), g, u, d, top_k=TOP_K,
                              num_experts=E, norm_topk_prob=False, cache_key=f"L{li}",
                              expert_store=store, expert_layer=f"L{li}")
            bound, pinned = _held(spans)
            orphans = [b for b, n in pinned.items() if n < 0 or not any(s <= b < s + bn for s, bn in bound.items())]
            assert not orphans, (f"call {call}: {len(orphans)} pinned wrap(s) over pages no binding holds any more "
                                 f"({len(pinned)} pins, {len(bound)} bindings, {len(pt._ptr_cache)} cached tables)")
            assert len(store._bound) <= N_SLOTS, f"call {call}: {len(store._bound)} experts bound for {N_SLOTS} slots"
            held = _union_bytes({**bound, **{b: n for b, n in pinned.items() if n > 0}})
            if held > worst[0]:
                worst = (held, call)
            assert held <= grant, (f"call {call}: bindings + pins hold {held} bytes, the grant is {grant} "
                                   f"({N_SLOTS} slots x 3 projections)")
    finally:
        pt.release_pinned_tables()
        for key in list(store._bound):
            for ptr in store._bound.pop(key)[0]:
                from nbxtensor.memory.allocator import DeviceAllocator as DA
                DA.unbind_mapped(ptr)
    print(f"worst held {worst[0]} bytes at call {worst[1]}, grant {grant}")
