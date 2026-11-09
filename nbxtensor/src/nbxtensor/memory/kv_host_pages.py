"""The KV host tier's raw pages and copies, for the Triton engine's paged cache (memory/paged_kv.py) — the mechanism
half of the tier whose bookkeeping is memory/kv_host_tier.py (reste_apres_064 point 10).

A host page holds one page index of EVERY layer's K and V, contiguous: [layer][K page bytes, V page bytes]. Its memory is
page-locked (the device's host allocation) when the engine asks for it — a discrete card, whose copies then run as DMA —
and ordinary pageable host memory otherwise: on unified memory the device's own host allocation is a GPU buffer counted
in the GPU working set, which is the scarce figure there (the engine decides from the machine's profile; this module
takes `pinned` as given). A page's K (or V) of one layer is `pool[page]`, a contiguous block of the layer's pool
`[pages, kv_heads, page_tokens, D]`: one copy per layer and per K/V. Offload waits for the device first (the pages may
still be written by queued kernels); restore writes into the pools in place (their addresses never move: replay plans
point at them), inside the cache's fence when it has one. Zero torch (R33)."""
from __future__ import annotations

import ctypes
from typing import Optional, Sequence

import numpy as np

from nbxtensor.memory.allocator import DeviceAllocator

_H2D, _D2H = 1, 2


class HostPages:
    """`capacity` host pages of `num_layers` layers' K pages of `k_page_bytes` and V pages of `v_page_bytes`."""

    def __init__(self, capacity: int, num_layers: int, k_page_bytes: int, v_page_bytes: int, pinned: bool):
        if capacity <= 0 or num_layers <= 0 or k_page_bytes <= 0 or v_page_bytes <= 0:
            raise ValueError(f"HostPages: {capacity} pages of {num_layers} layers x ({k_page_bytes} + {v_page_bytes}) B")
        self.capacity, self.num_layers = int(capacity), int(num_layers)
        self.k_page_bytes, self.v_page_bytes = int(k_page_bytes), int(v_page_bytes)
        self.page_bytes = self.num_layers * (self.k_page_bytes + self.v_page_bytes)
        n = self.capacity * self.page_bytes
        self._ptr = DeviceAllocator.malloc_host_pinned(n) if pinned else 0
        self.buf = (np.ctypeslib.as_array((ctypes.c_uint8 * n).from_address(self._ptr)) if pinned
                    else np.empty(n, dtype=np.uint8))
        self.pinned = bool(pinned)

    def address(self, host_page: int, layer: int, which: str) -> int:
        if not 0 <= host_page < self.capacity:
            raise IndexError(f"HostPages: host page {host_page} outside 0..{self.capacity - 1}")
        off = host_page * self.page_bytes + layer * (self.k_page_bytes + self.v_page_bytes)
        return self.buf.ctypes.data + off + (self.k_page_bytes if which == "v" else 0)

    def close(self) -> None:
        if self._ptr:
            DeviceAllocator.free_host_pinned(self._ptr)
            self._ptr = 0
        self.buf = None


def _page_bytes(pool) -> int:
    return (pool.numel() // int(pool.shape[0])) * pool.element_size()


def offload(k_pools: Sequence, v_pools: Sequence, pages: Sequence[int], host: HostPages,
            host_pages: Sequence[int]) -> None:
    """Copy each device page of `pages` (every layer's K and V) into the host page of the same rank in `host_pages`."""
    if len(pages) != len(host_pages):
        raise ValueError(f"kv_host_pages.offload: {len(pages)} device pages, {len(host_pages)} host pages")
    for i in range(host.num_layers):
        for which, pools, nb in (("k", k_pools, host.k_page_bytes), ("v", v_pools, host.v_page_bytes)):
            if _page_bytes(pools[i]) != nb:
                raise ValueError(f"kv_host_pages: layer {i}'s {which} page is {_page_bytes(pools[i])} B, the host "
                                 f"page holds {nb}")
    devices = {int(p._device_idx) for p in (*k_pools, *v_pools)}
    for d in devices:                                    # queued kernels may still write the pages
        DeviceAllocator.device_synchronize(d)
    for page, hp in zip(pages, host_pages):
        for i in range(host.num_layers):
            DeviceAllocator.set_device(k_pools[i]._device_idx)
            DeviceAllocator.memcpy(host.address(hp, i, "k"), k_pools[i].data_ptr() + int(page) * host.k_page_bytes,
                                   host.k_page_bytes, _D2H)
            DeviceAllocator.memcpy(host.address(hp, i, "v"), v_pools[i].data_ptr() + int(page) * host.v_page_bytes,
                                   host.v_page_bytes, _D2H)


def restore(k_pools: Sequence, v_pools: Sequence, host: HostPages, host_pages: Sequence[int],
            pages: Sequence[int], fence=None, owner=None) -> None:
    """Copy each host page of `host_pages` back into the device page of the same rank in `pages`, in place; inside
    `fence` (drain before, record after) when the cache has one."""
    if len(pages) != len(host_pages):
        raise ValueError(f"kv_host_pages.restore: {len(host_pages)} host pages, {len(pages)} device pages")
    if fence is not None:
        fence.drain(owner)
    for hp, page in zip(host_pages, pages):
        for i in range(host.num_layers):
            DeviceAllocator.set_device(k_pools[i]._device_idx)
            DeviceAllocator.memcpy(k_pools[i].data_ptr() + int(page) * host.k_page_bytes, host.address(hp, i, "k"),
                                   host.k_page_bytes, _H2D)
            DeviceAllocator.memcpy(v_pools[i].data_ptr() + int(page) * host.v_page_bytes, host.address(hp, i, "v"),
                                   host.v_page_bytes, _H2D)
    if fence is not None:
        fence.record()
