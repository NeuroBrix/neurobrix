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


# ---------------------------------------------------------------------------------------------------------------------
# Gathered copies: on a discrete card, one memcpy per page and per contiguous run of a device's layers, the pages first
# gathered on the device by one kernel per K and V. The per-layer copies above cost ~30-40 us of call each (V100 PCIe,
# 125 TinyLlama pages = 5 500 copies of 8 KB: 150-230 ms), one contiguous copy of the same 43 MB 7-21 ms (measured
# 2026-11-09, campaigns 2026_11_09_release_063/kv_host_timing). Which path a machine takes is the engine's choice from its
# profile (discrete or unified memory); both are bit-exact.

import triton
import triton.language as tl

_BITS = {1: tl.int8, 2: tl.int16, 4: tl.int32}        # the element's bits, copied as an integer of its size
_NBX_BITS = {1: "int8", 2: "int16", 4: "int32"}


@triton.jit
def _kv_page_rows_kernel(table_ptr, pages_ptr, stage_ptr, page_elems, row_len, stage_page_elems, layer_elems, base,
                         TO_STAGE: tl.constexpr, ELEM: tl.constexpr, BLOCK: tl.constexpr):
    """One program per (page rank, layer slot, row): the row's `row_len` elements of pool[page] (the layer's pool read
    through its address in `table`) copied, bits unchanged, to or from the staging buffer laid out as host pages
    [page rank][layer slot][K, V] (`base`: the K or V offset inside a layer slot)."""
    r = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1).to(tl.int64)
    row = tl.program_id(2).to(tl.int64)
    page = tl.load(pages_ptr + r).to(tl.int64)
    pool = tl.cast(tl.load(table_ptr + s), tl.pointer_type(ELEM), bitcast=True)
    offs = tl.arange(0, BLOCK)
    m = offs < row_len
    pool_at = pool + page * page_elems + row * row_len + offs
    stage_at = stage_ptr + r * stage_page_elems + s * layer_elems + base + row * row_len + offs
    if TO_STAGE:
        tl.store(stage_at, tl.load(pool_at, mask=m), mask=m)
    else:
        tl.store(pool_at, tl.load(stage_at, mask=m), mask=m)


def _layer_runs(k_pools):
    """{device: [(first layer, last layer + 1), ...]}: each device's layers as contiguous runs (a pipeline split keeps
    a device's layers contiguous; any other placement still works, one run per layer)."""
    runs = {}
    for i, pool in enumerate(k_pools):
        d = int(pool._device_idx)
        cur = runs.setdefault(d, [])
        if cur and cur[-1][1] == i:
            cur[-1] = (cur[-1][0], i + 1)
        else:
            cur.append((i, i + 1))
    return runs


def _copies(host_pages, runs, num_layers):
    """The host copies, (first rank, end rank, first layer, end layer, layers before it in a rank's staging slice): one per rank
    and run; consecutive host pages merged into one copy when the card holds every layer (one run (0, num_layers))."""
    if runs == [(0, num_layers)]:
        r0 = 0
        for r in range(1, len(host_pages) + 1):
            if r == len(host_pages) or host_pages[r] != host_pages[r - 1] + 1:
                yield r0, r, 0, num_layers, 0
                r0 = r
        return
    for r in range(len(host_pages)):
        off = 0
        for a, b in runs:
            yield r, r + 1, a, b, off
            off += b - a


def _address_table(pools):
    """The device addresses a shader reads for `pools` (Metal: the pinned wraps' GPU addresses, under a pin the caller
    releases; elsewhere their data pointers), as an int64 table on the current device."""
    from nbxtensor.backend.runtime import gpu_backend
    from nbxtensor.tensor import NBXTensor
    pin = None
    if gpu_backend() == "metal":
        from nbxtensor.backend.triton_ext_driver import pinned_addresses, pinned_gpu_address
        pin = pinned_addresses(*pools)
        pin.__enter__()
        addrs = [pinned_gpu_address(p.data_ptr()) for p in pools]
    else:
        addrs = [p.data_ptr() for p in pools]
    return NBXTensor.from_numpy(np.array(addrs, dtype=np.int64)), pin


def _gathered(k_pools, v_pools, host: HostPages, pages, host_pages, to_host: bool, fence=None, owner=None) -> None:
    from nbxtensor.backend.runtime import set_device as launch_on     # Triton's driver on the tensor's device
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.tensor import NBXTensor as _T
    if len(pages) != len(host_pages):
        raise ValueError(f"kv_host_pages: {len(pages)} device pages, {len(host_pages)} host pages")
    elem = k_pools[0].element_size()
    if elem not in _BITS or any(p.element_size() != elem for p in (*k_pools, *v_pools)):
        raise ValueError(f"kv_host_pages: pools of element sizes {sorted({p.element_size() for p in (*k_pools, *v_pools)})}"
                         f" — one size of 1, 2 or 4 bytes is copied as its bits")
    for i in range(host.num_layers):
        for which, pools, nb in (("k", k_pools, host.k_page_bytes), ("v", v_pools, host.v_page_bytes)):
            if _page_bytes(pools[i]) != nb or not pools[i].is_contiguous():
                raise ValueError(f"kv_host_pages: layer {i}'s {which} pool ({_page_bytes(pools[i])} B a page, "
                                 f"contiguous {pools[i].is_contiguous()}) is not the host page's {nb} B contiguous page")
    n, kb, vb = len(pages), host.k_page_bytes, host.v_page_bytes
    if not to_host and fence is not None:
        fence.drain(owner)
    for dev, runs in _layer_runs(k_pools).items():
        DeviceAllocator.set_device(dev)
        layers = [i for a, b in runs for i in range(a, b)]
        if to_host:
            DeviceAllocator.device_synchronize(dev)          # queued kernels may still write the pages
        stage = _T.empty((n * len(layers) * (kb + vb) // elem,), dtype=getattr(NBXDtype, _NBX_BITS[elem]),
                         device=f"cuda:{dev}")
        page_ids = _T.from_numpy(np.asarray(list(pages), dtype=np.int64))
        tables, pins = [], []
        try:
            for which, pools, nb, base in (("k", k_pools, kb, 0), ("v", v_pools, vb, kb)):
                table, pin = _address_table([pools[i] for i in layers])
                tables.append(table)
                pins.append(pin)
                rows = int(np.prod(pools[layers[0]].shape[1:-1]))
                row_len = int(pools[layers[0]].shape[-1])
                grid = (n, len(layers), rows)
                args = (table, page_ids, stage, nb // elem, row_len, len(layers) * (kb + vb) // elem,
                        (kb + vb) // elem, base // elem)
                if to_host:
                    launch_on(stage)                         # the launch contract (backend/runtime.set_device)
                    _kv_page_rows_kernel[grid](*args, TO_STAGE=True, ELEM=_BITS[elem],
                                               BLOCK=triton.next_power_of_2(row_len))
            seg = len(layers) * (kb + vb)                    # this device's slice of one host page, per run
            if to_host:
                DeviceAllocator.device_synchronize(dev)
            for r0, r1, a, b, off in _copies(host_pages, runs, host.num_layers):
                # host pages r0..r1-1 are consecutive and this card holds every layer: the staging IS their host layout
                nbytes = (r1 - r0 - 1) * seg + (b - a) * (kb + vb)
                h_addr = host.address(host_pages[r0], a, "k")
                d_addr = stage.data_ptr() + r0 * seg + off * (kb + vb)
                if to_host:
                    DeviceAllocator.memcpy(h_addr, d_addr, nbytes, _D2H)
                else:
                    DeviceAllocator.memcpy(d_addr, h_addr, nbytes, _H2D)
            if not to_host:
                for (which, pools, nb, base), table in zip((("k", k_pools, kb, 0), ("v", v_pools, vb, kb)), tables):
                    rows = int(np.prod(pools[layers[0]].shape[1:-1]))
                    row_len = int(pools[layers[0]].shape[-1])
                    launch_on(stage)
                    _kv_page_rows_kernel[(n, len(layers), rows)](
                        table, page_ids, stage, nb // elem, row_len, len(layers) * (kb + vb) // elem,
                        (kb + vb) // elem, base // elem, TO_STAGE=False, ELEM=_BITS[elem],
                        BLOCK=triton.next_power_of_2(row_len))
                DeviceAllocator.device_synchronize(dev)
        finally:
            for pin in pins:
                if pin is not None:
                    pin.__exit__(None, None, None)
    if not to_host and fence is not None:
        fence.record()


def offload_gathered(k_pools: Sequence, v_pools: Sequence, pages: Sequence[int], host: HostPages,
                     host_pages: Sequence[int]) -> None:
    """`offload`, the pages gathered on the device first (one kernel per K and V) and copied to the host once per page
    and per contiguous run of a device's layers."""
    _gathered(k_pools, v_pools, host, pages, host_pages, to_host=True)


def restore_scattered(k_pools: Sequence, v_pools: Sequence, host: HostPages, host_pages: Sequence[int],
                      pages: Sequence[int], fence=None, owner=None) -> None:
    """`restore`, the host pages copied to the device once per page and per run of a device's layers, then scattered
    into the pools in place (one kernel per K and V), inside `fence` when the cache has one."""
    _gathered(k_pools, v_pools, host, pages, host_pages, to_host=False, fence=fence, owner=owner)

