"""The Triton engine's paged KV cache (state engine step 2, docs/design/state-engine.md §4.3.1).

The decoder's K / V live in fixed-size pages of one pool per layer, granted as each sequence grows, never at max_len:
per layer a K pool and a V pool `[num_pages, kv_heads, page_tokens, D]` (NBXTensor), one PagePool for the cache and one
PageTable per sequence (core/state/pages.py, shared with the compiled engine and Metal). The kernels are
kernels/ops/paged_decode_attn.py.

What a recorded replay needs is kept: every buffer a kernel reads has a fixed address for the cache's lifetime, and a
decode step takes no host input — its append reads the write position from the layer's DEVICE length and the page from
the block table, and a second launch advances the length (the B3 contract of the contiguous cache). The host keeps the
lengths as the control-flow authority (split sizing, page growth). A request reserves its pages at prefill for the prompt
and its decode budget; a sequence that outgrows them is granted more and the block table is rewritten in place.

Zero torch (R33): NBXTensor and Triton only.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from neurobrix.core.state.pages import PagePool, PageTable, block_table as _block_table
from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator


class PagedTritonKVCache:
    """Per-layer paged K / V for a batch of decoding sequences. `fence` (optional, `record()` / `drain(...)`): taken
    around the host's in-place rewrites of the fixed device buffers on a device whose queue may still read them
    (§4.3.1); None on CUDA, where the host write is stream-ordered after the kernels already launched."""

    paged = True

    def __init__(self, num_layers: int, num_kv_heads: int, k_head_dim: int, v_head_dim: int, max_cache_len: int,
                 dtype: NBXDtype, page_tokens: int, async_release: bool, window_ceiling: int = 0,
                 decode_budget: int = 0, initial_cache_len: int = 0, fence=None):
        if page_tokens <= 0 or max_cache_len <= 0:
            raise ValueError(f"PagedTritonKVCache: page_tokens {page_tokens}, max_cache_len {max_cache_len}")
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.k_head_dim = int(k_head_dim)
        self.v_head_dim = int(v_head_dim)
        self.max_cache_len = int(max_cache_len)
        self.dtype = dtype
        self.page_tokens = int(page_tokens)
        self.window_ceiling = int(window_ceiling or 0)
        self.decode_budget = int(decode_budget or 0)
        self.initial_cache_len = int(initial_cache_len or 0)
        self.fence = fence
        self.async_release = bool(async_release)   # the vendor file's memory.async_release, never assumed
        self.batch = 0
        self.device_idx: Optional[int] = None
        self.pool: Optional[PagePool] = None
        self.tables: List[PageTable] = []
        self.k_pools: List[Optional[NBXTensor]] = [None] * self.num_layers
        self.v_pools: List[Optional[NBXTensor]] = [None] * self.num_layers
        self.lens: List[Optional[NBXTensor]] = [None] * self.num_layers   # device int32 [B] per layer
        self.block_table_dev: Optional[NBXTensor] = None                    # device int32 [B, pages_per_seq]
        self.length = 0                                                     # host authority: positions held
        self._published_pages = -1

    # -- allocation ---------------------------------------------------------------------------------------------------
    @property
    def pages_per_sequence(self) -> int:
        return -(-self.max_cache_len // self.page_tokens)

    def _ensure(self, k) -> None:
        """The pools, sized at the first prefill: `batch` sequences of at most max_cache_len positions each (the plan's
        cache) — the pool's capacity, granted page by page as sequences grow."""
        batch = int(k.shape[0])
        if self.pool is not None:
            if batch != self.batch:
                raise RuntimeError(f"ZERO FALLBACK: the paged KV cache holds {self.batch} sequences, a request brought "
                                   f"{batch}")
            return
        self.batch, self.device_idx = batch, int(k._device_idx)
        num_pages = batch * self.pages_per_sequence
        self.pool = PagePool(num_pages, self.page_tokens, async_release=self.async_release)
        self.tables = [PageTable(self.pool) for _ in range(batch)]
        dev = f"cuda:{self.device_idx}"
        DeviceAllocator.set_device(self.device_idx)
        for i in range(self.num_layers):
            self.k_pools[i] = NBXTensor.zeros((num_pages, self.num_kv_heads, self.page_tokens, self.k_head_dim),
                                              dtype=self.dtype, device=dev)
            self.v_pools[i] = NBXTensor.zeros((num_pages, self.num_kv_heads, self.page_tokens, self.v_head_dim),
                                              dtype=self.dtype, device=dev)
            self.lens[i] = NBXTensor.from_numpy(np.zeros((batch,), dtype=np.int32))
        self.block_table_dev = NBXTensor.from_numpy(np.full((batch, self.pages_per_sequence), -1, dtype=np.int32))

    def _write_in_place(self, dst: NBXTensor, host: np.ndarray) -> None:
        """Rewrite a fixed device buffer's contents (its address never changes: replay tuples point at it)."""
        if self.fence is not None:
            self.fence.drain(self)
        DeviceAllocator.set_device(self.device_idx)
        src = NBXTensor.from_numpy(np.ascontiguousarray(host))
        DeviceAllocator.memcpy(dst.data_ptr(), src.data_ptr(), host.nbytes, 3)
        if self.fence is not None:
            self.fence.record()

    def _reserve(self, length: int) -> None:
        """Every sequence holds `length` positions' pages (never beyond the plan's cache); the block table republished
        when a page was granted."""
        if length > self.max_cache_len:
            raise RuntimeError(f"ZERO FALLBACK: the paged KV cache holds at most {self.max_cache_len} positions per "
                               f"sequence (the plan's cache), {length} asked")
        for t in self.tables:
            t.reserve(length)
        if len(self.tables[0].pages) != self._published_pages:
            rows, width = _block_table(self.tables)
            table = np.full((self.batch, self.pages_per_sequence), -1, dtype=np.int32)
            table[:, :width] = np.asarray(rows, dtype=np.int32).reshape(self.batch, width)
            self._write_in_place(self.block_table_dev, table)
            self._published_pages = len(self.tables[0].pages)

    # -- the steps ----------------------------------------------------------------------------------------------------
    def prefill(self, layer_idx: int, k, v) -> None:
        """Append a prompt's K / V `[B, H_kv, T, D]` (positions length .. length+T-1). Layer 0 starts the request's
        reservation: the prompt and its decode budget."""
        from neurobrix.kernels import wrappers as W
        self._ensure(k)
        t = int(k.shape[2])
        if layer_idx == 0:
            # the prompt (refused by name past the plan's cache) and its decode budget, capped at the plan's cache
            need = self.length + t
            self._reserve(need if need > self.max_cache_len else min(need + self.decode_budget, self.max_cache_len))
        start = self.length
        for b in range(self.batch):
            slots = np.asarray(self.tables[b].slots(range(start, start + t)), dtype=np.int32)
            dev_slots = NBXTensor.from_numpy(slots)
            W.paged_kv_append(k[b].permute(1, 0, 2), self.k_pools[layer_idx], dev_slots)
            W.paged_kv_append(v[b].permute(1, 0, 2), self.v_pools[layer_idx], dev_slots)
        self._write_in_place(self.lens[layer_idx], np.full((self.batch,), start + t, dtype=np.int32))
        if layer_idx == self.num_layers - 1:
            self.length = start + t

    def decode(self, layer_idx: int, q, k, v, scale: float, block_n: int, q_round=None):
        """One decode step of one layer: append its new K / V `[B, H_kv, 1, D]` at the device length, advance it, and
        attend `q [B, H, 1, D]` over the sequence's pages. Returns `[B, H, 1, D_v]`."""
        import triton
        from neurobrix.kernels import wrappers as W
        from neurobrix.kernels.nbx_tensor import _set_device
        from neurobrix.kernels.ops.paged_decode_attn import paged_kv_append_step_kernel, paged_lens_advance_kernel
        if int(k.shape[2]) != 1:
            raise RuntimeError(f"PagedTritonKVCache.decode: {int(k.shape[2])} new positions — a decode step brings one")
        new_len = self.length + 1
        if layer_idx == 0 and new_len > len(self.tables[0].pages) * self.page_tokens:
            self._reserve(new_len)                              # past the request's reservation: one more page
        k2 = k.reshape(self.batch, self.num_kv_heads, self.k_head_dim).contiguous()
        v2 = v.reshape(self.batch, self.num_kv_heads, self.v_head_dim).contiguous()
        kp, vp, lens = self.k_pools[layer_idx], self.v_pools[layer_idx], self.lens[layer_idx]
        _set_device(kp)
        paged_kv_append_step_kernel[(self.batch, self.num_kv_heads)](
            k2, v2, kp, vp, self.block_table_dev, lens, self.page_tokens,
            k2.stride(0), k2.stride(1), v2.stride(0), v2.stride(1),
            kp.stride(0), kp.stride(1), kp.stride(2),
            vp.stride(0), vp.stride(1), vp.stride(2),
            self.block_table_dev.stride(0),
            BLOCK_D=max(triton.next_power_of_2(max(self.k_head_dim, self.v_head_dim)), 16),
            D_K=self.k_head_dim, D_V=self.v_head_dim, num_warps=1)
        paged_lens_advance_kernel[(self.batch,)](lens, 1)
        out = W.paged_decode_attention(q, kp, vp, self.block_table_dev, lens, scale, new_len, block_n,
                                       q_round=q_round)
        if layer_idx == self.num_layers - 1:
            self.length = new_len
        return out

    # -- the interceptor's protocol (triton/kv_cache.py: the cache answers for its own state) -----------------------
    def get_cache_len(self) -> int:
        return self.length

    def _write_lens(self, value: int) -> None:
        arr = np.full((self.batch,), int(value), dtype=np.int32)
        for i in range(self.num_layers):
            self._write_in_place(self.lens[i], arr)

    def skip_positions(self, n: int) -> None:
        """A census shadow skips `n` positions: the host length and every layer's device length move."""
        self.length += int(n)
        if self.pool is not None:
            self._write_lens(self.length)

    def replay_key(self, bucket: int):
        """The decode step's replay-bucket contribution, or None when a recorded replay could not reproduce it: the next
        position must lie in a page already reserved (a replayed step runs no host code, so it grants no page). The key
        carries the split count (the read's grid) — the block table and lengths are read on the device, their
        addresses fixed for the cache's lifetime, so no generation is needed."""
        from neurobrix.kernels.ops.paged_decode_attn import PAGED_DECODE_SPLIT_POSITIONS, PAGED_DECODE_MAX_SPLITS
        if self.pool is None or not self.tables:
            return None
        nxt = self.length + 1
        if nxt > len(self.tables[0].pages) * self.page_tokens:
            return None
        split = max(1, min(-(-nxt // PAGED_DECODE_SPLIT_POSITIONS), PAGED_DECODE_MAX_SPLITS))
        return ("kv_decode_paged", split, self.batch, id(self))

    def replay_advance(self) -> None:
        """The replayed launches appended one position per layer and advanced every device length: mirror it."""
        self.length += 1

    def replay_restore(self) -> None:
        """Rewind every device length to the host authority before the verify pass's normal re-run."""
        if self.pool is not None:
            self._write_lens(self.length)

    def replay_position_diag(self):
        """(device length of layer 0's first sequence, host length), or None."""
        import ctypes as _c
        if self.pool is None:
            return None
        b = (_c.c_int32 * 1)()
        DeviceAllocator.memcpy(_c.addressof(b), self.lens[0].data_ptr(), 4, 2)
        return int(b[0]), self.length

    def clone_empty(self) -> "PagedTritonKVCache":
        """An empty cache of this geometry — a decode branch (a CFG negative context)."""
        return PagedTritonKVCache(self.num_layers, self.num_kv_heads, self.k_head_dim, self.v_head_dim,
                                  self.max_cache_len, self.dtype, self.page_tokens, self.async_release,
                                  self.window_ceiling, self.decode_budget, self.initial_cache_len, self.fence)

    # -- lifetime -----------------------------------------------------------------------------------------------------

    def clear(self) -> None:
        """The request ends: its pages back to the pool (at once on CUDA), the lengths to zero. Buffers keep their
        addresses."""
        for t in self.tables:
            t.free()
        self.length = 0
        self._published_pages = -1
        if self.pool is not None:
            self._write_lens(0)

    def held_bytes(self) -> int:
        """Bytes the granted pages hold across layers — what the cache costs now (never max_len unless granted)."""
        if self.pool is None:
            return 0
        granted = self.pool.num_pages - self.pool.free_pages
        per_page = self.num_kv_heads * self.page_tokens * (self.k_head_dim + self.v_head_dim) \
            * self.k_pools[0].element_size()
        return granted * per_page * self.num_layers
