"""The paged KV cache on the device (the engine's state engine step 2, docs/design/state-engine.md §4.3.1).

The decoder's K / V live in fixed-size pages of one pool per layer, granted as each sequence grows, never at max_len:
per layer a K pool and a V pool `[num_pages, kv_heads, page_tokens, D]` (NBXTensor), one PagePool for the cache and one
PageTable per sequence (memory/pages.py, read by the engine's compiled cache too). The kernels are
kernels/ops/paged_decode_attn.py.

What a recorded replay needs is kept: every buffer a kernel reads has a fixed address for the cache's lifetime, and a
decode step takes no host input — its append reads the write position from the layer's DEVICE length and the page from
the block table, and a second launch advances the length (the B3 contract of the contiguous cache). The host keeps the
lengths as the control-flow authority (split sizing, page growth). A request reserves its pages at prefill for the prompt
and its decode budget; a sequence that outgrows them is granted more and the block table is rewritten in place.

Each layer's pools and length live on the card that runs the layer (the device of the K it first appends — a plan that
spreads the decoder over several cards, pipeline_parallel, runs its layers there), and every such card holds its own copy
of the block table, all rewritten together. One device for the whole cache made a layer on another card read pages on
the first one: Ming-Lite-Omni-1.5 triton on four V100-16GB, an illegal memory access at its first decode (2026-10-27).

Zero torch (R33): NBXTensor and Triton only.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from nbxtensor.memory.pages import PagePool, PageTable, block_table as _block_table
from nbxtensor.tensor import NBXTensor
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.allocator import DeviceAllocator


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
        self.layer_device: List[Optional[int]] = [None] * self.num_layers   # the card each layer runs on
        self.block_tables: Dict[int, NBXTensor] = {}                         # device -> int32 [B, pages_per_seq]
        self._table_host: Optional[np.ndarray] = None                       # the table every copy holds
        self.pool: Optional[PagePool] = None
        self.tables: List[PageTable] = []
        self.k_pools: List[Optional[NBXTensor]] = [None] * self.num_layers
        self.v_pools: List[Optional[NBXTensor]] = [None] * self.num_layers
        self.lens: List[Optional[NBXTensor]] = [None] * self.num_layers   # device int32 [B] per layer
        self.length = 0                                                     # host authority: positions held
        self._published_pages = -1
        self.capacity = 0                                                   # positions per sequence the pools hold
        self._uid = next(PagedTritonKVCache._uids)
        self.generation = 0                                                 # moves when the pools are reallocated
        self.regrowths = 0
        self.holder = None      # the serving session whose positions the cache holds (memory.kv_session), or None
        # called (uid, the generation that ended) when the pools are reallocated: their addresses moved, and whatever a
        # caller recorded against them (an engine's replay plans) is retired there; None: nothing is recorded
        self.on_reallocate = None

    # -- allocation ---------------------------------------------------------------------------------------------------
    import itertools as _itertools
    _uids = _itertools.count(1)

    @property
    def pages_per_sequence(self) -> int:
        return -(-self.capacity // self.page_tokens)

    def _ensure(self, layer_idx: int, k) -> None:
        """The page pool, at the first prefill: `batch` sequences of `capacity` positions — the plan's initial size
        (serve plans) or its cache — granted page by page as sequences grow; a request needing more grows them
        (`_grow`). Then the layer's own buffers on its card (`_ensure_layer`)."""
        batch = int(k.shape[0])
        if self.pool is not None:
            if batch == self.batch:
                self._ensure_layer(layer_idx, k)
                return
            if self.length != 0:
                raise RuntimeError(f"ZERO FALLBACK: the paged KV cache holds {self.batch} sequences mid-request "
                                   f"({self.length} positions), and a step brought {batch}")
            # A new request of another batch (a serving process: requests of any batch): the cache holds
            # no position, so it re-pages for the new batch — pools, tables, lengths and block table rebuilt exactly as
            # at a first prefill. Addresses move: the generation moves once and every replay plan recorded under the
            # previous one is retired, as `_grow` does (supervisor 2026-10-21 12:15: never refuse what can run).
            self._repage()
        self.batch = batch
        self.capacity = self.initial_cache_len or self.max_cache_len
        num_pages = batch * self.pages_per_sequence
        self.pool = PagePool(num_pages, self.page_tokens, async_release=self.async_release)
        self.tables = [PageTable(self.pool) for _ in range(batch)]
        self._table_host = np.full((batch, self.pages_per_sequence), -1, dtype=np.int32)
        self._ensure_layer(layer_idx, k)

    def _ensure_layer(self, layer_idx: int, k) -> None:
        """Layer `layer_idx`'s pools and length on the card of `k` — its card for the cache's lifetime — and that
        card's copy of the block table. A layer whose K arrives on another card than its pages is refused by name."""
        dev = int(k._device_idx)
        have = self.layer_device[layer_idx]
        if have is not None:
            if have != dev:
                raise RuntimeError(f"ZERO FALLBACK: the paged KV cache holds layer {layer_idx}'s pages on cuda:{have} "
                                   f"and its K arrives on cuda:{dev}")
            return
        DeviceAllocator.set_device(dev)
        self.layer_device[layer_idx] = dev
        self.k_pools[layer_idx], self.v_pools[layer_idx] = self._new_pools(self.pool.num_pages, dev)
        self.lens[layer_idx] = NBXTensor.from_numpy(np.full((self.batch,), self.length, dtype=np.int32))
        if dev not in self.block_tables:
            self.block_tables[dev] = NBXTensor.from_numpy(self._table_host)

    def _repage(self) -> None:
        """Drop the pools of the previous batch (the cache is empty) so `_ensure` allocates them for the new one."""
        if self.fence is not None:
            self.fence.drain(self)
        for tbl in self.tables:
            tbl.free()
        self.pool, self.tables = None, []
        self.k_pools = [None] * self.num_layers
        self.v_pools = [None] * self.num_layers
        self.lens = [None] * self.num_layers
        self.layer_device = [None] * self.num_layers
        self.block_tables, self._table_host = {}, None
        self._published_pages = -1
        old_gen, uid = self.generation, self._uid
        self.generation += 1
        if self.on_reallocate is not None:
            self.on_reallocate(uid, old_gen)

    def _new_pools(self, num_pages: int, device: int):
        dev = f"cuda:{device}"
        return (NBXTensor.zeros((num_pages, self.num_kv_heads, self.page_tokens, self.k_head_dim), dtype=self.dtype,
                                device=dev),
                NBXTensor.zeros((num_pages, self.num_kv_heads, self.page_tokens, self.v_head_dim), dtype=self.dtype,
                                device=dev))

    def _grow(self, positions: int) -> None:
        """Every sequence may hold `positions`: the pools reallocated at that size with the pages held copied in place
        (page ids unchanged), the block table widened. Addresses change, so the generation moves once and every
        replay plan recorded under the previous one is retired (their tuples bake the old addresses)."""
        old_pages = self.pool.num_pages
        self.capacity = int(positions)
        new_pages = self.batch * self.pages_per_sequence
        self.pool.grow(new_pages - old_pages)
        for i in range(self.num_layers):
            dev = self.layer_device[i]
            if dev is None:
                continue                    # its buffers are made at its first prefill, at the grown size
            DeviceAllocator.set_device(dev)
            kp, vp = self._new_pools(new_pages, dev)
            DeviceAllocator.memcpy(kp.data_ptr(), self.k_pools[i].data_ptr(),
                                   self.k_pools[i].numel() * self.k_pools[i].element_size(), 3)
            DeviceAllocator.memcpy(vp.data_ptr(), self.v_pools[i].data_ptr(),
                                   self.v_pools[i].numel() * self.v_pools[i].element_size(), 3)
            self.k_pools[i], self.v_pools[i] = kp, vp
        self._table_host = np.full((self.batch, self.pages_per_sequence), -1, dtype=np.int32)
        for dev in list(self.block_tables):
            DeviceAllocator.set_device(dev)
            self.block_tables[dev] = NBXTensor.from_numpy(self._table_host)
        self._published_pages = -1
        old_gen, uid = self.generation, self._uid
        self.generation += 1
        self.regrowths += 1
        if self.on_reallocate is not None:
            self.on_reallocate(uid, old_gen)

    def _write_in_place(self, dst: NBXTensor, host: np.ndarray) -> None:
        """Rewrite a fixed device buffer's contents (its address never changes: replay tuples point at it)."""
        if self.fence is not None:
            self.fence.drain(self)
        DeviceAllocator.set_device(dst._device_idx)
        src = NBXTensor.from_numpy(np.ascontiguousarray(host))
        DeviceAllocator.memcpy(dst.data_ptr(), src.data_ptr(), host.nbytes, 3)
        if self.fence is not None:
            self.fence.record()

    def _reserve(self, length: int) -> None:
        """Every sequence holds `length` positions' pages, the pools grown first when they cannot; the block table
        republished when a page was granted."""
        if length > self.capacity:
            self._grow(length)
        for t in self.tables:
            t.reserve(length)
        if len(self.tables[0].pages) != self._published_pages:
            rows, width = _block_table(self.tables)
            table = np.full((self.batch, self.pages_per_sequence), -1, dtype=np.int32)
            table[:, :width] = np.asarray(rows, dtype=np.int32).reshape(self.batch, width)
            self._table_host = table
            for bt in self.block_tables.values():       # every card's copy, together
                self._write_in_place(bt, table)
            self._published_pages = len(self.tables[0].pages)

    def capacity_for(self, prompt_end: int) -> int:
        """Positions per sequence the pools hold once a request whose prompt ends at `prompt_end` is prefilled: the
        current capacity (first: the plan's initial size, else its cache) or the request's need when larger."""
        base = self.capacity or self.initial_cache_len or self.max_cache_len
        return max(base, self._request_length(prompt_end))

    def _request_length(self, prompt_end: int) -> int:
        """Positions a request reserves at its prefill — the contiguous cache's prompt-aware rule (S1
        TINYLLAMA-KVCAP, 2026-09-01): the prompt and its decode budget, at least the plan's size; bounded by the
        model's window, and a prompt reaching the window is the model's own limit, refused by name."""
        if self.window_ceiling and prompt_end >= self.window_ceiling:
            raise RuntimeError(f"Prompt length {prompt_end} reaches the model's context window "
                               f"({self.window_ceiling} positions) — this is the model's own limit, not an engine budget.")
        need = prompt_end + self.decode_budget if self.decode_budget else prompt_end + 1
        if self.window_ceiling:
            need = min(need, self.window_ceiling)
        return need

    # -- the steps ----------------------------------------------------------------------------------------------------
    def prefill(self, layer_idx: int, k, v, real: Optional[int] = None) -> None:
        """Append a prompt's K / V `[B, H_kv, T, D]` (positions length .. length+T-1). Layer 0 starts the request's
        reservation: the prompt and its decode budget. `real`: the chunk's real rows when its last rows pad a planned
        chunk (the request's length is the real one; the padding rows still get their slots, overwritten later)."""
        from nbxtensor.ops import wrappers as W
        from nbxtensor.ops import attention_wrappers as _nbx_attention_wrappers  # the family the moved names live in (M2 split)
        self._ensure(layer_idx, k)
        t = int(k.shape[2])
        if layer_idx == 0:
            real_t = t if real is None else int(real)
            self._reserve(max(self._request_length(self.length + real_t), self.length + t))
        start = self.length
        DeviceAllocator.set_device(self.layer_device[layer_idx])     # the slots go to the layer's card
        for b in range(self.batch):
            slots = np.asarray(self.tables[b].slots(range(start, start + t)), dtype=np.int32)
            dev_slots = NBXTensor.from_numpy(slots)
            _nbx_attention_wrappers.paged_kv_append(k[b].permute(1, 0, 2), self.k_pools[layer_idx], dev_slots)
            _nbx_attention_wrappers.paged_kv_append(v[b].permute(1, 0, 2), self.v_pools[layer_idx], dev_slots)
        self._write_in_place(self.lens[layer_idx], np.full((self.batch,), start + t, dtype=np.int32))
        if layer_idx == self.num_layers - 1:
            self.length = start + t

    def decode(self, layer_idx: int, q, k, v, scale: float, q_round=None):
        """One decode step of one layer: append its new K / V `[B, H_kv, 1, D]` at the device length, advance it, and
        attend `q [B, H, 1, D]` over the sequence's pages. Returns `[B, H, 1, D_v]`."""
        import triton
        from nbxtensor.ops import wrappers as W
        from nbxtensor.ops import attention_wrappers as _nbx_attention_wrappers  # the family the moved names live in (M2 split)
        from nbxtensor.backend.runtime import set_device
        from nbxtensor.ops.paged_decode_attn import paged_kv_append_step_kernel, paged_lens_advance_kernel
        if int(k.shape[2]) != 1:
            raise RuntimeError(f"PagedTritonKVCache.decode: {int(k.shape[2])} new positions — a decode step brings one")
        new_len = self.length + 1
        if layer_idx == 0 and new_len > len(self.tables[0].pages) * self.page_tokens:
            self._reserve(new_len)                              # past the request's reservation: one more page
        k2 = k.reshape(self.batch, self.num_kv_heads, self.k_head_dim).contiguous()
        v2 = v.reshape(self.batch, self.num_kv_heads, self.v_head_dim).contiguous()
        kp, vp, lens = self.k_pools[layer_idx], self.v_pools[layer_idx], self.lens[layer_idx]
        bt = self.block_tables[self.layer_device[layer_idx]]
        set_device(kp)
        paged_kv_append_step_kernel[(self.batch, self.num_kv_heads)](
            k2, v2, kp, vp, bt, lens, self.page_tokens,
            k2.stride(0), k2.stride(1), v2.stride(0), v2.stride(1),
            kp.stride(0), kp.stride(1), kp.stride(2),
            vp.stride(0), vp.stride(1), vp.stride(2),
            bt.stride(0),
            BLOCK_D=max(triton.next_power_of_2(max(self.k_head_dim, self.v_head_dim)), 16),
            D_K=self.k_head_dim, D_V=self.v_head_dim, num_warps=1)
        paged_lens_advance_kernel[(self.batch,)](lens, 1)
        out = _nbx_attention_wrappers.paged_decode_attention(q, kp, vp, bt, lens, scale, new_len, q_round=q_round)
        if layer_idx == self.num_layers - 1:
            self.length = new_len
        return out

    # -- a session's cache between turns (memory/kv_session.py) --------------------------------------------------
    def truncate(self, length: int) -> None:
        """Keep the first `length` positions (a session's matched prefix, or a padded chunk's real end): the host and
        every layer's device length move back; the pages stay granted and the dropped slots are overwritten next."""
        length = int(length)
        if length < 0 or length > self.length:
            raise RuntimeError(f"ZERO FALLBACK: truncating the KV cache to {length} positions; it holds {self.length}")
        if length == self.length:
            return
        self.length = length
        if self.pool is not None:
            self._write_lens(length)

    def gather_pages(self, layer_idx: int, which: str, positions: int, heads: int, dtype) -> NBXTensor:
        """Layer `layer_idx`'s K (`which` "k") or V ("v") of every sequence over its first `positions` positions, a
        whole number of pages, contiguous `[B, heads, positions, D]` in `dtype` (`paged_kv_gather`). The pages must be
        granted (a prefill reserved them)."""
        from nbxtensor.ops import attention_wrappers as _aw
        if positions > len(self.tables[0].pages) * self.page_tokens:
            raise RuntimeError(f"ZERO FALLBACK: gathering {positions} positions of layer {layer_idx}; the sequence "
                               f"holds {len(self.tables[0].pages)} pages of {self.page_tokens}")
        pool = (self.k_pools if which == "k" else self.v_pools)[layer_idx]
        DeviceAllocator.set_device(self.layer_device[layer_idx])
        return _aw.paged_kv_gather(pool, self.block_tables[self.layer_device[layer_idx]], positions, heads, dtype)

    # -- the interceptor's protocol (triton/kv_cache.py: the cache answers for its own state) -----------------------
    def get_cache_len(self) -> int:
        return self.length

    def _write_lens(self, value: int) -> None:
        arr = np.full((self.batch,), int(value), dtype=np.int32)
        for i in range(self.num_layers):
            if self.lens[i] is not None:
                self._write_in_place(self.lens[i], arr)

    def skip_positions(self, n: int) -> None:
        """A census shadow skips `n` positions: the host length and every layer's device length move."""
        self.length += int(n)
        if self.pool is not None:
            self._write_lens(self.length)

    def replay_key(self, bucket: int):
        """The decode step's replay-bucket contribution, or None when a recorded replay could not reproduce it: the next
        position must lie in a page already reserved (a replayed step runs no host code, so it grants no page). The key
        carries the split count (the read's grid) and the pools' identity (uid, generation): a growth reallocates them
        and retires every plan recorded against the old addresses."""
        from nbxtensor.ops.paged_decode_attn import paged_decode_splits
        if self.pool is None or not self.tables:
            return None
        nxt = self.length + 1
        if nxt > len(self.tables[0].pages) * self.page_tokens:
            return None
        return ("kv_decode_paged", paged_decode_splits(nxt), self.batch, self._uid, self.generation)

    def replay_advance(self) -> None:
        """The replayed launches appended one position per layer and advanced every device length: mirror it."""
        self.length += 1

    def replay_restore(self) -> None:
        """Rewind every device length to the host authority before the verify pass's normal re-run."""
        if self.pool is not None:
            self._write_lens(self.length)

    def replay_lens_snapshot(self):
        """Every layer's device lengths as they are now (a host copy): a tail-replay verify (`replay.maybe_run`) rewinds
        exactly what its replay advanced — `replay_restore` writes the host length to EVERY layer, which also rewound
        the head's layers that ran directly and had already advanced (their next step then overwrote a position)."""
        import ctypes as _c
        if self.pool is None:
            return None
        snap = []
        for i in range(self.num_layers):
            if self.lens[i] is None:
                snap.append(None)
                continue
            b = (_c.c_int32 * self.batch)()
            DeviceAllocator.set_device(self.layer_device[i])
            DeviceAllocator.memcpy(_c.addressof(b), self.lens[i].data_ptr(), 4 * self.batch, 2)
            snap.append(np.frombuffer(bytes(b), dtype=np.int32).copy())
        return snap

    def replay_lens_restore(self, snap) -> None:
        if snap is None:
            return
        for i, arr in enumerate(snap):
            if arr is not None:
                self._write_in_place(self.lens[i], arr)

    def replay_position_diag(self):
        """(device length of layer 0's first sequence, host length), or None."""
        import ctypes as _c
        if self.pool is None:
            return None
        b = (_c.c_int32 * 1)()
        DeviceAllocator.set_device(self.layer_device[0])
        DeviceAllocator.memcpy(_c.addressof(b), self.lens[0].data_ptr(), 4, 2)
        return int(b[0]), self.length

    def clone_empty(self) -> "PagedTritonKVCache":
        """An empty cache of this geometry — a decode branch (a CFG negative context)."""
        twin = PagedTritonKVCache(self.num_layers, self.num_kv_heads, self.k_head_dim, self.v_head_dim,
                                  self.max_cache_len, self.dtype, self.page_tokens, self.async_release,
                                  self.window_ceiling, self.decode_budget, self.initial_cache_len, self.fence)
        twin.on_reallocate = self.on_reallocate
        return twin

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


def chunk_mask(device_idx: int, start: int, rows: int, keys: int, memo: Dict) -> NBXTensor:
    """A prefill chunk's additive attention mask [rows, keys] (fp32): 0 where key j <= start + i (causal, bottom-right
    at the cached length), -inf elsewhere — the values the causal bias adds — which also masks the whole pages' keys
    past the true end. One per device in `memo` (the caller's, per chunk): every layer reads it."""
    m = memo.get(device_idx)
    if m is None:
        host = np.where(np.arange(keys)[None, :] <= (start + np.arange(rows))[:, None], 0.0, -np.inf)
        DeviceAllocator.set_device(device_idx)
        m = NBXTensor.from_numpy(host.astype(np.float32))
        memo[device_idx] = m
    return m


def require_causal_mask(attn_mask, rows: int, memo: Dict, where: str) -> None:
    """A graph's own prefill mask may be replaced by the chunk's only when it IS the causal pattern over the chunk's
    rows: a padding, window or bias mask has no reading over the cached keys, refused by name. Read once per mask
    buffer in `memo` (a host copy of rows x rows)."""
    if attn_mask is None:
        return
    key = ("graph_mask", attn_mask.data_ptr())
    if key in memo:
        return
    import ctypes
    m = attn_mask.contiguous()
    if m._dtype != NBXDtype.bool_:
        m = m.to(NBXDtype.float32)
    buf = (ctypes.c_char * m._nbytes)()
    DeviceAllocator.set_device(m._device_idx)
    DeviceAllocator.sync_device()
    DeviceAllocator.memcpy(ctypes.addressof(buf), m.data_ptr(), m._nbytes, 2)
    host = np.frombuffer(bytes(buf), dtype=(np.bool_ if m._dtype == NBXDtype.bool_ else np.float32))
    host = host.reshape(tuple(int(x) for x in m.shape))
    tri = np.tril(np.ones((rows, rows), dtype=bool))
    ok = host.shape[-2:] == (rows, rows)
    if ok:
        flat = host.reshape(-1, rows, rows)
        ok = bool((flat == tri).all()) if host.dtype == np.bool_ else bool(
            ((flat == 0) == tri).all() and (np.isneginf(flat) | (flat <= np.finfo(np.float16).min))[:, ~tri].all())
    if not ok:
        raise RuntimeError(f"ZERO FALLBACK: a prefill chunk's attention ({where}) carries a graph mask "
                           f"{tuple(m.shape)} that is not the causal pattern over its {rows} rows; it has no reading "
                           f"over the cached keys")
    memo[key] = True


def chunk_attention(cache: "PagedTritonKVCache", layer_idx: int, q, start: int, heads: int, scale, dropout_p,
                    memo: Dict):
    """A prefill chunk of q's rows at cached position `start` (its K/V already appended) attending over the cache's
    keys gathered as WHOLE PAGES: every key read back from the pages (the cache's dtype, widened exactly to the
    query's), the key length rounded up to whole pages, the explicit 0/-inf mask (`chunk_mask`) — then the existing
    attention, unchanged. `heads`: the query heads when the graph handed expanded K/V, else the KV heads."""
    from nbxtensor.ops.attention_wrappers import scaled_dot_product_attention_wrapper
    rows = int(q.shape[2])
    keys = -(-(int(start) + rows) // cache.page_tokens) * cache.page_tokens
    kg = cache.gather_pages(layer_idx, "k", keys, heads, q._dtype)
    vg = cache.gather_pages(layer_idx, "v", keys, heads, q._dtype)
    mask = chunk_mask(int(q._device_idx), int(start), rows, keys, memo)
    return scaled_dot_product_attention_wrapper(q, kg, vg, attn_mask=mask, dropout_p=dropout_p, is_causal=False,
                                                scale=scale, k_pre_transposed=False)
