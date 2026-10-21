"""The compiled engine's paged KV cache (state engine step 2, docs/design/state-engine.md §4.3.1).

Mirror of triton/paged_kv_cache.py with the compiled engine's raw allocation: per layer a K pool and a V pool
`[num_pages, kv_heads, page_tokens, D]` (torch tensors on the device that layer's attention runs on — a model split
across cards keeps each layer's state on its card), one PagePool and one PageTable per sequence (core/state/pages.py).
The append writes the step's positions into their slots; the attention reads the sequence's pages GATHERED through the
table into the K / V it already takes (the design's compiled branch: one gather per layer per step, the attention code
unchanged). The pool starts at the plan's size and grows by pages toward the ceiling (the model's window when the
factory raised it): never allocated at max_len up front.

The interface the wrapper (kv_cache_wrapper.py) and both compiled flows call: update, get_seq_len, get_cache, clear.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from neurobrix.core.dtype.config import get_torch_dtype
from neurobrix.core.state.pages import PagePool, PageTable


class PagedKVCache:
    paged = True

    def __init__(self, config, page_tokens: int, async_release: bool):
        if page_tokens <= 0:
            raise ValueError(f"PagedKVCache: page_tokens {page_tokens}")
        self.config = config
        self.page_tokens = int(page_tokens)
        self.async_release = bool(async_release)
        self.batch = 0
        self.pool: Optional[PagePool] = None
        self.tables: List[PageTable] = []
        self.k_pools: Dict[int, torch.Tensor] = {}
        self.v_pools: Dict[int, torch.Tensor] = {}
        self.lens: Dict[int, int] = {}
        self._seq_len = 0

    # -- capacity --------------------------------------------------------------------------------------------------
    def _pages_for(self, positions: int) -> int:
        return -(-int(positions) // self.page_tokens)

    def _ensure(self, k: torch.Tensor) -> None:
        batch = int(k.shape[0])
        if self.pool is None:
            self.batch = batch
            start = self.config.initial_cache_len or self.config.max_cache_len
            self.pool = PagePool(batch * self._pages_for(start), self.page_tokens, self.async_release)
            self.tables = [PageTable(self.pool) for _ in range(batch)]
        elif batch != self.batch:
            if self._seq_len != 0:
                raise RuntimeError(f"ZERO FALLBACK: the paged KV cache holds {self.batch} sequences mid-request "
                                   f"({self._seq_len} positions), and a step brought {batch}")
            # A new request of another batch (`neurobrix serve`): the cache is empty, so it re-pages for the new batch
            # — pool, tables and per-layer pools rebuilt as at a first prefill (supervisor 2026-10-21 12:15)
            for t in self.tables:
                t.free()
            self.k_pools.clear(); self.v_pools.clear(); self.lens.clear()
            self.batch = batch
            start = self.config.initial_cache_len or self.config.max_cache_len
            self.pool = PagePool(batch * self._pages_for(start), self.page_tokens, self.async_release)
            self.tables = [PageTable(self.pool) for _ in range(batch)]

    def _layer_pools(self, layer_idx: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """The layer's pools on its device, extended to the pool's page count (pages added by `grow`)."""
        n = self.pool.num_pages
        dtype = get_torch_dtype(self.config.dtype)
        kp, vp = self.k_pools.get(layer_idx), self.v_pools.get(layer_idx)
        if kp is None:
            kp = torch.zeros((n, self.config.num_kv_heads, self.page_tokens, self.config.k_head_dim),
                             dtype=dtype, device=device)
            vp = torch.zeros((n, self.config.num_kv_heads, self.page_tokens, self.config.v_head_dim),
                             dtype=dtype, device=device)
        elif kp.shape[0] < n:
            extra = n - kp.shape[0]
            kp = torch.cat([kp, kp.new_zeros((extra,) + tuple(kp.shape[1:]))])
            vp = torch.cat([vp, vp.new_zeros((extra,) + tuple(vp.shape[1:]))])
        self.k_pools[layer_idx], self.v_pools[layer_idx] = kp, vp
        return kp, vp

    def _reserve(self, length: int) -> None:
        """Every sequence holds `length` positions; the pool grows by the pages it lacks, up to the ceiling."""
        if length > self.config.max_cache_len:
            authority = "the model's context window" if getattr(self.config, "ceiling_is_window", False) \
                else "the plan's cache"
            raise RuntimeError(f"ZERO FALLBACK: {length} positions exceed {authority} ({self.config.max_cache_len})")
        lacking = sum(max(0, self._pages_for(length) - len(t.pages)) for t in self.tables) - self.pool.free_pages
        if lacking > 0:
            self.pool.grow(lacking)
        for t in self.tables:
            t.reserve(length)

    # -- the interface the wrapper calls -----------------------------------------------------------------------------
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append k / v `[B, H_kv, T, D]` at the layer's length; return the layer's whole K / V `[B, H_kv, L, D]`
        gathered through the page tables."""
        if layer_idx < 0 or layer_idx >= self.config.num_layers:
            raise RuntimeError(f"ZERO FALLBACK: layer_idx {layer_idx} out of range [0, {self.config.num_layers})")
        self._ensure(k)
        start = self.lens.get(layer_idx, 0)
        t = int(k.shape[2])
        end = start + t
        if end > len(self.tables[0].pages) * self.page_tokens:
            self._reserve(end)
        kp, vp = self._layer_pools(layer_idx, k.device)
        tpp = self.page_tokens
        positions = torch.arange(start, end, device=k.device)
        ks, vs = [], []
        for b, table in enumerate(self.tables):
            pages = torch.tensor(table.pages, device=k.device, dtype=torch.long)
            kp[pages[positions // tpp], :, positions % tpp] = k[b].transpose(0, 1).to(kp.dtype)
            vp[pages[positions // tpp], :, positions % tpp] = v[b].transpose(0, 1).to(vp.dtype)
            n = self._pages_for(end)
            ks.append(kp[pages[:n]].permute(1, 0, 2, 3).reshape(kp.shape[1], n * tpp, kp.shape[3])[:, :end])
            vs.append(vp[pages[:n]].permute(1, 0, 2, 3).reshape(vp.shape[1], n * tpp, vp.shape[3])[:, :end])
        self.lens[layer_idx] = end
        if layer_idx == 0:
            self._seq_len = end
        return torch.stack(ks), torch.stack(vs)

    def get_seq_len(self) -> int:
        return self._seq_len

    def get_cache(self, layer_idx: int):
        if layer_idx not in self.k_pools or not self.lens.get(layer_idx):
            return None, None
        end, tpp = self.lens[layer_idx], self.page_tokens
        kp, vp = self.k_pools[layer_idx], self.v_pools[layer_idx]
        n = self._pages_for(end)
        ks, vs = [], []
        for table in self.tables:
            pages = torch.tensor(table.pages[:n], device=kp.device, dtype=torch.long)
            ks.append(kp[pages].permute(1, 0, 2, 3).reshape(kp.shape[1], n * tpp, kp.shape[3])[:, :end])
            vs.append(vp[pages].permute(1, 0, 2, 3).reshape(vp.shape[1], n * tpp, vp.shape[3])[:, :end])
        return torch.stack(ks), torch.stack(vs)

    def clear(self) -> None:
        for t in self.tables:
            t.free()
        self.lens.clear()
        self._seq_len = 0

    def get_layer_devices(self) -> Dict[int, torch.device]:
        return {i: p.device for i, p in self.k_pools.items()}

    def held_bytes(self) -> int:
        """Bytes the granted pages hold across layers."""
        if self.pool is None or not self.k_pools:
            return 0
        granted = self.pool.num_pages - self.pool.free_pages
        kp, vp = next(iter(self.k_pools.values())), next(iter(self.v_pools.values()))
        per_page = (kp[0].numel() * kp.element_size()) + (vp[0].numel() * vp.element_size())
        return granted * per_page * len(self.k_pools)
