"""The compiled engine's paged KV cache (state engine step 2, docs/design/state-engine.md §4.3.1).

Mirror of nbxtensor/memory/paged_kv.py with the compiled engine's raw allocation: per layer a K pool and a V pool
`[num_pages, kv_heads, page_tokens, D]` (torch tensors on the device that layer's attention runs on — a model split
across cards keeps each layer's state on its card), one PagePool and one PageTable per sequence (nbxtensor/memory/pages.py).
The append writes the step's positions into their slots; the attention reads the sequence's pages GATHERED through the
table into the K / V it already takes (the design's compiled branch: one gather per layer per step, the attention code
unchanged). The pool starts at the plan's size and grows by pages toward the ceiling (the model's window when the
factory raised it): never allocated at max_len up front.

The interface the wrapper (kv_cache_wrapper.py) and both compiled flows call: update, get_seq_len, get_cache, clear.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from neurobrix.core.dtype.config import get_torch_dtype
from nbxtensor.memory.pages import PagePool, PageTable


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
        self.spilled = 0        # positions spilled to host memory mid-decode (the Triton cache's protocol; the KV tier reads it)
        self.holder = None      # the serving session whose positions the cache holds (core/state/kv_session), or None

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

    def _reserve(self, length: int, real: Optional[int] = None) -> None:
        """Every sequence holds `length` positions; the pool grows by the pages it lacks, up to the ceiling. `real`:
        the request's own end when the last rows pad a planned chunk (the ceiling bounds the request, not the
        padding)."""
        if (length if real is None else real) > self.config.max_cache_len:
            authority = "the model's context window" if getattr(self.config, "ceiling_is_window", False) \
                else "the plan's cache"
            raise RuntimeError(f"ZERO FALLBACK: {length} positions exceed {authority} ({self.config.max_cache_len})")
        lacking = sum(max(0, self._pages_for(length) - len(t.pages)) for t in self.tables) - self.pool.free_pages
        if lacking > 0:
            self.pool.grow(lacking)
        for t in self.tables:
            t.reserve(length)

    # -- the interface the wrapper calls -----------------------------------------------------------------------------
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
               real: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append k / v `[B, H_kv, T, D]` at the layer's length; return the layer's whole K / V `[B, H_kv, L, D]`
        gathered through the page tables. `real`: the rows of a planned chunk that are the request's (the rest pad
        it; `truncate` drops them after the chunk)."""
        if layer_idx < 0 or layer_idx >= self.config.num_layers:
            raise RuntimeError(f"ZERO FALLBACK: layer_idx {layer_idx} out of range [0, {self.config.num_layers})")
        self._ensure(k)
        start = self.lens.get(layer_idx, 0)
        t = int(k.shape[2])
        end = start + t
        if end > len(self.tables[0].pages) * self.page_tokens:
            self._reserve(end, None if real is None else start + int(real))
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

    def truncate(self, length: int) -> None:
        """Keep the first `length` positions of every layer (a session's matched prefix, or a padded chunk's real end):
        the pages stay granted and the dropped slots are overwritten next."""
        length = int(length)
        if length < 0 or length > self._seq_len:
            raise RuntimeError(f"ZERO FALLBACK: truncating the KV cache to {length} positions; it holds {self._seq_len}")
        for layer in list(self.lens):
            self.lens[layer] = min(self.lens[layer], length)
        self._seq_len = length

    def whole_pages(self, layer_idx: int, positions: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Layer `layer_idx`'s K / V `[B, H_kv, positions, D]` over its first `positions` positions, a whole number of
        pages, gathered through the page tables (the positions past the layer's length are whatever the slots hold:
        the caller masks them)."""
        tpp = self.page_tokens
        if positions % tpp or positions > len(self.tables[0].pages) * tpp:
            raise RuntimeError(f"ZERO FALLBACK: gathering {positions} positions of layer {layer_idx} as whole pages of "
                               f"{tpp}; the sequence holds {len(self.tables[0].pages)} pages")
        kp, vp = self.k_pools[layer_idx], self.v_pools[layer_idx]
        n = positions // tpp
        ks, vs = [], []
        for table in self.tables:
            pages = torch.tensor(table.pages[:n], device=kp.device, dtype=torch.long)
            ks.append(kp[pages].permute(1, 0, 2, 3).reshape(kp.shape[1], positions, kp.shape[3]))
            vs.append(vp[pages].permute(1, 0, 2, 3).reshape(vp.shape[1], positions, vp.shape[3]))
        return torch.stack(ks), torch.stack(vs)

    # -- the host tier's protocol (nbxtensor.memory.kv_host_tier / kv_host_pages; one sequence: a serving session) ----
    def tier_layout(self):
        """(layers, K page bytes, V page bytes) of one page index, once every layer's pools exist."""
        if len(self.k_pools) != self.config.num_layers:
            raise RuntimeError("ZERO FALLBACK: the KV host tier read the cache's layout before every layer prefilled")
        kp, vp = self.k_pools[0], self.v_pools[0]
        return self.config.num_layers, kp[0].numel() * kp.element_size(), vp[0].numel() * vp.element_size()

    def tier_length(self) -> int:
        return self._seq_len

    def _host_rows(self, host, host_pages) -> np.ndarray:
        return np.stack([host.page_view(hp) for hp in host_pages])

    @torch.inference_mode()        # the pools are the compiled flow's inference tensors
    def tier_offload(self, pages, host, host_pages) -> None:
        """Each layer's K and V of `pages` gathered by one index per pool and copied to the host once per pool (a page
        at a time cost a call each: 5 500 copies for 125 TinyLlama pages)."""
        rows = np.empty((len(pages), host.page_bytes), dtype=np.uint8)
        for layer in range(self.config.num_layers):
            off = layer * (host.k_page_bytes + host.v_page_bytes)
            for pool, nb in ((self.k_pools[layer], host.k_page_bytes), (self.v_pools[layer], host.v_page_bytes)):
                idx = torch.tensor(list(pages), device=pool.device, dtype=torch.long)
                rows[:, off:off + nb] = pool[idx].reshape(len(pages), -1).view(torch.uint8).cpu().numpy()
                off += nb
        for i, hp in enumerate(host_pages):
            host.page_view(hp)[:] = rows[i]

    @torch.inference_mode()
    def tier_restore(self, host, host_pages, pages) -> None:
        rows = torch.from_numpy(self._host_rows(host, host_pages))
        for layer in range(self.config.num_layers):
            off = layer * (host.k_page_bytes + host.v_page_bytes)
            for pool, nb in ((self.k_pools[layer], host.k_page_bytes), (self.v_pools[layer], host.v_page_bytes)):
                idx = torch.tensor(list(pages), device=pool.device, dtype=torch.long)
                pool[idx] = rows[:, off:off + nb].contiguous().view(pool.dtype).reshape(
                    (len(pages),) + tuple(pool.shape[1:])).to(pool.device)
                off += nb

    @torch.inference_mode()
    def tier_resume(self, length: int) -> None:
        """The sequence holds `length` positions restored from the host: their pages granted (every layer's pool
        extended to them), the lengths moved."""
        self._reserve(int(length))
        for layer in range(self.config.num_layers):
            self._layer_pools(layer, self.k_pools[layer].device)
            self.lens[layer] = int(length)
        self._seq_len = int(length)

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
