"""The positions a decoding sequence spilled from the card's paged cache to host memory (point 10, the mid-decode
spill; Dell + Mac 2026-11-10): the card keeps the newest positions, the oldest whole pages move here when the card may
hold no more, and each decode step's attention over them is computed here (ops/host_decode_partial.py), joined to the
card's as the reduce's last split.

Per layer, K and V as float32 in the attention's own layout [H_kv, positions, D], written ONCE per spilled page: a
per-step conversion and transpose cost 7x the math (measured 2026-11-10, 131 072 positions, M4 Pro). The price is twice
the card's half-precision bytes per position; `budget_bytes` (the serving layer's, from the profile and the user's cap:
an int, or a callable giving the bytes the spill may hold now — the share of one host budget the KV tier's parked pages
leave) bounds it, and a spill past it is refused by name. Grows by doubling up to the budget, never at max_len.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from nbxtensor.ops.host_decode_partial import host_decode_partial


class HostSpill:
    def __init__(self, num_layers: int, kv_heads: int, k_dim: int, v_dim: int, budget_bytes):
        self.num_layers, self.kv_heads, self.k_dim, self.v_dim = int(num_layers), int(kv_heads), int(k_dim), int(v_dim)
        self._budget = budget_bytes if callable(budget_bytes) else (lambda b=int(budget_bytes): b)
        self.k: List[Optional[np.ndarray]] = [None] * self.num_layers
        self.v: List[Optional[np.ndarray]] = [None] * self.num_layers
        self.held = [0] * self.num_layers                    # positions each layer holds

    def position_bytes(self) -> int:
        """Host bytes one spilled position costs across every layer (float32 K and V)."""
        return 4 * self.kv_heads * (self.k_dim + self.v_dim) * self.num_layers

    @property
    def budget_bytes(self) -> int:
        return int(self._budget())

    def reach(self) -> int:
        """Positions the budget can hold now."""
        return self.budget_bytes // self.position_bytes()

    @property
    def length(self) -> int:
        return self.held[0]

    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.k + self.v if a is not None)

    def append(self, layer: int, k_pages: np.ndarray, v_pages: np.ndarray) -> None:
        """Layer `layer`'s next spilled pages, in position order: [pages, H_kv, page_tokens, D] as the card pool holds
        them (any float dtype) — converted and laid out once here."""
        n = int(k_pages.shape[0]) * int(k_pages.shape[2])
        end = self.held[layer] + n
        if end > self.reach():
            raise RuntimeError(f"ZERO FALLBACK: spilling to {end} positions in host memory; the host budget "
                               f"({self.budget_bytes / 2 ** 20:.0f} MB, {self.position_bytes()} bytes a position across "
                               f"{self.num_layers} layers) holds {self.reach()} — the request is past the cache's reach")
        cap = 0 if self.k[layer] is None else self.k[layer].shape[1]
        if end > cap:
            new_cap = min(max(end, 2 * cap), self.reach())
            for arrs, d in ((self.k, self.k_dim), (self.v, self.v_dim)):
                grown = np.zeros((self.kv_heads, new_cap, d), np.float32)
                if arrs[layer] is not None:
                    grown[:, :self.held[layer]] = arrs[layer][:, :self.held[layer]]
                arrs[layer] = grown
        for arrs, pages in ((self.k, k_pages), (self.v, v_pages)):
            # [pages, H_kv, P, D] -> [H_kv, pages * P, D]
            flat = np.asarray(pages).transpose(1, 0, 2, 3).reshape(self.kv_heads, n, pages.shape[3])
            arrs[layer][:, self.held[layer]:end] = flat
        self.held[layer] = end

    def partial(self, layer: int, q: np.ndarray, groups: int, scale: float):
        """The attention partial of query rows q [H, D] (float32, rounded as the card reads them) over this layer's
        spilled positions (acc, m, l)."""
        if self.k[layer] is None:
            return host_decode_partial(q, np.zeros((self.kv_heads, 0, self.k_dim), np.float32),
                                       np.zeros((self.kv_heads, 0, self.v_dim), np.float32), 0, groups, scale)
        # the scores take, at once, at most what the budget leaves beside the spilled positions themselves
        return host_decode_partial(q, self.k[layer], self.v[layer], self.held[layer], groups, scale,
                                   scores_bytes=max(0, self.budget_bytes - self.nbytes()))

    def blocks(self, layer: int, positions: int):
        """This layer's spilled positions in order, as (k, v) float32 views [H_kv, n, D] of at most `positions`
        positions each — what a prefill chunk hands back to the card block by block (memory/paged_kv.py: many query
        rows over every spilled position is the card's work; `partial` is the decode step's one row)."""
        if int(positions) <= 0:
            raise ValueError(f"HostSpill.blocks: blocks of {positions} positions")
        for a in range(0, self.held[layer], int(positions)):
            b = min(a + int(positions), self.held[layer])
            yield self.k[layer][:, a:b], self.v[layer][:, a:b]

    def truncate(self, length: int) -> None:
        """Keep the first `length` spilled positions (a session's matched prefix)."""
        if length > self.length:
            raise RuntimeError(f"ZERO FALLBACK: truncating the host spill to {length} positions; it holds {self.length}")
        self.held = [min(h, int(length)) for h in self.held]

    def clear(self) -> None:
        self.k = [None] * self.num_layers
        self.v = [None] * self.num_layers
        self.held = [0] * self.num_layers
