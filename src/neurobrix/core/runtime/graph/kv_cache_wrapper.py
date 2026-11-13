"""
The compiled engine's KV attention interceptor over the state engine's paged cache (step 2).

The cache is core/runtime/graph/paged_kv_cache.PagedKVCache: per-layer pools of fixed pages on the device that layer's
attention runs on (LOCALITY: a model split across cards keeps each layer's state on its card), granted as sequences
grow, the sequence's K / V gathered through the page tables into what the attention reads.

STRATEGY COMPATIBILITY:
- Single GPU: All cache on cuda:0
- FGP (Fine-Grained Pipeline): Cache follows layer weights
- TP (Tensor Parallel): Each GPU stores its portion of KV heads
- PP (Pipeline Parallel): Each stage has its own cache
- Zero3: Similar to FGP, follows weight placement

ZERO HARDCODE: All config values from defaults.json lm_config.
ZERO FALLBACK: Missing layer_idx annotation = explicit error.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from neurobrix.core.dtype.config import get_torch_dtype
from neurobrix.core.runtime.graph.attention_mask import require_causal_frozen_mask


@dataclass
class KVCacheConfig:
    """
    KV Cache configuration from defaults.json lm_config section.

    NOTE: No device parameter - device is determined per-layer at runtime
    based on the incoming q tensor's device (lazy allocation).

    All values MUST be present - ZERO FALLBACK.
    Supports asymmetric K/V dimensions (MLA: k=192, v=128).
    """
    num_layers: int
    num_kv_heads: int
    k_head_dim: int
    v_head_dim: int
    max_cache_len: int
    dtype: str                  # Dtype string — converted to torch.dtype at allocation boundaries
    initial_cache_len: int = 0  # Initial buffer size (0 = allocate at max, on-demand growth)
    ceiling_is_window: bool = False  # True when max_cache_len is the MODEL's context window (prompt-aware sizing) — the overflow message names the right authority
    page_tokens: int = 0             # positions per state page (core/state/price.paged_cache_params); the paged cache
    async_release: bool = False      # the vendor file's memory.async_release (read with page_tokens, never assumed)

    @property
    def head_dim(self) -> int:
        """Backward compat for code that reads .head_dim"""
        return self.k_head_dim


class KVBranch:
    """One decode context's state on the attention wrapper (R30 mirror of
    triton/kv_cache.KVBranch): its cache and the prefill / call / position
    bookkeeping, swapped by KVCacheAttentionWrapper.use_branch."""
    __slots__ = ("cache", "is_prefill", "call_count", "position_offset", "decode_actual_seq_len")

    def __init__(self, cache, is_prefill: bool, call_count: int, position_offset: int, decode_actual_seq_len=None):
        self.cache = cache
        self.is_prefill = is_prefill
        self.call_count = call_count
        self.position_offset = position_offset
        self.decode_actual_seq_len = decode_actual_seq_len


def _partial(qg, k, v, sc: float, allowed):
    """(acc, m, l) of query rows qg [H_kv, n, D] over keys k / v [H_kv, P, D] (fp32): the unnormalised attention, each
    row's maximum score (-inf: no key) and the sum of its exponentials. ONE score matrix alive: scaled, masked
    (`allowed` [n, P] bool, None: every key) and exponentiated in place."""
    s = torch.matmul(qg, k.transpose(1, 2))
    s.mul_(sc)
    if allowed is not None:
        s.masked_fill_(~allowed, float("-inf"))
    m = s.amax(dim=2)
    s.sub_(m.masked_fill(torch.isneginf(m), 0.0)[..., None]).exp_()
    return torch.matmul(s, v), m, s.sum(dim=2)


def _merge_partials(a, b):
    """Two partials of the same rows over disjoint keys as one (log-sum-exp); `a` None: `b`. A row with no key in
    a part (l = 0) weighs nothing."""
    if a is None:
        return b
    (acc, m, l), (acc2, m2, l2) = a, b
    mn = torch.maximum(m, m2)
    c1 = torch.where(l > 0, torch.exp(m - mn), torch.zeros_like(l))
    c2 = torch.where(l2 > 0, torch.exp(m2 - mn), torch.zeros_like(l2))
    return acc * c1[..., None] + acc2 * c2[..., None], mn, l * c1 + l2 * c2


class KVCacheAttentionWrapper:
    """
    Intercepts scaled_dot_product_attention for KV cache injection.

    DISTRIBUTED ARCHITECTURE:
    - Uses the paged cache (per-layer pools on each layer's device) for multi-GPU support
    - Device determined from incoming q tensor (lazy allocation)
    - Works automatically with FGP, TP, PP, Single GPU

    The built graph contains attention(Q, K, V) WITHOUT cache context.
    This wrapper:
    1. Reads layer_idx from op annotation (preferred) OR uses call counter (fallback)
    2. Concatenates incoming K/V with cached values
    3. Updates cache
    4. Calls attention with full context
    """

    # no prefill chunk under way until `begin_chunk` (an instance built without __init__ reads these)
    _chunk_start = 0
    _chunk_real = None
    _whole_pages = False

    def __init__(self, config: KVCacheConfig, num_heads: int = 0):
        """
        Initialize wrapper with distributed cache.

        Args:
            config: KVCacheConfig (no device - determined lazily per-layer)
            num_heads: Total number of query heads (for GQA un-expand/re-expand).
                       If 0, auto-detected from first Q tensor.
        """
        self.config = config
        self.cache = self._new_cache()
        self._is_prefill = True
        self._call_count = 0
        self._num_layers = config.num_layers
        self._num_kv_heads = config.num_kv_heads
        self._num_heads = num_heads  # 0 = auto-detect from Q
        self.spill_scores_bytes: Optional[int] = None   # the plan's fp32 scores a spilled prefill chunk may write at once
        self._gqa_group_size = 0  # 0 = not yet computed
        self._position_offset = 0
        # During decode, K/V may be padded (e.g., [B, heads, 64, dim] instead of [B, heads, 1, dim])
        # This tracks the ACTUAL number of tokens to cache (not the padded length)
        self._decode_actual_seq_len: Optional[int] = None
        # The prefill chunk under way (`begin_chunk`, R30 mirror of triton/kv_cache.py): its first position, its real
        # rows, and whether it attends over the cache's keys gathered as whole pages.
        self._chunk_start = 0
        self._chunk_real: Optional[int] = None
        self._whole_pages = False
        self._chunk_masks: Dict = {}

    def begin_chunk(self, start: int, real: int, whole_pages: bool) -> None:
        """The next prefill run is one chunk: positions start .. start+rows-1 of which `real` are the request's, the
        rest pad a planned chunk. `whole_pages`: its attention reads every cached key as whole pages
        (docs/design/prism_kv.md §5bis). The cache must hold exactly `start` positions."""
        if self.cache.get_seq_len() != int(start):
            raise RuntimeError(f"ZERO FALLBACK: a prefill chunk at position {start} over a cache of "
                               f"{self.cache.get_seq_len()} positions")
        self._is_prefill = True
        self._call_count = 0
        self._position_offset = int(start)
        self._chunk_start = int(start)
        self._chunk_real = int(real)
        self._whole_pages = bool(whole_pages)
        self._chunk_masks = {}
        self._decode_actual_seq_len = None

    def end_request(self) -> None:
        """The request ends and its positions stay in the cache (a serving session's, core/state/kv_session.py): the
        per-request state resets, the cache is kept."""
        self._is_prefill = True
        self._call_count = 0
        self._position_offset = 0
        self._decode_actual_seq_len = None
        self._chunk_start, self._chunk_real, self._whole_pages, self._chunk_masks = 0, None, False, {}

    # ── decode branches (R30 mirror of triton/kv_cache.KVBranch) ─────────
    def branch_state(self) -> "KVBranch":
        self._sync_branch()
        if getattr(self, "_branch", None) is None:
            self._branch = KVBranch(self.cache, self._is_prefill, self._call_count, self._position_offset,
                                    getattr(self, "_decode_actual_seq_len", None))
        return self._branch

    def _new_cache(self):
        """The state engine's paged cache (step 2) over the plan's cache: its page size is derived by the factory from
        the vendor file and carried by the config. A config without one is refused by name."""
        if not self.config.page_tokens:
            raise RuntimeError("ZERO FALLBACK: the KV cache config carries no page size — the factory derives it "
                               "(core/state/price.paged_cache_params) from the executing card's vendor file")
        from neurobrix.core.runtime.graph.paged_kv_cache import PagedKVCache
        return PagedKVCache(self.config, self.config.page_tokens, self.config.async_release)

    def new_branch(self) -> "KVBranch":
        return KVBranch(self._new_cache(), True, 0, 0, None)

    def use_branch(self, st: "KVBranch") -> None:
        self._sync_branch()
        self._branch = st
        self.cache = st.cache
        self._is_prefill = st.is_prefill
        self._call_count = st.call_count
        self._position_offset = st.position_offset
        self._decode_actual_seq_len = st.decode_actual_seq_len

    def _sync_branch(self) -> None:
        b = getattr(self, "_branch", None)
        if b is not None:
            b.cache, b.is_prefill, b.call_count, b.position_offset = (
                self.cache, self._is_prefill, self._call_count, self._position_offset)
            b.decode_actual_seq_len = getattr(self, "_decode_actual_seq_len", None)

    def intercept_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = True,
        scale: Optional[float] = None,
        layer_idx: int = -1,
        k_pre_transposed: Optional[bool] = None,
        v_pre_transposed: Optional[bool] = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        """
        Intercept SDPA for KV cache injection.
        `enable_gqa` is the graph's own SDPA kwarg (a decoder built with grouped keys, chatterbox's
        t3 backbone, 2026-10-18): the cache keeps the key-value heads unexpanded and re-expands them
        for the call, so the final SDPA names it only when its keys still carry fewer heads than q.

        DEVICE LOCALITY: Cache is allocated on k.device automatically.
        This ensures FGP/TP compatibility without explicit device management.

        Args:
            q: Query [batch, heads, seq_q, head_dim]
            k: Key [batch, kv_heads, seq_k, head_dim]
            v: Value [batch, kv_heads, seq_v, head_dim]
            attn_mask: Optional attention mask
            dropout_p: Dropout probability (0 for inference)
            is_causal: Whether to use causal masking
            scale: Optional attention scale
            layer_idx: Layer index from graph.json annotation (-1 = use call counter)

        Returns:
            attention output [batch, heads, seq_q, head_dim]
        """
        # K/V may arrive transposed [B,H,D,S] from pattern-reassembled SDPA;
        # SDPA and KV cache expect [B,H,S,D].
        #
        # The graph is the authority (GraphExecutor._mark_sdpa_k_layout,
        # bound onto this interceptor by CompiledSequence._compile_op). The
        # shape test kept below for callers that carry no recorded layout
        # cannot decide the square case — at seq_len == head_dim both layouts
        # have the same shape — and deciding it wrongly ran attention with
        # K's axes crossed (measured 2026-09-07: TinyLlama at a 64-token
        # prompt, argmax 29892 at 6.41 against the float64 oracle's 3864 at
        # 22.36). It refuses there rather than guess.
        if k.ndim == 4 and q.ndim == 4:
            if k_pre_transposed is not None:
                if k_pre_transposed:
                    k = k.transpose(-2, -1)
            elif (k.shape[-1] == q.shape[-1]
                    and k.shape[-2] == q.shape[-2] == q.shape[-1]):
                raise RuntimeError(
                    f"KV interceptor: K is square {tuple(k.shape)} with "
                    f"seq_len == head_dim and no layout was recorded for this "
                    f"op. Refusing to guess.")
            elif (k.shape[-1] != q.shape[-1] and k.shape[-2] == q.shape[-1]):
                k = k.transpose(-2, -1)
        if v.ndim == 4 and q.ndim == 4:
            if v_pre_transposed is not None:
                if v_pre_transposed:
                    v = v.transpose(-2, -1)
            elif (v.shape[-1] == q.shape[-1]
                    and v.shape[-2] == q.shape[-2] == q.shape[-1]):
                raise RuntimeError(
                    f"KV interceptor: V is square {tuple(v.shape)} with "
                    f"seq_len == head_dim and no layout was recorded for this "
                    f"op. Refusing to guess.")
            elif (v.shape[-1] != q.shape[-1] and v.shape[-2] == q.shape[-1]):
                v = v.transpose(-2, -1)

        # Slice attention mask to match runtime sequence length
        if attn_mask is not None:
            seq_q, seq_k = q.shape[2], k.shape[2]
            if attn_mask.shape[-2] > seq_q or attn_mask.shape[-1] > seq_k:
                attn_mask = attn_mask[..., :seq_q, :seq_k].contiguous()

        # Resolve layer_idx: explicit annotation OR call counter fallback
        if layer_idx < 0:
            layer_idx = self._call_count % self._num_layers
            self._call_count += 1

        # During decode with padding, slice K/V to only cache real tokens
        # E.g., if K is [B, heads, 64, dim] but actual_seq_len=1, slice to [B, heads, 1, dim]
        if not self._is_prefill and self._decode_actual_seq_len is not None:
            actual_len = self._decode_actual_seq_len
            k = k[:, :, :actual_len, :]
            v = v[:, :, :actual_len, :]

        # GQA: The built graph may expand K/V from num_kv_heads to num_heads
        # via aten::expand before SDPA. Cache stores at num_kv_heads to save memory.
        # Un-expand before caching, re-expand after retrieval.
        incoming_heads = k.shape[1]
        if self._gqa_group_size == 0:
            # Auto-detect GQA on first call
            if self._num_heads == 0:
                self._num_heads = q.shape[1]
            if incoming_heads > self._num_kv_heads and incoming_heads == self._num_heads:
                self._gqa_group_size = self._num_heads // self._num_kv_heads
            else:
                self._gqa_group_size = 1  # MHA or no expansion in graph

        if self._gqa_group_size > 1 and incoming_heads == self._num_heads:
            # Un-expand: [B, num_heads, S, D] → [B, num_kv_heads, group, S, D] → take [:, :, 0]
            batch, _, seq_len, head_dim = k.shape
            k = k.view(batch, self._num_kv_heads, self._gqa_group_size, seq_len, head_dim)[:, :, 0]
            v = v.view(batch, self._num_kv_heads, self._gqa_group_size, seq_len, v.shape[-1])[:, :, 0]

        if self._is_prefill and self._whole_pages:
            return self._prefill_over_pages(q, k, v, attn_mask, dropout_p, scale, layer_idx)

        # Update cache (lazy allocation on k's device)
        k_full, v_full = self.cache.update(layer_idx, k, v)
        if self.cache.spilled and not self._is_prefill:
            return self._decode_over_spill(q, k_full, v_full, attn_mask, scale, layer_idx)

        # GQA: Re-expand cached K/V from num_kv_heads to num_heads for SDPA
        if self._gqa_group_size > 1:
            batch, kv_heads, cached_len, k_dim = k_full.shape
            k_full = k_full.unsqueeze(2).expand(-1, -1, self._gqa_group_size, -1, -1).reshape(batch, self._num_heads, cached_len, k_dim)
            v_dim = v_full.shape[-1]
            v_full = v_full.unsqueeze(2).expand(-1, -1, self._gqa_group_size, -1, -1).reshape(batch, self._num_heads, cached_len, v_dim)

        # Drop stale mask from pattern-reassembled SDPA.
        # After KV cache concatenation, k_full's seq_len grows beyond
        # the mask's static dimensions. The wrapper handles causal
        # masking via is_causal (prefill) or attend-all (decode).
        mask_dropped = False
        if attn_mask is not None:
            kv_seq = k_full.shape[2]
            if attn_mask.shape[-1] != kv_seq:
                # The mask's content decides, never its shape alone: only the
                # exact causal pattern has a defined extension (prefill: the
                # causal flag with equal query and key lengths; decode: every
                # cached key). Anything else — padding, a window, a bias — is
                # refused by name (attention_mask.require_causal_frozen_mask).
                require_causal_frozen_mask(
                    attn_mask, q.shape[2], kv_seq,
                    f"KV interceptor (layer {layer_idx}, "
                    f"{'prefill' if self._is_prefill else 'decode'})",
                    align_lengths=self._is_prefill)
                attn_mask = None
                mask_dropped = True

        # Determine causal masking
        # CRITICAL FIX: During decode with KV cache, is_causal=True is WRONG!
        # PyTorch's is_causal creates a lower-triangular mask based on Q length.
        # With padded Q (seq=64) and cached K/V (seq=cache_len+1), position 0 of Q
        # would only see position 0 of K, instead of ALL cached positions!
        # Solution: During decode (not prefill), disable is_causal and attend to all K/V.
        if not self._is_prefill:
            # Decode: Q position 0 should attend to ALL cached K/V (no causal mask needed)
            use_causal = False
        elif mask_dropped:
            # Prefill with dropped mask: use causal masking to replace it.
            # Pattern-reassembled SDPA may have is_causal=False (because it
            # expected an explicit mask), but since the mask was dropped,
            # we MUST enable causal masking to prevent future-token leakage.
            use_causal = True
        else:
            # Prefill: Use causal mask as requested
            use_causal = is_causal

        # Align SDPA inputs to the KV cache dtype (set by Prism plan at allocation).
        # When bf16→fp16 on V100, query may arrive fp32 from graph intermediates
        # while KV cache stores fp16. Cast q DOWN so SDPA runs entirely in the
        # Prism-resolved dtype and its output propagates fp16 downstream.
        cache_dtype = k_full.dtype  # KV cache dtype = Prism plan dtype
        if q.dtype != cache_dtype:
            q = q.to(cache_dtype)
        if attn_mask is not None and attn_mask.dtype != cache_dtype and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(cache_dtype)

        # Fix broadcast-scalar masks for SDPA efficient attention kernel.
        # MLA models (DeepSeek-V2) produce masks shaped [B, heads, q_len, 1].
        # The efficient attention kernel requires the last dim to be contiguous,
        # but broadcast (stride=0) violates this. Expand + contiguous fixes it.
        # Image-family masks (Janus) already have correct kv_len → unchanged.
        kv_len = k_full.shape[2]
        if attn_mask is not None and attn_mask.dim() == 4 and attn_mask.shape[-1] < kv_len:
            attn_mask = attn_mask.expand(-1, -1, -1, kv_len).contiguous()

        # Call SDPA with full cached context.
        # Triton mode: Dao-AILab Flash Attention kernel. Native: PyTorch F.sdpa.
        return F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=use_causal,
            scale=scale,
            **({"enable_gqa": True} if k_full.shape[1] != q.shape[1] else {}),
        )

    def _prefill_over_pages(self, q, k, v, attn_mask, dropout_p, scale, layer_idx: int) -> torch.Tensor:
        """A prefill chunk's attention over the session's keys gathered as WHOLE PAGES (R30 mirror of the Triton
        interceptor's): the chunk's K/V (its KV heads) appended to the pages, every cached key read back from them in
        the cache's dtype, the key length rounded up to whole pages, an explicit 0/-inf mask (causal at the cached
        length, the keys past the true end masked). A graph mask is replaced only when it is the causal pattern."""
        start = self.cache.lens.get(layer_idx, 0)    # the positions this layer holds before the chunk appends
        rows = int(q.shape[2])
        if attn_mask is not None:
            require_causal_frozen_mask(attn_mask, rows, rows, f"KV interceptor (layer {layer_idx}, prefill chunk)",
                                       align_lengths=True)
        self.cache.update(layer_idx, k, v, real=self._chunk_real)
        if self.cache.spilled:
            return self._prefill_over_spill(q, layer_idx, start, rows, scale)
        tpp = self.cache.page_tokens
        k_full, v_full = self.cache.whole_pages(layer_idx, -(-(start + rows) // tpp) * tpp)
        return self._attend_per_query_group(q.to(k_full.dtype), k_full, v_full, start, dropout_p, scale)

    def _attend_per_query_group(self, q, k_full, v_full, start: int, dropout_p, scale) -> torch.Tensor:
        """q [1, H, rows, D] at cached position `start` over k_full / v_full [1, H_kv, keys, D], causal: ONE attention
        per query group of a KV head over the chunk's whole rows, K / V as the KV heads — never repeated for every
        query head (groups x the cached keys, about 1 GB at a V100's reach, was copied at every layer of every pass).
        Its dense [rows, keys] mask is the session chunk's priced reserve (`price.session_chunk_attention_bytes`).
        Measured against cutting the rows too (one page at a time, 256 calls a layer): 141.9 s against 44.2 s for a
        19 931-token turn of Qwen3-Coder-30B-A3B on two V100-32GB (2026-11-10) — the calls cost, the rows stay whole."""
        rows, heads, kv_heads = int(q.shape[2]), int(q.shape[1]), int(k_full.shape[1])
        groups = heads // kv_heads
        mask = self._chunk_mask(q.device, q.dtype, start, rows, int(k_full.shape[2]))
        qg = q.reshape(1, kv_heads, groups, rows, int(q.shape[3]))          # query head h = kv head * groups + g
        got = [F.scaled_dot_product_attention(qg[:, :, g], k_full, v_full, attn_mask=mask, dropout_p=dropout_p,
                                              is_causal=False, scale=scale) for g in range(groups)]
        return torch.stack(got, dim=2).reshape(1, heads, rows, -1)

    def _chunk_mask(self, device, dtype, start: int, rows: int, keys: int) -> torch.Tensor:
        """The chunk's additive mask [rows, keys]: 0 where key j <= start + i (causal, bottom-right at the cached
        length — torch's `is_causal` aligns top-left when the query is shorter than the keys), -inf elsewhere, which
        also masks the whole pages' keys past the true end. One per device and chunk, written in place: zeros, the
        triangle over the chunk's own columns, -inf past them (no boolean rows x keys beside it)."""
        key = (str(device), dtype)
        m = self._chunk_masks.get(key)
        if m is None:
            m = self._chunk_masks[key] = torch.zeros((rows, keys), dtype=dtype, device=device)
            m[:, start:].fill_(float("-inf"))
            m[:, start:start + rows].triu_(1)                                # row i keeps -inf strictly after its own key
        return m

    def _prefill_over_spill(self, q, layer_idx: int, start: int, rows: int, scale) -> torch.Tensor:
        """A prefill chunk of a sequence whose first positions spilled to host memory (R30 mirror of nbxtensor
        memory/paged_kv._chunk_attention_spilled): the card's keys and the spilled ones (the library's blocks), read in
        key blocks by spans of query rows, each (span, block) one fp32 partial, merged by log-sum-exp in a fixed
        order — exact. (The host partial is the decode step's, one row.) What it holds at once is cut to the session
        chunk's reserve (`spill_scores_bytes`, `_spill_cut`): ONE score matrix and one fp32 key block."""
        cache = self.cache
        card_start = start - cache.spilled
        keys = card_start + rows                                             # the card's keys the chunk may read
        k_card, v_card = cache._whole(layer_idx, -(-keys // cache.page_tokens) * cache.page_tokens)
        heads, d, kv_heads = int(q.shape[1]), int(q.shape[3]), int(k_card.shape[1])
        groups, cache_dtype = heads // kv_heads, k_card.dtype
        sc = float(scale) if scale is not None else d ** -0.5
        block, span = self._spill_cut(heads, kv_heads, d + int(v_card.shape[3]), keys)
        qr = q.to(cache_dtype).float()[0].reshape(kv_heads, groups, rows, d)  # q as the card reads it
        out = []
        for a in range(0, rows, span):
            b = min(a + span, rows)
            qg, got = qr[:, :, a:b].reshape(kv_heads, groups * (b - a), d), None
            row_key = card_start + torch.arange(a, b, device=q.device)       # the last card key each row may read
            for c in range(0, card_start + b, block):                        # the card's keys, block by block
                e = min(c + block, card_start + b)
                allowed = None if e <= card_start + a + 1 else (
                    torch.arange(c, e, device=q.device)[None, :] <= row_key[:, None]).repeat(groups, 1)
                got = _merge_partials(got, _partial(qg, k_card[0, :, c:e].float(), v_card[0, :, c:e].float(), sc, allowed))
            for kb, vb in cache.host_spill.blocks(layer_idx, block):          # every spilled position precedes the chunk
                got = _merge_partials(got, _partial(qg, torch.from_numpy(kb).to(q.device),
                                                    torch.from_numpy(vb).to(q.device), sc, None))
            acc, _m, l = got
            out.append((acc / torch.where(l > 0, l, torch.ones_like(l))[..., None]).reshape(heads, b - a, -1))
        return torch.cat(out, dim=1)[None].to(cache_dtype)

    def _spill_cut(self, heads: int, kv_heads: int, kv_dims: int, keys: int) -> Tuple[int, int]:
        """(key positions a block, query rows a span) of a spilled chunk under `spill_scores_bytes`: a quarter of it at
        most for one fp32 key block (K and V of whole pages), the rest for the one score matrix a (span, block) writes
        (4 bytes x heads x rows x positions). Unstated (a plan with no session chunk): one page of rows over `keys`.
        Refused by name when one row over one page does not fit."""
        tpp = self.cache.page_tokens
        budget = self.spill_scores_bytes
        if budget is None:
            budget = 4 * heads * tpp * keys + 4 * kv_heads * kv_dims * tpp
        block = max(tpp, min(-(-keys // tpp) * tpp, (int(budget) // 4) // (4 * kv_heads * kv_dims) // tpp * tpp))
        span = (int(budget) - 4 * kv_heads * kv_dims * block) // (4 * heads * block)
        if span < 1:
            raise RuntimeError(f"ZERO FALLBACK: one row of a spilled prefill chunk over one block of {block} keys "
                               f"writes {4 * heads * block} bytes of scores beside {4 * kv_heads * kv_dims * block} "
                               f"bytes of keys; the plan leaves {budget}")
        return block, span

    def _decode_over_spill(self, q, k_card, v_card, attn_mask, scale, layer_idx: int) -> torch.Tensor:
        """A decode step of a sequence whose oldest positions spilled to host memory: the card's partial over its
        positions (unnormalised acc, max m, sum l, fp32, as the Triton split kernel's) and the host's over the spilled
        ones (nbxtensor ops/host_decode_partial, the same op as the Triton engine's), merged by log-sum-exp — exact.
        q [1, H, 1, D] · k_card / v_card [1, H_kv, T_card, D] (the KV heads, never expanded) -> [1, H, 1, D_v]."""
        if attn_mask is not None:
            require_causal_frozen_mask(attn_mask, 1, int(attn_mask.shape[-1]),
                                       f"KV interceptor (layer {layer_idx}, decode over spilled positions)")
        cache_dtype = k_card.dtype
        heads, d = int(q.shape[1]), int(q.shape[3])
        kv_heads = int(k_card.shape[1])
        groups = heads // kv_heads
        sc = float(scale) if scale is not None else d ** -0.5
        qr = q.to(cache_dtype).float()                                     # q as the card reads it
        qg = qr.reshape(kv_heads, groups, d)
        s = torch.matmul(qg, k_card[0].float().transpose(1, 2)) * sc       # [H_kv, groups, T_card]
        m = s.amax(dim=2)
        p = torch.exp(s - m[..., None])
        acc, l = torch.matmul(p, v_card[0].float()), p.sum(dim=2)
        h_acc, h_m, h_l = (torch.from_numpy(x).to(q.device) for x in self.cache.host_spill.partial(
            layer_idx, qr.reshape(heads, d).cpu().numpy(), groups, sc))
        h_acc, h_m, h_l = h_acc.reshape(kv_heads, groups, -1), h_m.reshape(kv_heads, groups), h_l.reshape(kv_heads, groups)
        mn = torch.maximum(m, h_m)
        c1 = torch.where(torch.isneginf(m), torch.zeros_like(m), torch.exp(m - mn))
        c2 = torch.where(torch.isneginf(h_m), torch.zeros_like(h_m), torch.exp(h_m - mn))
        out = (acc * c1[..., None] + h_acc * c2[..., None]) / (l * c1 + h_l * c2)[..., None]
        return out.reshape(1, heads, 1, -1).to(cache_dtype)

    def reset_for_new_sequence(self) -> None:
        """Reset cache for a new generation sequence."""
        self.cache.clear()
        self._is_prefill = True
        self._call_count = 0
        self._position_offset = 0
        self._decode_actual_seq_len = None
        self._chunk_start, self._chunk_real, self._whole_pages, self._chunk_masks = 0, None, False, {}

    def set_decode_mode(self, actual_seq_len: Optional[int] = None) -> None:
        """
        Switch to decode mode after prefill.

        Args:
            actual_seq_len: Number of actual tokens per decode step (not padded length).
                           For VQ image generation, this is typically 1.
                           If None, no padding adjustment is applied.
        """
        self._is_prefill = False
        self._position_offset = self.cache.get_seq_len()
        self._decode_actual_seq_len = actual_seq_len
        self._chunk_start, self._chunk_real, self._whole_pages, self._chunk_masks = 0, None, False, {}

    def get_cache_len(self) -> int:
        """Get current cache sequence length."""
        return self.cache.get_seq_len()

    def update_position_offset(self) -> None:
        """Update position offset for next decode step."""
        if not self._is_prefill:
            self._position_offset = self.cache.get_seq_len()

    @staticmethod
    def _sanitize_device_kwarg(kwargs: dict) -> dict:
        """Remap a baked graph device that is invalid in this process.

        The builder freezes the physical build device (e.g. cuda:2) into
        aten::arange kwargs. The normal dispatch path rewrites it via the
        Prism runtime device (sequential_dispatcher._resolve_attr_value),
        but this KV-cache interceptor sits on a separate path and receives
        the raw kwarg. Under a CUDA_VISIBLE_DEVICES mask the frozen index
        can exceed the visible device count → `invalid device ordinal`
        (Janus-Pro under --hardware v100-32g pinned to one GPU). Remap ONLY
        when the index is out of range (the masked case); a valid index is
        left untouched, so LLMs whose arange already resolves to cuda:0 are
        a strict no-op. Placement-agnostic (R19) — no compute change.
        """
        dev = kwargs.get("device")
        if (isinstance(dev, torch.device) and dev.type == "cuda"
                and dev.index is not None
                and dev.index >= torch.cuda.device_count()):
            return {**kwargs,
                    "device": torch.device("cuda", torch.cuda.current_device())}
        return kwargs

    def intercept_arange(self, *args, **kwargs) -> torch.Tensor:
        """
        Intercept aten::arange to fix RoPE positions during decode.

        Problem: During decode, arange(seq_len=1) produces [0.0], building
        a 1-row RoPE cos/sin table. But position_ids=[[cache_len]] indexes
        row cache_len → OOB.

        Fix: Shift arange START to cache_len so it produces [cache_len, ...,
        cache_len+seq_len-1]. The output SIZE stays the same (seq_len elements),
        preserving symbolic shapes. Combined with relative position_ids=[[0]],
        the index correctly selects row 0 of a table containing RoPE values
        for the absolute position.

        Float aranges (RoPE) AND integer aranges (position_ids for encoder-decoder
        models like Whisper V3 Turbo) are shifted. This is safe because MoE routing
        aranges live in the model component, not the decoder — the KV cache wrapper
        only intercepts ops in the decoder component where aranges are positional.
        """
        kwargs = self._sanitize_device_kwarg(kwargs)
        if self._is_prefill and not self._chunk_start:
            return torch.arange(*args, **kwargs)

        cache_len = self._position_offset
        if cache_len > 0 and args and isinstance(args[0], (int, float)):
            # Shift the positional window to absolute decode positions by cache_len.
            # Two graph forms occur:
            #   arange(end)        — relative [0, end)      → [cache_len, cache_len+end)
            #   arange(start, end) — relative [start, end)  → [cache_len+start, cache_len+end)
            # The single-arg path is byte-identical to the prior behaviour (start=0).
            # The two-arg form is what Llama-class graphs emit for cache_position
            # (arange(0, seq_len) with start frozen at 0 from build); treating its
            # args[0]=0 as `end` previously produced arange(cache_len, cache_len) =
            # an EMPTY range, collapsing the RoPE cos/sin and the q/k seq dims at decode.
            if len(args) == 1:
                return torch.arange(cache_len, cache_len + args[0], **kwargs)
            if isinstance(args[1], (int, float)):
                start, end = args[0], args[1]
                return torch.arange(cache_len + start, cache_len + end, *args[2:], **kwargs)

        return torch.arange(*args, **kwargs)

    def intercept_position_slice(self, table, dim=0, start=0, end=None, step=1):
        """Intercept `aten::slice(table, dim, 0, seq_len)` — the rows of a
        positional table a whisper-class decoder reads for its tokens. At
        decode the graph asks for rows [0, 1): the token would sit at
        position 0 every step (whisper-large, 2026-09-05: the language token
        re-predicted 448 times). Shift the window by the cache length, same
        size, same semantics as intercept_arange."""
        if (self._is_prefill and not self._chunk_start) or self._position_offset <= 0 or end is None:
            return torch.ops.aten.slice(table, dim, start, end, step)
        cache_len = self._position_offset
        return torch.ops.aten.slice(table, dim, cache_len + start, cache_len + end, step)

    def intercept_position_cumsum(self, x, dim=0, dtype=None):
        """The positions a decoder counts over its own sequence (`cumsum(ones_like(tokens))`, the
        dual-AR decoder's rotary index): at decode the count restarts at 1 for the one token handed
        in, so the cache's length is added — the token's absolute position, the output size kept."""
        out = torch.ops.aten.cumsum(x, dim, dtype=dtype)
        if (self._is_prefill and not self._chunk_start) or self._position_offset <= 0:
            return out
        return out + self._position_offset

    def get_cache_info(self) -> Dict:
        """Get cache statistics."""
        return {
            "seq_len": self.cache.get_seq_len(),
            "num_layers": self.cache.config.num_layers,
            "is_prefill": self._is_prefill,
            "position_offset": self._position_offset,
            "layer_devices": self.cache.get_layer_devices(),
        }

    # ==========================================================================
    # Variant-specific interceptors for different SDPA op signatures
    # ==========================================================================

    def intercept_efficient_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        compute_log_sumexp: bool = False,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        *,
        scale: Optional[float] = None,
        k_pre_transposed: Optional[bool] = None,
        v_pre_transposed: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Intercept aten::_scaled_dot_product_efficient_attention.

        Signature: (q, k, v, attn_bias, compute_log_sumexp, dropout_p, is_causal, *, scale)
        Returns: (output, log_sumexp, philox_seed, philox_offset)
        """
        if isinstance(is_causal, (float, int)) and not isinstance(is_causal, bool):
            is_causal = bool(is_causal)

        output = self.intercept_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=float(dropout_p) if dropout_p is not None else 0.0,
            is_causal=is_causal,
            scale=scale,
            layer_idx=-1,
            k_pre_transposed=k_pre_transposed,
            v_pre_transposed=v_pre_transposed,
        )

        return (output, None, None, None)


    def intercept_flash_attention_for_cpu(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = True,
        *,
        scale: Optional[float] = None,
        k_pre_transposed: Optional[bool] = None,
        v_pre_transposed: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Intercept aten::_scaled_dot_product_flash_attention_for_cpu.

        This variant has a DIFFERENT signature than standard SDPA:
        (q, k, v, dropout_p, is_causal, *, scale=None)

        NO attn_mask parameter!

        CRITICAL: Returns a TUPLE (attention_output, logsumexp) to match
        the original op's output format. The graph unpacks both outputs.

        Used by Janus and other models on V100 (no native bf16 flash attention).
        """
        output = self.intercept_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            layer_idx=-1,
            k_pre_transposed=k_pre_transposed,
            v_pre_transposed=v_pre_transposed,
        )

        # Create logsumexp output (not used in inference but graph expects it)
        # Shape: [batch, heads, seq_len]
        batch, heads, seq_len, _ = output.shape
        logsumexp = torch.zeros(batch, heads, seq_len, dtype=output.dtype, device=output.device)

        return (output, logsumexp)

    def get_interceptors(self) -> Dict[str, Callable]:
        """
        Get dictionary mapping op_type to appropriate interceptor method.

        Returns:
            Dict mapping op_types to appropriate interceptor methods
        """
        return {
            "aten::scaled_dot_product_attention": self.intercept_attention,
            "aten::_scaled_dot_product_efficient_attention": self.intercept_efficient_attention,
            "aten::_scaled_dot_product_flash_attention": self.intercept_attention,
            "aten::_scaled_dot_product_flash_attention_for_cpu": self.intercept_flash_attention_for_cpu,
            "aten::_scaled_dot_product_cudnn_attention": self.intercept_attention,
            "aten::arange": self.intercept_arange,
        }

