"""The Triton engine's KV attention interceptor and decode branches.

The cache it drives is the state engine's paged cache (triton/paged_kv_cache.py, step 2): prefill appends into the
pages and attends over its own K / V, decode appends at the device length and reads through the block table. The
interceptor asks the cache for its state through one protocol (get_cache_len, skip_positions, replay_key,
replay_advance, replay_restore, replay_position_diag). Zero torch dependency.
"""


class KVBranch:
    """One decode context's state on an attention interceptor: its cache and
    the prefill / call / position bookkeeping (see use_branch)."""
    __slots__ = ("cache", "is_prefill", "call_count", "position_offset")

    def __init__(self, cache, is_prefill: bool, call_count: int, position_offset: int):
        self.cache = cache
        self.is_prefill = is_prefill
        self.call_count = call_count
        self.position_offset = position_offset


class TritonAttentionInterceptor:
    """Intercepts SDPA for KV cache + calls Triton Flash Attention."""

    def __init__(self, cache, num_heads: int = 0):
        self.cache = cache
        self._is_prefill = True
        self._call_count = 0
        self._num_heads = num_heads
        self._num_kv_heads = cache.num_kv_heads
        self._gqa_group_size = 0
        # Absolute decode position for RoPE arange shifting (see intercept_arange).
        self._position_offset = 0

    # ── decode branches ─────────────────────────────────────────────────
    # The interceptor is registered once per executor; a flow that keeps
    # two contexts alive on the same LM (VibeVoice: the positive prompt+audio
    # context and the CFG negative audio-only context) swaps the per-sequence
    # state — the cache and the prefill/position bookkeeping — between them.
    # `branch_state()` is the live state; `new_branch()` an empty one with
    # the same geometry; `use_branch()` makes one live, saving the other.
    def branch_state(self) -> "KVBranch":
        self._sync_branch()
        if getattr(self, "_branch", None) is None:
            self._branch = KVBranch(self.cache, self._is_prefill, self._call_count, self._position_offset)
        return self._branch

    def new_branch(self) -> "KVBranch":
        return KVBranch(self.cache.clone_empty(), True, 0, 0)

    def use_branch(self, st: "KVBranch") -> None:
        self._sync_branch()
        self._branch = st
        self.cache = st.cache
        self._is_prefill = st.is_prefill
        self._call_count = st.call_count
        self._position_offset = st.position_offset

    def _sync_branch(self) -> None:
        b = getattr(self, "_branch", None)
        if b is not None:
            b.cache, b.is_prefill, b.call_count, b.position_offset = (
                self.cache, self._is_prefill, self._call_count, self._position_offset)

    def intercept(self, q, k, v, attn_mask=None, dropout_p=0.0,
                  is_causal=True, scale=None, layer_idx=-1,
                  k_pre_transposed=None, enable_gqa=None):
        # `enable_gqa`: the graph's own SDPA kwarg (grouped keys); the interceptor groups keys by its
        # own ratio (`_gqa_group_size`), so the kwarg is accepted and carries nothing more here.
        """Intercept SDPA: update KV cache for decode, passthrough for prefill.

        Self-managed dtype (Phase 1 opt-in cleanup): Flash Attention works
        in fp16/bf16 not fp32, and the interceptor casts q/k/v internally
        to the cache dtype. The DtypeEngine wrap would either no-op
        (correct dtypes already) or create unwanted upcasts. Marked
        explicit on the bound method below.
        """
        from neurobrix.kernels.wrappers import scaled_dot_product_attention_wrapper

        if layer_idx < 0:
            layer_idx = self._call_count % self.cache.num_layers
            self._call_count += 1

        # Fix pre-transposed K from graph math decomposition path.
        #
        # `k_pre_transposed` is the graph's recorded answer (GraphExecutor.
        # _mark_sdpa_k_layout); the shape test below is only for callers that
        # do not carry it, and it refuses the square case rather than guess —
        # at seq_len == head_dim the two layouts have the same shape, and
        # guessing there ran attention with K's axes crossed.
        if k.ndim == 4 and q.ndim == 4:
            if k_pre_transposed is not None:
                if k_pre_transposed:
                    k = k.transpose(2, 3).contiguous()
            elif (k.shape[2] == q.shape[3] and k.shape[3] == q.shape[2]
                    and k.shape[2] == q.shape[2]):
                raise RuntimeError(
                    f"KV interceptor: K is square {tuple(k.shape)} with "
                    f"seq_len == head_dim, so its layout cannot be read from "
                    f"its shape and none was recorded. Refusing to guess.")
            elif (k.shape[2] == q.shape[3] and k.shape[3] == q.shape[2]):
                k = k.transpose(2, 3).contiguous()

        # PREFILL: Use standard SDPA with is_causal=True (drop explicit mask).
        # Also populate the KV cache so decode steps have context.
        if self._is_prefill:
            # GQA: un-expand to KV heads for cache storage
            k_for_cache = k
            v_for_cache = v
            if self._gqa_group_size == 0:
                if self._num_heads == 0:
                    self._num_heads = q.shape[1]
                incoming = k.shape[1]
                if incoming > self._num_kv_heads and incoming == self._num_heads:
                    self._gqa_group_size = self._num_heads // self._num_kv_heads
                else:
                    self._gqa_group_size = 1

            if self._gqa_group_size > 1 and k.shape[1] == self._num_heads:
                batch, _, seq_len, head_dim = k.shape
                k_for_cache = k.view(batch, self._num_kv_heads, self._gqa_group_size, seq_len, head_dim)
                k_for_cache = k_for_cache.select(2, 0).contiguous()
                v_for_cache = v.view(batch, self._num_kv_heads, self._gqa_group_size, seq_len, v.shape[-1])
                v_for_cache = v_for_cache.select(2, 0).contiguous()

            # Store in cache for decode: into the paged cache's pages, granted for the request
            self.cache.prefill(layer_idx, k_for_cache, v_for_cache)

            # Run SDPA with full Q/K/V (graph-expanded, not cache)
            if hasattr(q, '_dtype') and hasattr(k, '_dtype') and q._dtype != k._dtype:
                k = k.to(q._dtype)
                v = v.to(q._dtype)
            return scaled_dot_product_attention_wrapper(
                q, k, v, attn_mask=None, dropout_p=dropout_p,
                is_causal=True, scale=scale, k_pre_transposed=False)

        # GQA: un-expand K/V if needed
        if self._gqa_group_size == 0:
            if self._num_heads == 0:
                self._num_heads = q.shape[1]
            incoming_heads = k.shape[1]
            if incoming_heads > self._num_kv_heads and incoming_heads == self._num_heads:
                self._gqa_group_size = self._num_heads // self._num_kv_heads
            else:
                self._gqa_group_size = 1

        if self._gqa_group_size > 1 and k.shape[1] == self._num_heads:
            batch, _, seq_len, head_dim = k.shape
            k = k.view(batch, self._num_kv_heads, self._gqa_group_size, seq_len, head_dim)[:, :, 0]
            v = v.view(batch, self._num_kv_heads, self._gqa_group_size, seq_len, v.shape[-1])[:, :, 0]

        # The paged cache: append at the device length, read through the block table (state engine step 2).
        # Each sequence reads its own length, so no pad mask exists. The graph's own decode mask covers only the
        # step's keys (the traced causal mask over the new position, [.., 1, 1] for one token): one query at the
        # last position sees every cached position, which the paged read does. A mask over MORE keys than the step
        # brought would constrain the cache and cannot be honoured: refused by name, never dropped.
        if attn_mask is not None and int(attn_mask.shape[-1]) != int(k.shape[2]):
            raise RuntimeError(
                "ZERO FALLBACK: a decode step's attention carries a mask over the cache the paged KV read does not "
                f"apply ({tuple(attn_mask.shape)}); the paged cache reads each sequence's own length")
        import math
        from neurobrix.kernels.ops.paged_decode_attn import PAGED_DECODE_JUDGED_TILE
        s = float(scale) if scale is not None else 1.0 / math.sqrt(int(q.shape[-1]))
        return self.cache.decode(layer_idx, q, k, v, s, PAGED_DECODE_JUDGED_TILE, q_round=self.cache.dtype)

    def intercept_efficient(self, q, k, v, attn_bias=None, compute_log_sumexp=False,
                            dropout_p=0.0, is_causal=False, scale=None,
                            layer_idx=-1, k_pre_transposed=None, **kwargs):
        """aten::_scaled_dot_product_efficient_attention / _cudnn_attention.

        Same KV-cache logic as intercept(), but these ATen variants insert a
        ``compute_log_sumexp`` bool at arg[4], shifting ``is_causal`` to arg[6]
        and making ``scale`` kwarg-only. Binding intercept()'s plain-SDPA
        signature directly to those args mis-reads scale (the positional
        is_causal lands in the scale slot AND a scale kwarg arrives ->
        "multiple values for 'scale'") and drops the causal flag. Remap
        explicitly. Mirrors the per-variant interceptors on the compiled side
        (core/runtime/graph/kv_cache_wrapper.py:622).
        """
        return self.intercept(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p,
                              is_causal=is_causal, scale=scale, layer_idx=layer_idx,
                              k_pre_transposed=k_pre_transposed)

    def intercept_flash(self, q, k, v, dropout_p=0.0, is_causal=False,
                        return_debug_mask=False, scale=None, layer_idx=-1,
                        k_pre_transposed=None, **kwargs):
        """aten::_scaled_dot_product_flash_attention — is_causal at arg[4], no
        attn_bias, ``scale`` kwarg-only. Remap to intercept()."""
        return self.intercept(q, k, v, attn_mask=None, dropout_p=dropout_p,
                              is_causal=is_causal, scale=scale, layer_idx=layer_idx,
                              k_pre_transposed=k_pre_transposed)

    def reset(self):
        self.cache.clear()
        self._is_prefill = True
        self._call_count = 0

    def set_decode_mode(self):
        """Mirror of core kv_cache_wrapper.set_decode_mode: the first decode step's positional offset is the
        cache's length after the prefill (it stayed 0 here: openaudio's first cached token on the Triton engine
        was rotated at position 0, 2026-10-19)."""
        self._is_prefill = False
        self._call_count = 0
        self._position_offset = self.get_cache_len()

    def update_position_offset(self):
        """Called before each decode step. Resets per-step call counter and
        captures the current cache length as the absolute RoPE position offset
        (mirror of core kv_cache_wrapper.update_position_offset)."""
        self._call_count = 0
        if not self._is_prefill:
            self._position_offset = self.get_cache_len()

    def intercept_position_slice(self, table, dim=0, start=0, end=None, step=1):
        """Mirror of core kv_cache_wrapper.intercept_position_slice: the rows
        [0, seq_len) of a positional table, shifted by the cache length at
        decode. R33-pure: an NBXTensor narrow (a view)."""
        if self._is_prefill or self._position_offset <= 0 or end is None:
            return table.narrow(int(dim), int(start), int(end - start) if end is not None else table.shape[int(dim)] - int(start))
        cache_len = self._position_offset
        return table.narrow(int(dim), cache_len + int(start), int(end - start))

    def intercept_position_cumsum(self, x, dim=0, dtype=None):
        """Mirror of core kv_cache_wrapper.intercept_position_cumsum: the positions a decoder counts
        over its own sequence, the cache's length added at decode. R33-pure: the cumsum and add
        wrappers of the kernel library."""
        from neurobrix.kernels import wrappers as W
        out = W.cumsum_wrapper(x, int(dim), dtype)
        if self._is_prefill or self._position_offset <= 0:
            return out
        return W.add(out, self._position_offset)

    def intercept_arange(self, *args, **kwargs):
        """Intercept aten::arange to fix RoPE positions during decode.

        For RoPE models with NO position_ids input (orpheus), positions come
        from an internal aten::arange(0, seq_len). At decode seq_len=1 so the
        graph emits arange(0, 1) = [0] — every decoded token is RoPE-encoded at
        position 0, the KV cache fills with mis-rotated keys, and generation
        degrades into a non-terminating ramble. Shift the window START to
        cache_len so it yields [cache_len, ..., cache_len+seq_len-1], i.e. the
        token's true absolute position, while keeping the output SIZE (symbolic
        shape) unchanged. Exact mirror of
        core/runtime/graph/kv_cache_wrapper.py:intercept_arange — the missing
        R30 half that left triton-compiled decode broken while the op-by-op
        triton-sequential oracle (full-context recompute, correct arange) passed.
        R33-pure: returns NBXTensor via the triton _create_arange dispatch.
        """
        from neurobrix.kernels.dispatch import _create_arange
        if self._is_prefill:
            return _create_arange(*args, **kwargs)
        cache_len = self._position_offset
        if cache_len > 0 and args and isinstance(args[0], (int, float)):
            # arange(end)        → [cache_len, cache_len+end)
            # arange(start, end) → [cache_len+start, cache_len+end)
            if len(args) == 1:
                return _create_arange(cache_len, cache_len + args[0], **kwargs)
            if isinstance(args[1], (int, float)):
                start, end = args[0], args[1]
                return _create_arange(cache_len + start, cache_len + end,
                                      *args[2:], **kwargs)
        return _create_arange(*args, **kwargs)

    def get_cache_len(self) -> int:
        """The cache's current length (the cache answers; every cache implements the protocol)."""
        return self.cache.get_cache_len()

    # ------------------------------------------------------------------
    # Replay registration contract (P-REPLAY-KV-DECODE) — the refined
    # interceptor guard: a sequence whose interceptors ALL implement
    # this contract becomes replay-eligible, with the interceptor state
    # entering the bucket signature. Three methods:
    #   replay_signature(funcs) -> hashable | None (None = not eligible
    #       under the CURRENT state; refused for this run only)
    #   replay_advance()  — after each replayed run: mirror the state
    #       the replayed launches advanced device-side onto the host
    #       authority (current_len drives position_ids and bucket math)
    #   replay_restore()  — before the verify pass's normal re-run:
    #       undo the replayed pass's non-idempotent device writes (the
    #       position-counter increments; k/v/mask writes are idempotent
    #       — same bytes at the same positions on identical inputs)
    # ------------------------------------------------------------------

    def replay_signature(self, registered_funcs):
        """Signature contribution, or None while ineligible.

        Eligible = decode mode, B3 bucketed path active, every layer on
        the constant-tuple append kernel (batch 1, device counter and
        mask live). The arange interceptor (position_ids-less models)
        bakes a per-step position scalar into its launch tuple — a
        recorded step would replay stale positions forever, so its
        registration refuses the whole sequence (that model class needs
        the device-side arange increment first, a named extension)."""
        import os
        for f in registered_funcs:
            if getattr(f, "__func__", None) is \
                    TritonAttentionInterceptor.intercept_arange:
                return None
        if self._is_prefill:
            return None
        return self.cache.replay_key(int(os.environ.get("NBX_KV_BUCKET", "256")))

    def skip_positions(self, n: int) -> None:
        """A census shadow skips `n` positions of the decode at once (kernels/census.py::pace):
        every layer's counter moves as if `n` tokens had been written. Never on a live run —
        the caller asks the shadow first; here the counters simply move."""
        self.cache.skip_positions(n)

    def replay_advance(self):
        """The replayed launches appended one position per layer and
        advanced every device counter; mirror on the host authority
        (position_ids and the bucket signature read current_len)."""
        self.cache.replay_advance()

    def replay_restore(self):
        """Rewind the device counters to the host authority so the
        verify pass's normal re-run appends at the same position the
        replayed pass wrote (identical inputs -> idempotent k/v/mask
        writes; only the counter increment needs undoing)."""
        self.cache.replay_restore()

    def replay_lens_snapshot(self):
        """The cache's own per-layer device lengths, for a tail replay's exact rewind (`replay.maybe_run`); the
        cache must provide it — a cache without one makes the tail decline."""
        return self.cache.replay_lens_snapshot()

    def replay_lens_restore(self, snap):
        self.cache.replay_lens_restore(snap)


# Phase 1 opt-in cleanup: mark the interceptor's underlying function
# as self-managing dtype. TritonSequence._compile_op walks
# func.__func__ for bound methods to pick up this flag.
TritonAttentionInterceptor.intercept.self_manages_dtype = True  # type: ignore[attr-defined]
TritonAttentionInterceptor.intercept_efficient.self_manages_dtype = True  # type: ignore[attr-defined]
TritonAttentionInterceptor.intercept_flash.self_manages_dtype = True  # type: ignore[attr-defined]
