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

    # no prefill chunk under way until `begin_chunk` (an instance built without __init__ reads these)
    _chunk_start = 0
    _chunk_real = None
    _whole_pages = False

    def __init__(self, cache, num_heads: int = 0):
        self.cache = cache
        self._is_prefill = True
        self._call_count = 0
        self._num_heads = num_heads
        self._num_kv_heads = cache.num_kv_heads
        self._gqa_group_size = 0
        # Absolute decode position for RoPE arange shifting (see intercept_arange).
        self._position_offset = 0
        # The prefill chunk under way (`begin_chunk`): its first position, its real rows, and whether it attends over
        # the cache's keys gathered as whole pages (a session's continuation, or a planned chunk size).
        self._chunk_start = 0
        self._chunk_real = None
        self._whole_pages = False
        self._chunk_masks = {}

    def begin_chunk(self, start: int, real: int, whole_pages: bool) -> None:
        """The next prefill run is one chunk: positions start .. start+rows-1 of which `real` are the request's, the
        rest pad a planned chunk. `whole_pages`: its attention reads every cached key as whole pages
        (docs/design/prism_kv.md §5bis). The cache must hold exactly `start` positions."""
        if self.cache.get_cache_len() != int(start):
            raise RuntimeError(f"ZERO FALLBACK: a prefill chunk at position {start} over a cache of "
                               f"{self.cache.get_cache_len()} positions")
        self._is_prefill = True
        self._call_count = 0
        self._chunk_start = int(start)
        self._chunk_real = int(real)
        self._whole_pages = bool(whole_pages)
        self._chunk_masks = {}

    def _chunk_mask(self, device_idx: int, start: int, rows: int, keys: int):
        """The chunk's additive attention mask [rows, keys]: 0 where key j <= start + i (causal, bottom-right at the
        cached length), -inf elsewhere — the values the causal bias adds — which also masks the whole pages' keys past
        the true end. One per device and chunk, every layer reads it."""
        m = self._chunk_masks.get(device_idx)
        if m is None:
            import numpy as np
            from nbxtensor.tensor import NBXTensor
            from nbxtensor.memory.allocator import DeviceAllocator
            host = np.where(np.arange(keys)[None, :] <= (start + np.arange(rows))[:, None], 0.0, -np.inf)
            DeviceAllocator.set_device(device_idx)
            m = NBXTensor.from_numpy(host.astype(np.float32))
            self._chunk_masks[device_idx] = m
        return m

    def _require_causal_graph_mask(self, attn_mask, rows: int, layer_idx: int) -> None:
        """The graph's own prefill mask is replaced by the chunk's only when it IS the causal pattern over the chunk's
        rows (R30 mirror of the compiled `require_causal_frozen_mask`): a padding, window or bias mask has no reading
        over the cached keys, refused by name. Read once per mask buffer and chunk (a host copy of rows x rows)."""
        if attn_mask is None:
            return
        key = ("graph_mask", attn_mask.data_ptr())
        if key in self._chunk_masks:
            return
        import ctypes
        import numpy as np
        from nbxtensor.dtype.nbx_dtype import NBXDtype
        from nbxtensor.memory.allocator import DeviceAllocator
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
            raise RuntimeError(f"ZERO FALLBACK: a prefill chunk's attention (layer {layer_idx}) carries a graph mask "
                               f"{tuple(m.shape)} that is not the causal pattern over its {rows} rows; it has no "
                               f"reading over the cached keys")
        self._chunk_masks[key] = True

    def _prefill_over_pages(self, q, k, v, k_for_cache, v_for_cache, expanded: bool, layer_idx: int,
                            dropout_p, scale, attn_mask=None):
        """A prefill chunk's attention over the session's keys gathered as WHOLE PAGES: the chunk's K/V appended to the
        pages, every cached key read back from them (the cache's dtype, widened exactly to the query's), the key length
        rounded up to whole pages, and the explicit 0/-inf mask (causal at the cached length, the keys past the true
        end masked). The existing attention then runs unchanged."""
        from nbxtensor.ops.attention_wrappers import scaled_dot_product_attention_wrapper
        start = self.cache.get_cache_len()          # the host length moves at the chunk's last layer
        rows = int(q.shape[2])
        self._require_causal_graph_mask(attn_mask, rows, layer_idx)
        self.cache.prefill(layer_idx, k_for_cache, v_for_cache, real=self._chunk_real)
        pt = self.cache.page_tokens
        keys = -(-(start + rows) // pt) * pt
        heads = int(q.shape[1]) if expanded else self._num_kv_heads
        kg = self.cache.gather_pages(layer_idx, "k", keys, heads, q._dtype)
        vg = self.cache.gather_pages(layer_idx, "v", keys, heads, q._dtype)
        mask = self._chunk_mask(int(q._device_idx), start, rows, keys)
        return scaled_dot_product_attention_wrapper(q, kg, vg, attn_mask=mask, dropout_p=dropout_p,
                                                    is_causal=False, scale=scale, k_pre_transposed=False)

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
        from nbxtensor.ops.attention_wrappers import scaled_dot_product_attention_wrapper

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

            if self._whole_pages:
                return self._prefill_over_pages(q, k, v, k_for_cache, v_for_cache,
                                                k_for_cache is not k, layer_idx, dropout_p, scale, attn_mask)

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
        s = float(scale) if scale is not None else 1.0 / math.sqrt(int(q.shape[-1]))
        return self.cache.decode(layer_idx, q, k, v, s, q_round=self.cache.dtype)

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
        self._chunk_start, self._chunk_real, self._whole_pages, self._chunk_masks = 0, None, False, {}

    def end_request(self) -> None:
        """The request ends and its positions stay in the cache (a serving session's, core/state/kv_session.py): the
        per-request state resets, the cache is kept."""
        self._is_prefill = True
        self._call_count = 0
        self._chunk_start, self._chunk_real, self._whole_pages, self._chunk_masks = 0, None, False, {}

    def set_decode_mode(self):
        """Mirror of core kv_cache_wrapper.set_decode_mode: the first decode step's positional offset is the
        cache's length after the prefill (it stayed 0 here: openaudio's first cached token on the Triton engine
        was rotated at position 0, 2026-10-19)."""
        self._is_prefill = False
        self._call_count = 0
        self._position_offset = self.get_cache_len()
        self._chunk_start, self._chunk_real, self._whole_pages, self._chunk_masks = 0, None, False, {}

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
        if self._is_prefill and self._chunk_start > 0 and end is not None:
            # a prefill chunk at a cached length: its rows' positions, read on the host (a prefill is never replayed)
            return table.narrow(int(dim), self._chunk_start + int(start), int(end - start))
        if self._is_prefill or self._position_offset <= 0 or end is None:
            return table.narrow(int(dim), int(start), int(end - start) if end is not None else table.shape[int(dim)] - int(start))
        pos = self._device_positions(int(start), int(end))
        if pos is not None:         # the rows gathered at the device's own length: a replayed step reads the live one
            from nbxtensor.ops import index_wrappers as _iw
            return _iw.index_select_wrapper(table, int(dim), pos)
        cache_len = self._position_offset
        return table.narrow(int(dim), cache_len + int(start), int(end - start))

    def _device_len(self) -> bool:
        lens = getattr(self.cache, "lens", None)
        return bool(lens) and lens[0] is not None and getattr(self.cache, "batch", 1) == 1

    def _device_positions(self, start: int, end: int):
        """[len + start, len + end) on the device, `len` the cache's device length of its first layer (before this step
        appends), or None without one: a launch reads the position instead of baking it, so a replayed step is exact."""
        if not self._device_len():
            return None
        lens = self.cache.lens
        from nbxtensor.ops.dispatch import dispatch
        from nbxtensor.ops import elementwise_wrappers as _ew
        from nbxtensor.dtype.nbx_dtype import NBXDtype
        ar = dispatch("arange")(start, end, dtype=NBXDtype.int64)
        return _ew.add(ar, lens[0].to(NBXDtype.int64))

    def intercept_position_cumsum(self, x, dim=0, dtype=None):
        """Mirror of core kv_cache_wrapper.intercept_position_cumsum: the positions a decoder counts
        over its own sequence, the cache's length added at decode. R33-pure: the cumsum and add
        wrappers of the kernel library."""
        from nbxtensor.ops import wrappers as W
        from nbxtensor.ops import elementwise_wrappers as _nbx_elementwise_wrappers  # the family the moved names live in (M2 split)
        from nbxtensor.ops import reduction_wrappers as _nbx_reduction_wrappers  # the family the moved names live in (M2 split)
        out = _nbx_reduction_wrappers.cumsum_wrapper(x, int(dim), dtype)
        if self._is_prefill and self._chunk_start > 0:
            return _nbx_elementwise_wrappers.add(out, self._chunk_start)   # a prefill chunk at a cached length
        if self._is_prefill or self._position_offset <= 0:
            return out
        if self._device_len():
            return _nbx_elementwise_wrappers.add(out, self.cache.lens[0].to(out.dtype))   # the device length, read by the launch
        return _nbx_elementwise_wrappers.add(out, self._position_offset)

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
        R33-pure: returns NBXTensor through the library's public aten::arange handler.
        """
        from nbxtensor.ops.dispatch import dispatch
        _create_arange = dispatch("arange")
        if self._is_prefill and self._chunk_start > 0 and args and all(isinstance(a, (int, float)) for a in args[:2]) \
                and len(args) <= 2:
            # a prefill chunk at a cached length: the window shifted by it on the host (a prefill is never replayed)
            c = self._chunk_start
            return (_create_arange(c, c + args[0], **kwargs) if len(args) == 1
                    else _create_arange(c + args[0], c + args[1], **kwargs))
        if self._is_prefill:
            return _create_arange(*args, **kwargs)
        cache_len = self._position_offset
        if (cache_len > 0 and args and all(isinstance(a, int) for a in args[:2])
                and len(args) <= 2 and not kwargs.get("step")):
            pos = self._device_positions(0, args[0]) if len(args) == 1 else self._device_positions(args[0], args[1])
            if pos is not None:
                dt = kwargs.get("dtype")
                return pos if dt is None or dt == pos.dtype else pos.to(dt)
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
        mask live). The position interceptors (arange, table slice, count) read the cache's device length
        (`_device_positions`), so a replayed step is at its true position; a cache without a device length bakes the
        host scalar and refuses (whisper-large's decoder: its table slice baked one step's rows, the plan was rejected at
        verify; read on the device it replays, 216 -> 57 ms a token on a V100, the same transcript, 2026-11-06)."""
        import os
        position_funcs = (TritonAttentionInterceptor.intercept_arange, TritonAttentionInterceptor.intercept_position_slice,
                          TritonAttentionInterceptor.intercept_position_cumsum)
        if self._is_prefill:
            return None
        if any(getattr(f, "__func__", None) in position_funcs for f in registered_funcs) and not self._device_len():
            return None       # positions only on the host: a recorded step would bake one
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
