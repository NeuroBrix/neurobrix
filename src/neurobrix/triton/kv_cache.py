"""Triton KV Cache — pre-allocated NBXTensor buffers.

Zero torch dependency. Uses NBXTensor for allocation, cudaMemcpy for
indexed writes, and the Triton Flash Attention wrapper for SDPA.
"""

from typing import Dict

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator


class KVCacheLayer:
    """Single layer KV cache using NBXTensor buffers."""

    __slots__ = ('k_buffer', 'v_buffer', 'current_len', '_buffer_len',
                 'max_len', 'num_kv_heads', 'k_head_dim', 'v_head_dim',
                 'batch_size', 'dtype', 'device_idx', 'mask_buffer',
                 'pos_counter', '_synced')

    def __init__(self, device_idx: int, dtype: NBXDtype, max_len: int,
                 num_kv_heads: int, k_head_dim: int, v_head_dim: int,
                 batch_size: int = 1):
        self.device_idx = device_idx
        self.dtype = dtype
        self.max_len = max_len
        self.num_kv_heads = num_kv_heads
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.batch_size = batch_size
        self.current_len = 0
        self._buffer_len = max_len

        dev = f"cuda:{device_idx}"
        self.k_buffer = NBXTensor.zeros(
            (batch_size, num_kv_heads, max_len, k_head_dim), dtype=dtype, device=dev)
        self.v_buffer = NBXTensor.zeros(
            (batch_size, num_kv_heads, max_len, v_head_dim), dtype=dtype, device=dev)
        # B2' (bucket-padded decode views): persistent additive length
        # mask [1, 1, 1, max_len] — 0 for valid positions, -inf beyond.
        # Fixed address for the plan's lifetime; extending validity is
        # ONE tiny write per appended position (never a per-step
        # allocation). Lazy: built on first bucketed update.
        self.mask_buffer = None
        # B3: per-layer device position counter (int32 [1]) — the
        # kv_append kernel reads its write position from it, making the
        # append launch tuple CONSTANT across steps. Host current_len
        # stays the control-flow authority (bucket math, view sizes).
        self.pos_counter = None
        # Device state (mask validity, pos counter value) in sync with
        # the host authority. clear() flips this instead of dropping
        # the buffers: their ADDRESSES must survive requests (recorded
        # replay tuples point at them), only their CONTENTS reset.
        self._synced = False

    def update(self, k, v):
        """Write new K/V to buffer, return view of all cached values.

        Args:
            k: [batch, kv_heads, new_len, head_dim]
            v: [batch, kv_heads, new_len, v_dim]

        Returns:
            (k_cached, v_cached): views of buffer[:, :, :current_len+new_len, :]
        """
        new_len = k.shape[2]
        end_pos = self.current_len + new_len

        if end_pos > self._buffer_len:
            raise RuntimeError(
                f"KV cache overflow: {self.current_len}+{new_len} > {self._buffer_len}")

        # Indexed write via __setitem__ (cudaMemcpy)
        self.k_buffer[:self.batch_size, :, self.current_len:end_pos, :] = k
        self.v_buffer[:self.batch_size, :, self.current_len:end_pos, :] = v

        self.current_len = end_pos

        # Return views
        k_cached = self.k_buffer[:self.batch_size, :, :self.current_len, :]
        v_cached = self.v_buffer[:self.batch_size, :, :self.current_len, :]
        return k_cached, v_cached

    def update_bucketed(self, k, v, bucket: int):
        """B2' decode update: append k/v, then return views padded to
        the LENGTH BUCKET plus the additive mask view excluding the pad.

        The padding never changes results, only memory geometry: the
        region beyond current_len is excluded by the -inf mask (never
        by content — stale K there must not matter), and every
        downstream allocation sized from these views is constant
        within a bucket (the address-stability prerequisite measured
        in the B1 gate).

        Returns (k_view, v_view, mask_view, padded_len)."""
        import numpy as np
        new_len = k.shape[2]
        if self.mask_buffer is None or not self._synced:
            # First bucketed call of a request: validate EVERYTHING
            # cached so far (the prefill populated [0:current_len)
            # through the plain update path) — only positions beyond
            # current_len stay masked. Missing this masked out the
            # whole prompt. On a warm re-request (clear() flipped
            # _synced) the SAME buffers are rewritten IN PLACE — their
            # addresses are part of recorded replay plans.
            from neurobrix.triton.dtype import numpy_staging_dtype
            neg = np.full((1, 1, 1, self._buffer_len), -np.inf,
                          dtype=numpy_staging_dtype(self.dtype))
            neg[:, :, :, :self.current_len] = 0.0
            DeviceAllocator.set_device(self.device_idx)
            staged = NBXTensor.from_numpy(neg)
            if staged.nbx_dtype != self.dtype:
                staged = staged.to(self.dtype)  # bf16: no numpy repr
            if self.mask_buffer is None:
                self.mask_buffer = staged
            else:
                self.mask_buffer[:, :, :, :] = staged
            if self.pos_counter is not None:
                self._write_pos(self.current_len)
            self._synced = True

        if new_len == 1 and self.batch_size == 1:
            # B3 constant-tuple append: k/v/mask written at the DEVICE
            # counter's position; the counter advances in a second
            # stream-ordered launch. Host current_len mirrors for
            # control flow only.
            import triton as _tr
            from neurobrix.kernels.ops.kv_append import (
                kv_append_kernel, kv_pos_inc_kernel)
            if self.pos_counter is None:
                DeviceAllocator.set_device(self.device_idx)
                self.pos_counter = NBXTensor.from_numpy(
                    np.array([self.current_len], dtype=np.int32))
            if self.current_len + 1 > self._buffer_len:
                raise RuntimeError(
                    f"KV cache overflow: {self.current_len}+1 > "
                    f"{self._buffer_len}")
            ks = k.contiguous()
            vs = v.contiguous()
            block = max(_tr.next_power_of_2(max(self.k_head_dim,
                                                self.v_head_dim)), 16)
            from neurobrix.kernels.nbx_tensor import _set_device
            _set_device(self.k_buffer)
            kv_append_kernel[(self.num_kv_heads,)](
                ks, vs, self.k_buffer, self.v_buffer,
                self.mask_buffer, self.pos_counter,
                self.k_buffer.stride(1), self.k_buffer.stride(2),
                self.v_buffer.stride(1), self.v_buffer.stride(2),
                D_K=self.k_head_dim, D_V=self.v_head_dim,
                BLOCK=block, num_warps=1)
            kv_pos_inc_kernel[(1,)](self.pos_counter)
            self.current_len += 1
        else:
            # Prefill / multi-token / batched append: the plain setitem
            # path + host mask-validity write (not on the per-step hot
            # loop; batch>1 is outside the append kernel's contract).
            new_from = self.current_len
            self.update(k, v)
            from neurobrix.triton.dtype import numpy_staging_dtype
            zeros = np.zeros(
                (1, 1, 1, self.current_len - new_from),
                dtype=numpy_staging_dtype(self.dtype))
            staged = NBXTensor.from_numpy(zeros)
            if staged.nbx_dtype != self.dtype:
                staged = staged.to(self.dtype)  # bf16: no numpy repr
            self.mask_buffer[:, :, :, new_from:self.current_len] = staged
            if self.pos_counter is not None:
                # Keep the device counter mirroring the host authority
                # even off the kernel path.
                self._write_pos(self.current_len)

        padded = min(((self.current_len + bucket - 1) // bucket) * bucket,
                     self._buffer_len)
        k_view = self.k_buffer[:self.batch_size, :, :padded, :]
        v_view = self.v_buffer[:self.batch_size, :, :padded, :]
        mask_view = self.mask_buffer[:, :, :, :padded]
        return k_view, v_view, mask_view, padded

    def _write_pos(self, value: int):
        """Overwrite the device position counter IN PLACE (4-byte D2D;
        the counter's address never changes — recorded replay tuples
        point at it). Used at request-boundary resync and by the
        verify-first restore hook (the counter increment is the append
        step's only non-idempotent device write)."""
        import numpy as np
        DeviceAllocator.set_device(self.device_idx)
        src = NBXTensor.from_numpy(np.array([value], dtype=np.int32))
        DeviceAllocator.memcpy(self.pos_counter.data_ptr(),
                               src.data_ptr(), 4, 3)

    def clear(self):
        self.current_len = 0
        # Device state (mask validity, position counter) now stale —
        # resynced IN PLACE at the next bucketed call so buffer
        # addresses survive the request boundary (replay plans record
        # them). Dropping the buffers here would silently invalidate
        # every recorded plan on the second warm request.
        self._synced = False


class TritonKVCache:
    """Distributed KV cache with per-layer lazy allocation.

    Prompt-aware sizing (S1 finding TINYLLAMA-KVCAP, fixed 2026-09-01):
    the planned `max_cache_len` (Prism run-mode plan or the legacy
    `max_tokens + prompt_margin` fallback) assumes the prompt fits its
    margin. Layers allocate LAZILY at their first update — which IS the
    prefill, where the true prompt length is known — so the allocation
    sizes itself there: when `prompt + decode_budget` exceeds the plan
    size, the buffer allocates at that need, bounded by the model's own
    context window (`window_ceiling`, from lm_config, data-driven).
    Short prompts allocate the plan's initial size (serve plans) or the
    plan size (run mode), byte-unchanged. Buffers are never resized in
    place: a warm layer that cannot hold a later request is REPLACED at
    that request's prefill, the cache generation bumps and the replay
    plans recorded under the previous generation are retired (the
    recorded-replay contract — addresses baked into plans — is kept by
    invalidation, not by pinning). A prompt at or beyond the window
    raises the loud overflow: that is the model's true limit, not a
    budget."""

    import itertools as _itertools
    _uids = _itertools.count(1)

    def __init__(self, num_layers: int, num_kv_heads: int, k_head_dim: int,
                 v_head_dim: int, max_cache_len: int, dtype: NBXDtype,
                 window_ceiling: int = 0, decode_budget: int = 0,
                 initial_cache_len: int = 0):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.max_cache_len = max_cache_len
        self.dtype = dtype
        self.window_ceiling = int(window_ceiling or 0)
        self.decode_budget = int(decode_budget or 0)
        # Prism's `initial_cache_len` (serve-mode plans: max_tokens +
        # margin; 0 = the plan size, the run-mode legacy). The compiled
        # cache always honoured it (initial allocation + growth toward
        # the ceiling); this cache allocated the ceiling up front — in
        # serve the ceiling is the whole remaining VRAM, so the first
        # request pinned 10.8 GB of KV for a 30B MoE on a 32 GB card and
        # every later prefill ran at the memory edge (190 attention-chunk
        # halvings, 480 pool flushes, 46 s instead of the CLI's 24 s at
        # 8.3k context — measured 2026-09-03). R30 mirror of the compiled
        # `StateCacheFactory` semantics: initial size, grow on need.
        self.initial_cache_len = int(initial_cache_len or 0)
        self.regrowths = 0   # layers replaced for a longer request (diagnostic)
        # Replay identity: the bucket signature carries (uid, generation);
        # a buffer replacement bumps the generation ONCE per request and
        # retires every plan recorded under the previous one (the plans
        # bake buffer addresses, strides and captured graphs).
        self._uid = next(TritonKVCache._uids)
        self.generation = 0
        self._generation_bumped = False
        self._layers: Dict[int, KVCacheLayer] = {}

    def _alloc_len_for(self, first_len: int) -> int:
        """Buffer length for a layer whose FIRST update carries
        `first_len` positions (the prefill). Plan size when it fits;
        prompt + decode budget when it does not; the model window is
        the ceiling."""
        alloc = self.initial_cache_len or self.max_cache_len
        needed = first_len + self.decode_budget if self.decode_budget \
            else first_len + 1
        if needed > alloc:
            # Grow to the need: up to the plan ceiling first, then (the
            # 2026-09-01 rule) up to the model window when the plan
            # itself cannot hold the prompt.
            grown = needed if not self.window_ceiling \
                else min(needed, self.window_ceiling)
            if grown > alloc:
                alloc = grown
        if self.window_ceiling and first_len >= self.window_ceiling:
            raise RuntimeError(
                f"Prompt length {first_len} reaches the model's context "
                f"window ({self.window_ceiling} positions) — this is the "
                f"model's own limit, not an engine budget.")
        return alloc

    def _get_layer(self, layer_idx: int, k) -> KVCacheLayer:
        if layer_idx in self._layers:
            layer = self._layers[layer_idx]
            if layer.current_len == 0:
                # First update of a request (the prefill) on a WARM layer:
                # a request the buffers cannot hold is served by REPLACING
                # the layer's buffers (fresh allocation by the ONE sizing
                # rule, `_alloc_len_for` — which raises the loud window
                # limit before any side effect), never by growing them in
                # place under a live view. Addresses change: the cache
                # generation is bumped once per request and every replay
                # plan recorded under the previous generation is retired
                # (its bucket signature carries the generation, so a
                # surviving plan could never match; the retirement frees
                # the slabs and captured graphs it pinned).
                first_len = int(k.shape[2])
                if self._alloc_len_for(first_len) > layer._buffer_len:
                    if not self._generation_bumped:
                        old_gen = self.generation
                        self.generation += 1
                        self._generation_bumped = True
                        from neurobrix.triton import replay as _replay
                        uid = self._uid
                        _replay.drop_plans_by_contribution(
                            lambda c: (isinstance(c, tuple) and len(c) >= 5
                                       and c[0] == "kv_decode"
                                       and c[3] == uid and c[4] == old_gen))
                    del self._layers[layer_idx]
                    del layer
                    self.regrowths += 1
        if layer_idx not in self._layers:
            device_idx = k._device_idx if hasattr(k, '_device_idx') else 0
            batch_size = k.shape[0]
            self._layers[layer_idx] = KVCacheLayer(
                device_idx=device_idx, dtype=self.dtype,
                max_len=self._alloc_len_for(int(k.shape[2])),
                num_kv_heads=self.num_kv_heads,
                k_head_dim=self.k_head_dim,
                v_head_dim=self.v_head_dim,
                batch_size=batch_size)
        return self._layers[layer_idx]

    def update(self, layer_idx: int, k, v):
        """Update cache for a layer. Lazy allocation on first call."""
        return self._get_layer(layer_idx, k).update(k, v)

    def update_bucketed(self, layer_idx: int, k, v, bucket: int):
        """B2' decode update: bucket-padded views + additive pad mask
        (see KVCacheLayer.update_bucketed). Lazy allocation as above."""
        return self._get_layer(layer_idx, k).update_bucketed(k, v, bucket)

    def clear(self):
        self._generation_bumped = False
        for layer in self._layers.values():
            layer.clear()


class TritonAttentionInterceptor:
    """Intercepts SDPA for KV cache + calls Triton Flash Attention."""

    def __init__(self, cache: TritonKVCache, num_heads: int = 0):
        self.cache = cache
        self._is_prefill = True
        self._call_count = 0
        self._num_heads = num_heads
        self._num_kv_heads = cache.num_kv_heads
        self._gqa_group_size = 0
        # Absolute decode position for RoPE arange shifting (see intercept_arange).
        self._position_offset = 0

    def intercept(self, q, k, v, attn_mask=None, dropout_p=0.0,
                  is_causal=True, scale=None, layer_idx=-1):
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
        if (k.ndim == 4
                and k.shape[2] == q.shape[3]
                and k.shape[3] == q.shape[2]
                and k.shape[2] != q.shape[2]):
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

            # Store in cache for decode
            self.cache.update(layer_idx, k_for_cache, v_for_cache)

            # Run SDPA with full Q/K/V (graph-expanded, not cache)
            if hasattr(q, '_dtype') and hasattr(k, '_dtype') and q._dtype != k._dtype:
                k = k.to(q._dtype)
                v = v.to(q._dtype)
            return scaled_dot_product_attention_wrapper(
                q, k, v, attn_mask=None, dropout_p=dropout_p,
                is_causal=True, scale=scale)

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

        # Update cache. B2' (NBX_REPLAY_KV_DECODE): bucket-padded views
        # + persistent additive pad mask — padding never changes the
        # result, only the memory geometry (downstream allocation sizes
        # become bucket-constant, the address-stability prerequisite).
        import os as _os_b2
        _bucket = 0
        if _os_b2.environ.get("NBX_REPLAY_KV_DECODE") == "1":
            _bucket = int(_os_b2.environ.get("NBX_KV_BUCKET", "256"))
        if _bucket:
            k_full, v_full, _pad_mask, _ = self.cache.update_bucketed(
                layer_idx, k, v, _bucket)
            if attn_mask is not None and \
                    attn_mask.shape[-1] != k_full.shape[2]:
                attn_mask = None
            attn_mask = _pad_mask if attn_mask is None else attn_mask
        else:
            k_full, v_full = self.cache.update(layer_idx, k, v)

        # GQA: DO NOT re-expand.
        #
        # This used to broadcast K and V up to `num_heads` and
        # materialise the result, so that downstream code saw an MHA
        # shape. At a 4,164-token context that copied 34 MB per call,
        # 4.9 GB per decode step for K and as much again for V — 9.8 GB
        # of pure data movement to generate ONE token, and the dominant
        # term in our context scaling (`strided_copy` fitted at 11.9 us
        # per context token, against the real attention matmul's 3.8).
        #
        # Nothing downstream needs it. The SDPA wrapper already handles
        # H != H_kv natively — `gqa_groups = nheads // nheads_k` with a
        # head mapping inside the kernel, its own comment recording that
        # "K/V keep their native (b, nheads_k, s, d) layout ... zero
        # cost" — and `_math_attention`, the path decode takes, now
        # groups Q's heads inside each KV head instead of broadcasting
        # K/V, which is a pure view.
        #
        # So K and V travel in their native cache layout, and the
        # expansion that existed only to flatten a shape is gone.

        # Causal masking
        use_causal = is_causal if self._is_prefill else False

        # Drop stale mask
        if attn_mask is not None:
            kv_seq = k_full.shape[2]
            if attn_mask.shape[-1] != kv_seq:
                attn_mask = None

        # Cast Q to cache dtype — Flash Attention works in fp16/bf16, not fp32.
        # AMP may upcast Q to fp32 but the kernel handles precision internally.
        if hasattr(q, '_dtype') and hasattr(k_full, '_dtype') and q._dtype != k_full._dtype:
            q = q.to(k_full._dtype)

        return scaled_dot_product_attention_wrapper(
            q, k_full, v_full,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=use_causal,
            scale=scale)

    def intercept_efficient(self, q, k, v, attn_bias=None, compute_log_sumexp=False,
                            dropout_p=0.0, is_causal=False, scale=None,
                            layer_idx=-1, **kwargs):
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
                              is_causal=is_causal, scale=scale, layer_idx=layer_idx)

    def intercept_flash(self, q, k, v, dropout_p=0.0, is_causal=False,
                        return_debug_mask=False, scale=None, layer_idx=-1, **kwargs):
        """aten::_scaled_dot_product_flash_attention — is_causal at arg[4], no
        attn_bias, ``scale`` kwarg-only. Remap to intercept()."""
        return self.intercept(q, k, v, attn_mask=None, dropout_p=dropout_p,
                              is_causal=is_causal, scale=scale, layer_idx=layer_idx)

    def reset(self):
        self.cache.clear()
        self._is_prefill = True
        self._call_count = 0

    def set_decode_mode(self):
        self._is_prefill = False
        self._call_count = 0

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
        """Return current cache length from first layer."""
        for layer in self.cache._layers.values():
            return layer.current_len
        return 0

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
        if os.environ.get("NBX_REPLAY_KV_DECODE") != "1":
            return None
        bucket = int(os.environ.get("NBX_KV_BUCKET", "256"))
        if not self.cache._layers:
            return None
        for layer in self.cache._layers.values():
            if (layer.batch_size != 1 or layer.mask_buffer is None
                    or layer.pos_counter is None or not layer._synced):
                return None
        length = self.get_cache_len()
        buffer_len = next(iter(self.cache._layers.values()))._buffer_len
        # The bucket THIS step will compute in: the append advances
        # current_len before the padded views are taken.
        padded = min(((length + 1 + bucket - 1) // bucket) * bucket,
                     buffer_len)
        # Buffer identity in the bucket key: a layer replacement (longer
        # request on a warm cache) changes the generation, so no plan
        # recorded against the old buffers can ever match again.
        return ("kv_decode", bucket, padded, self.cache._uid, self.cache.generation)

    def replay_advance(self):
        """The replayed launches appended one position per layer and
        advanced every device counter; mirror on the host authority
        (position_ids and the bucket signature read current_len)."""
        for layer in self.cache._layers.values():
            layer.current_len += 1

    def replay_restore(self):
        """Rewind the device counters to the host authority so the
        verify pass's normal re-run appends at the same position the
        replayed pass wrote (identical inputs -> idempotent k/v/mask
        writes; only the counter increment needs undoing)."""
        for layer in self.cache._layers.values():
            if layer.pos_counter is not None:
                layer._write_pos(layer.current_len)


# Phase 1 opt-in cleanup: mark the interceptor's underlying function
# as self-managing dtype. TritonSequence._compile_op walks
# func.__func__ for bound methods to pick up this flag.
TritonAttentionInterceptor.intercept.self_manages_dtype = True  # type: ignore[attr-defined]
TritonAttentionInterceptor.intercept_efficient.self_manages_dtype = True  # type: ignore[attr-defined]
TritonAttentionInterceptor.intercept_flash.self_manages_dtype = True  # type: ignore[attr-defined]
