"""Triton KV Cache — pre-allocated NBXTensor buffers.

Zero torch dependency. Uses NBXTensor for allocation, cudaMemcpy for
indexed writes, and the Triton Flash Attention wrapper for SDPA.
"""

from typing import Dict

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype


class KVCacheLayer:
    """Single layer KV cache using NBXTensor buffers."""

    __slots__ = ('k_buffer', 'v_buffer', 'current_len', '_buffer_len',
                 'max_len', 'num_kv_heads', 'k_head_dim', 'v_head_dim',
                 'batch_size', 'dtype', 'device_idx')

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

    def clear(self):
        self.current_len = 0


class TritonKVCache:
    """Distributed KV cache with per-layer lazy allocation."""

    def __init__(self, num_layers: int, num_kv_heads: int, k_head_dim: int,
                 v_head_dim: int, max_cache_len: int, dtype: NBXDtype):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.max_cache_len = max_cache_len
        self.dtype = dtype
        self._layers: Dict[int, KVCacheLayer] = {}

    def update(self, layer_idx: int, k, v):
        """Update cache for a layer. Lazy allocation on first call."""
        if layer_idx not in self._layers:
            device_idx = k._device_idx if hasattr(k, '_device_idx') else 0
            batch_size = k.shape[0]
            self._layers[layer_idx] = KVCacheLayer(
                device_idx=device_idx, dtype=self.dtype,
                max_len=self.max_cache_len,
                num_kv_heads=self.num_kv_heads,
                k_head_dim=self.k_head_dim,
                v_head_dim=self.v_head_dim,
                batch_size=batch_size)

        return self._layers[layer_idx].update(k, v)

    def clear(self):
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

        # Update cache
        k_full, v_full = self.cache.update(layer_idx, k, v)

        # GQA: re-expand
        if self._gqa_group_size > 1:
            batch, kv_heads, cached_len, k_dim = k_full.shape
            k_full = k_full.unsqueeze(2).expand(-1, -1, self._gqa_group_size, -1, -1)
            k_full = k_full.reshape(batch, self._num_heads, cached_len, k_dim).contiguous()
            v_dim = v_full.shape[-1]
            v_full = v_full.unsqueeze(2).expand(-1, -1, self._gqa_group_size, -1, -1)
            v_full = v_full.reshape(batch, self._num_heads, cached_len, v_dim).contiguous()

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
        """Called before each decode step. Resets per-step call counter."""
        self._call_count = 0

    def get_cache_len(self) -> int:
        """Return current cache length from first layer."""
        for layer in self.cache._layers.values():
            return layer.current_len
        return 0


# Phase 1 opt-in cleanup: mark the interceptor's underlying function
# as self-managing dtype. TritonSequence._compile_op walks
# func.__func__ for bound methods to pick up this flag.
TritonAttentionInterceptor.intercept.self_manages_dtype = True  # type: ignore[attr-defined]
TritonAttentionInterceptor.intercept_efficient.self_manages_dtype = True  # type: ignore[attr-defined]
TritonAttentionInterceptor.intercept_flash.self_manages_dtype = True  # type: ignore[attr-defined]
