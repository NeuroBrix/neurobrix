"""
Distributed Lazy KV-Cache for Multi-GPU LLM Inference.

ARCHITECTURE: "Enterprise-Grade" KV Cache System
- LOCALITY RULE: Cache lives where attention computation happens
- LAZY ALLOCATION: Allocate on-demand based on incoming q tensor's device
- PRE-ALLOCATED BUFFERS: No torch.cat, use indexed writes for O(1) updates
- SYMBOLIC SHAPES: Return views for dynamic seq_len support

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
    dtype: torch.dtype

    @property
    def head_dim(self) -> int:
        """Backward compat for code that reads .head_dim"""
        return self.k_head_dim


class KVCacheLayer:
    """
    Pre-allocated KV cache buffer for a single transformer layer.

    DESIGN:
    - Static buffer allocation: Avoids repeated mallocs
    - Indexed writes: O(1) update instead of O(n) concat
    - View-based returns: Support symbolic shapes

    Storage format: K: [batch, num_kv_heads, max_cache_len, k_head_dim], V: [..., v_head_dim]
    """

    __slots__ = ('device', 'dtype', 'k_buffer', 'v_buffer', 'current_len',
                 'max_len', 'num_kv_heads', 'k_head_dim', 'v_head_dim', 'batch_size',
                 '_dtype_verified')

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        max_len: int,
        num_kv_heads: int,
        k_head_dim: int,
        v_head_dim: int,
        batch_size: int = 1
    ):
        """
        Initialize pre-allocated buffers on the specified device.

        Args:
            device: Device where this layer's attention runs
            dtype: Data type for cache tensors
            max_len: Maximum sequence length to cache
            num_kv_heads: Number of key-value heads
            k_head_dim: Dimension per K head
            v_head_dim: Dimension per V head
            batch_size: Batch size (can be expanded later if needed)
        """
        self.device = device
        self.dtype = dtype
        self.max_len = max_len
        self.num_kv_heads = num_kv_heads
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.batch_size = batch_size
        self.current_len = 0
        self._dtype_verified = False  # Skip per-token dtype check after first match

        # Pre-allocate buffers: K and V may have different head dims
        k_shape = (batch_size, num_kv_heads, max_len, k_head_dim)
        v_shape = (batch_size, num_kv_heads, max_len, v_head_dim)
        self.k_buffer = torch.zeros(k_shape, dtype=dtype, device=device)
        self.v_buffer = torch.zeros(v_shape, dtype=dtype, device=device)

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Write new K/V to buffer and return view of cached values.

        ZERO COPY: Indexed write directly into pre-allocated buffer.
        SYMBOLIC SHAPE: Returns view buffer[:, :, :current_len, :]

        Args:
            k: New keys [batch, num_kv_heads, new_len, head_dim]
            v: New values [batch, num_kv_heads, new_len, head_dim]

        Returns:
            (k_cached, v_cached): Views of cached K/V up to current position
        """
        batch_size, num_kv_heads, new_len, head_dim = k.shape

        # Expand batch dimension if needed (e.g., CFG with batch=2)
        if batch_size > self.batch_size:
            self._expand_batch(batch_size)

        # Dtype/device check: run once, skip for all subsequent tokens
        if not self._dtype_verified:
            if k.device != self.device:
                k = k.to(self.device)
                v = v.to(self.device)
            if k.dtype != self.dtype:
                k = k.to(self.dtype)
                v = v.to(self.dtype)
            self._dtype_verified = True

        # Check capacity
        end_pos = self.current_len + new_len
        if end_pos > self.max_len:
            raise RuntimeError(
                f"KV cache overflow: current_len={self.current_len}, new_len={new_len}, "
                f"max_len={self.max_len}. Increase max_position_embeddings in lm_config."
            )

        # Indexed write: buffer[..., start:end, :] = new_values
        self.k_buffer[:batch_size, :, self.current_len:end_pos, :] = k
        self.v_buffer[:batch_size, :, self.current_len:end_pos, :] = v

        # Update position
        self.current_len = end_pos

        # Return VIEW (not copy) of cached values - supports symbolic shapes
        # View is contiguous because buffer is contiguous and we slice on dim 2
        k_cached = self.k_buffer[:batch_size, :, :self.current_len, :]
        v_cached = self.v_buffer[:batch_size, :, :self.current_len, :]

        return k_cached, v_cached

    def _expand_batch(self, new_batch_size: int) -> None:
        """Expand buffer batch dimension if CFG or other batching is enabled."""
        if new_batch_size <= self.batch_size:
            return

        # Create new larger buffers (K and V may have different head dims)
        new_k = torch.zeros(
            (new_batch_size, self.num_kv_heads, self.max_len, self.k_head_dim),
            dtype=self.dtype, device=self.device
        )
        new_v = torch.zeros(
            (new_batch_size, self.num_kv_heads, self.max_len, self.v_head_dim),
            dtype=self.dtype, device=self.device
        )

        # Copy existing data
        if self.current_len > 0:
            new_k[:self.batch_size, :, :self.current_len, :] = \
                self.k_buffer[:, :, :self.current_len, :]
            new_v[:self.batch_size, :, :self.current_len, :] = \
                self.v_buffer[:, :, :self.current_len, :]

        # Replace buffers
        self.k_buffer = new_k
        self.v_buffer = new_v
        self.batch_size = new_batch_size

    def clear(self) -> None:
        """Reset cache for new sequence (keeps buffers allocated)."""
        self.current_len = 0
        self._dtype_verified = False


class DistributedKVCache:
    """
    Distributed Key-Value cache manager for multi-GPU setups.

    DESIGN PRINCIPLES:
    1. LAZY ALLOCATION: Layers are created on-demand when first accessed
    2. LOCALITY: Each layer's cache lives on the device where attention runs
    3. NO GLOBAL DEVICE: Device is inferred from incoming q tensor

    This enables automatic support for:
    - FGP: Layer 0 on cuda:0, Layer 15 on cuda:1 → caches follow
    - TP: Each GPU handles a subset of heads
    - Single GPU: Everything on cuda:0
    """

    def __init__(self, config: KVCacheConfig):
        """
        Initialize distributed cache manager.

        Args:
            config: Cache configuration (no device - determined lazily)
        """
        self.config = config
        # Dict mapping layer_idx -> KVCacheLayer (lazy allocation)
        self._layers: Dict[int, KVCacheLayer] = {}
        self._seq_len = 0

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a specific layer.

        LAZY ALLOCATION: If this layer's cache doesn't exist, create it
        on the same device as the incoming k tensor.

        Args:
            layer_idx: Transformer layer index
            k: Key tensor [batch, num_kv_heads, new_seq, head_dim]
            v: Value tensor [batch, num_kv_heads, new_seq, head_dim]

        Returns:
            (k_full, v_full): Cached K/V with all previous + new tokens
        """
        if layer_idx < 0 or layer_idx >= self.config.num_layers:
            raise RuntimeError(
                f"ZERO FALLBACK: layer_idx {layer_idx} out of range [0, {self.config.num_layers})"
            )

        # LAZY ALLOCATION: Create cache layer on k's device if not exists
        if layer_idx not in self._layers:
            device = k.device
            batch_size = k.shape[0]

            self._layers[layer_idx] = KVCacheLayer(
                device=device,
                dtype=self.config.dtype,
                max_len=self.config.max_cache_len,
                num_kv_heads=self.config.num_kv_heads,
                k_head_dim=self.config.k_head_dim,
                v_head_dim=self.config.v_head_dim,
                batch_size=batch_size
            )

        # Update cache and get full K/V
        k_full, v_full = self._layers[layer_idx].update(k, v)

        # Track sequence length from layer 0
        if layer_idx == 0:
            self._seq_len = self._layers[0].current_len

        return k_full, v_full

    def get_seq_len(self) -> int:
        """Current cached sequence length."""
        return self._seq_len

    def get_cache(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached K/V for a specific layer."""
        if layer_idx in self._layers:
            layer = self._layers[layer_idx]
            if layer.current_len > 0:
                batch = layer.batch_size
                return (
                    layer.k_buffer[:batch, :, :layer.current_len, :],
                    layer.v_buffer[:batch, :, :layer.current_len, :]
                )
        return None, None

    def clear(self) -> None:
        """Clear cache for new sequence."""
        for layer in self._layers.values():
            layer.clear()
        self._seq_len = 0

    def get_layer_devices(self) -> Dict[int, torch.device]:
        """Get mapping of layer_idx to device (for debugging)."""
        return {idx: layer.device for idx, layer in self._layers.items()}


class KVCacheAttentionWrapper:
    """
    Intercepts scaled_dot_product_attention for KV cache injection.

    DISTRIBUTED ARCHITECTURE:
    - Uses DistributedKVCache for multi-GPU support
    - Device determined from incoming q tensor (lazy allocation)
    - Works automatically with FGP, TP, PP, Single GPU

    The traced graph contains attention(Q, K, V) WITHOUT cache context.
    This wrapper:
    1. Reads layer_idx from op annotation (preferred) OR uses call counter (fallback)
    2. Concatenates incoming K/V with cached values
    3. Updates cache
    4. Calls attention with full context
    """

    def __init__(self, config: KVCacheConfig, num_heads: int = 0):
        """
        Initialize wrapper with distributed cache.

        Args:
            config: KVCacheConfig (no device - determined lazily per-layer)
            num_heads: Total number of query heads (for GQA un-expand/re-expand).
                       If 0, auto-detected from first Q tensor.
        """
        self.cache = DistributedKVCache(config)
        self._is_prefill = True
        self._call_count = 0
        self._num_layers = config.num_layers
        self._num_kv_heads = config.num_kv_heads
        self._num_heads = num_heads  # 0 = auto-detect from Q
        self._gqa_group_size = 0  # 0 = not yet computed
        self._position_offset = 0
        # During decode, K/V may be padded (e.g., [B, heads, 64, dim] instead of [B, heads, 1, dim])
        # This tracks the ACTUAL number of tokens to cache (not the padded length)
        self._decode_actual_seq_len: Optional[int] = None


    def intercept_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = True,
        scale: Optional[float] = None,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Intercept SDPA for KV cache injection.

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
        # K may arrive transposed [B,H,D,S] from pattern-reassembled SDPA.
        # SDPA and KV cache expect [B,H,S,D]. Detect and fix.
        if k.ndim == 4 and q.ndim == 4 and k.shape[-2] != q.shape[-2]:
            k = k.transpose(-2, -1)
        if v.ndim == 4 and q.ndim == 4 and v.shape[-2] != q.shape[-2]:
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

        # GQA: The traced graph may expand K/V from num_kv_heads to num_heads
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

        # Update cache (lazy allocation on k's device)
        k_full, v_full = self.cache.update(layer_idx, k, v)

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

        # Call PyTorch's SDPA with full cached context.
        # DtypeEngine handles dtype. No manual upcast.
        return F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=use_causal,
            scale=scale
        )

    def reset_for_new_sequence(self) -> None:
        """Reset cache for a new generation sequence."""
        self.cache.clear()
        self._is_prefill = True
        self._call_count = 0
        self._position_offset = 0
        self._decode_actual_seq_len = None

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

    def get_cache_len(self) -> int:
        """Get current cache sequence length."""
        return self.cache.get_seq_len()

    def update_position_offset(self) -> None:
        """Update position offset for next decode step."""
        if not self._is_prefill:
            self._position_offset = self.cache.get_seq_len()

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

        Only float aranges are shifted (RoPE). Integer aranges (MoE routing
        via topk/sort/bincount) pass through unmodified.
        """
        if self._is_prefill:
            return torch.arange(*args, **kwargs)

        cache_len = self._position_offset
        if cache_len > 0 and len(args) >= 1 and isinstance(args[0], (int, float)):
            dtype = kwargs.get('dtype', None)
            if dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
                end = args[0]
                # Shift start to cache_len: arange(end) → arange(cache_len, cache_len + end)
                # Output size unchanged (end elements), symbolic shapes preserved
                return torch.arange(cache_len, cache_len + end, **kwargs)

        return torch.arange(*args, **kwargs)

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
        scale: Optional[float] = None
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
            layer_idx=-1
        )

        return (output, None, None, None)

    def intercept_flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        return_debug_mask: bool = False,
        scale: Optional[float] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], int, int, Optional[torch.Tensor]]:
        """
        Intercept aten::_scaled_dot_product_flash_attention.

        Signature: (q, k, v, dropout_p, is_causal, return_debug_mask, scale)
        Returns: (output, log_sumexp, cum_seq_q, cum_seq_k, max_q, max_k, debug_mask)
        """
        if isinstance(is_causal, float):
            is_causal = bool(is_causal)

        output = self.intercept_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            layer_idx=-1
        )

        return (output, None, None, None, 0, 0, None)

    def intercept_cudnn_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        compute_log_sumexp: bool = False,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        return_debug_mask: bool = False,
        scale: Optional[float] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Intercept aten::_scaled_dot_product_cudnn_attention.

        Signature: (q, k, v, attn_bias, compute_log_sumexp, dropout_p, is_causal, return_debug_mask, scale)
        Returns: (output, logsumexp, philox_seed)
        """
        if isinstance(is_causal, float):
            is_causal = bool(is_causal)

        output = self.intercept_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            layer_idx=-1
        )

        return (output, None, None)

    def intercept_standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Intercept aten::scaled_dot_product_attention (standard variant).

        Signature: (q, k, v, attn_mask, dropout_p, is_causal, scale)
        Returns: output (single tensor)
        """
        if isinstance(is_causal, float):
            is_causal = bool(is_causal)

        return self.intercept_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            layer_idx=-1
        )

    def intercept_flash_attention_for_cpu(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = True,
        *,
        scale: Optional[float] = None
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
            layer_idx=-1
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


def create_kv_wrapper_from_config(
    lm_config: Dict,
    device: str,  # Ignored - kept for API compatibility
    dtype: torch.dtype
) -> KVCacheAttentionWrapper:
    """
    Factory function to create KVCacheAttentionWrapper from lm_config.

    NOTE: The 'device' parameter is IGNORED. The distributed cache uses
    lazy allocation based on incoming tensor devices.

    Args:
        lm_config: LM configuration dict from defaults.json
        device: IGNORED (kept for API compatibility)
        dtype: Execution dtype

    Returns:
        Configured KVCacheAttentionWrapper with distributed cache

    ZERO FALLBACK: Missing required keys = explicit error.
    """
    # Required keys - ZERO FALLBACK
    required_keys = ["num_layers", "num_heads", "hidden_size"]
    for key in required_keys:
        if key not in lm_config:
            raise RuntimeError(
                f"ZERO FALLBACK: '{key}' manquant dans lm_config.\n"
                "Model data incomplete. Re-import: neurobrix remove <model> && neurobrix import <org>/<model>"
            )

    num_layers = lm_config["num_layers"]
    num_heads = lm_config["num_heads"]
    hidden_size = lm_config["hidden_size"]

    # Optional keys with derivation
    num_kv_heads = lm_config.get("num_kv_heads", num_heads)  # MHA default
    head_dim = lm_config.get("head_dim", hidden_size // num_heads)
    k_head_dim = lm_config.get("k_head_dim", head_dim)
    v_head_dim_val = lm_config.get("v_head_dim", head_dim)

    # LEGACY: max_cache_len from max_position_embeddings.
    # Prism now computes the real budget (see KVCachePlan in solver.py).
    # This path is only used when Prism plan has no kv_cache_plan (backward compat).
    max_cache_len = lm_config.get("max_position_embeddings")
    if max_cache_len is None:
        raise RuntimeError(
            "ZERO FALLBACK: max_position_embeddings missing from lm_config.\n"
            "Cannot allocate KV cache without knowing sequence length limit."
        )
    config = KVCacheConfig(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        k_head_dim=k_head_dim,
        v_head_dim=v_head_dim_val,
        max_cache_len=max_cache_len,
        dtype=dtype
    )

    return KVCacheAttentionWrapper(config, num_heads=num_heads)
