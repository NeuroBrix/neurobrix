"""
Compiled Ops - 100% Autonomous Op Resolution for Compiled Mode

This module provides ALL operations for CompiledSequence without any dependency
on NativeATenDispatcher. The compiled mode is 100% independent from native mode.

Key Features:
- Pre-compiled function factories that capture all special logic at compile time
- Zero runtime isinstance() checks
- Direct PyTorch ATen calls via torch.ops.aten
- All multi-resolution fixes built-in
- All stability fixes built-in

ZERO DEPENDENCY: This module does NOT import from sequential_dispatcher.py
"""

import math
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, Optional


def _cast_attn_mask(mask, query):
    """Cast attention mask to query dtype if needed. Shared across attention variants."""
    if mask is not None and isinstance(mask, torch.Tensor) and mask.dtype != query.dtype:
        return mask.to(query.dtype)
    return mask


def _align_qkv_dtypes(q, k, v):
    """Align Q/K/V dtypes for SDPA. Required when upstream AMP ops produce mixed dtypes.

    On fp16 hardware, FP16_PRECISION_OPS (bmm) upcast to fp32. If Q or K flows
    through bmm but V doesn't, SDPA receives mixed fp32/fp16 → crash.
    Cast all to the narrowest common dtype (prefer compute_dtype for performance).
    """
    if q.dtype == k.dtype == v.dtype:
        return q, k, v
    # Use the narrowest dtype (fp16/bf16 preferred over fp32 for SDPA performance)
    target = min((q.dtype, k.dtype, v.dtype), key=lambda d: torch.tensor([], dtype=d).element_size())
    if q.dtype != target:
        q = q.to(target)
    if k.dtype != target:
        k = k.to(target)
    if v.dtype != target:
        v = v.to(target)
    return q, k, v


def _safe_is_causal(val: Any) -> bool:
    """Convert is_causal to Python bool. Handles tensor, int, bool, and float args.

    CRITICAL: Flex FlowMatch passes is_causal as a tensor. DtypeEngine may
    downcast it to fp16. Must extract bool value or default to False.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, torch.Tensor):
        return bool(val.item()) if val.numel() == 1 else False
    if isinstance(val, (int, float)):
        return bool(val)
    return False


def _safe_dropout(val: Any) -> float:
    """Convert dropout_p to Python float. Handles tensor and numeric args."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, torch.Tensor):
        return float(val.item()) if val.numel() == 1 else 0.0
    return 0.0


def _tensor_to_int(val: Any) -> Any:
    """Convert 0-d tensor to Python int if needed."""
    if isinstance(val, torch.Tensor) and val.ndim == 0:
        return int(val.item())
    return val


def _tensor_to_int_or_none(val: Any) -> Any:
    """Convert 0-d tensor to Python int, or return None if None."""
    if val is None:
        return None
    return _tensor_to_int(val)


# ============================================================================
# OP RESOLVER - Main Entry Point
# ============================================================================

class CompiledOpResolver:
    """
    Autonomous op resolution for compiled mode.

    This is the ONLY source of truth for compiled mode operations.
    No dependency on NativeATenDispatcher.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, graph_dtype: Optional[torch.dtype] = None,
                 amp_enabled: bool = True):
        self.device = device
        self.dtype = dtype
        self._op_cache: Dict[str, Callable] = {}

        # DtypeEngine: single entry point for all dtype decisions
        from neurobrix.core.dtype.engine import DtypeEngine
        self.dtype_engine = DtypeEngine(dtype, graph_dtype=graph_dtype, amp_enabled=amp_enabled)

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def get_op_func(self, op_name: str, attrs: Dict[str, Any]) -> Callable:
        """
        Get the function for an operation with all logic pre-baked.

        Args:
            op_name: The bare operation name (without aten:: prefix)
            attrs: The operation attributes from graph.json

        Returns:
            A callable that executes the operation (AMP-wrapped if applicable)
        """
        # lift_fresh is special — not dtype-related
        if op_name == "lift_fresh":
            return self._make_lift_fresh(attrs)

        # _to_copy and all other ops go through DtypeEngine.compile_op()
        if op_name == "_to_copy":
            return self.dtype_engine.compile_op(f"aten::{op_name}", None, attrs)

        # Resolve the raw function via special handlers or standard lookup
        func = self._resolve_op_func(op_name, attrs)

        # DtypeEngine wraps with AMP casting at compile time
        # Custom ops already have their prefix; only add aten:: for standard ops
        dtype_op_type = op_name if "::" in op_name else f"aten::{op_name}"
        return self.dtype_engine.compile_op(dtype_op_type, func, attrs)

    def _resolve_op_func(self, op_name: str, attrs: Dict[str, Any]) -> Callable:
        """Resolve raw op function (before DtypeEngine wrapping)."""
        if op_name == "rsqrt":
            return self._make_rsqrt()

        if op_name == "slice":
            return self._make_slice()

        if op_name == "narrow":
            return self._make_narrow()

        if op_name == "select":
            return self._make_select()

        if op_name in ("squeeze", "unsqueeze"):
            return self._make_squeeze_unsqueeze(op_name)

        if op_name == "transpose":
            return self._make_transpose()

        if op_name in ("split", "chunk"):
            return self._make_split_chunk(op_name)

        if op_name == "embedding":
            return self._make_embedding()

        if op_name in ("gather", "index_select"):
            return self._make_gather_index_select(op_name)

        if op_name in ("index_add", "scatter", "scatter_add"):
            return self._make_scatter_ops(op_name)

        # RMSNorm (pattern-reassembled)
        if op_name == "custom::rms_norm" or op_name == "rms_norm":
            return self._make_rms_norm(attrs)

        # Attention ops
        if "scaled_dot_product" in op_name and "attention" in op_name:
            return self._make_attention(op_name, attrs)

        # Upsample ops (multi-resolution fix)
        if op_name in ("upsample_nearest2d", "upsample_bilinear2d", "upsample_bicubic2d"):
            return self._make_upsample(op_name)

        # Expand (multi-resolution fix)
        if op_name == "expand":
            return self._make_expand()

        # Cat (Gemma2 scalar handling)
        if op_name == "cat":
            return self._make_cat()

        # View/Reshape (KV cache compatibility - transpose returns non-contiguous tensors)
        if op_name in ("view", "_unsafe_view"):
            return self._make_view_reshape()

        # Standard ops - resolve via PyTorch
        return self._get_standard_op(op_name)

    # ========================================================================
    # SPECIAL OP FACTORIES
    # ========================================================================

    def _make_lift_fresh(self, attrs: Dict[str, Any]) -> Callable:
        """
        Create lift_fresh function with all logic pre-compiled.

        lift_fresh is used by FakeTensorMode to mark tensors as "fresh".
        Graph capture sometimes incorrectly links to wrong tensor, causing
        shape mismatches. This handles all those cases.

        CRITICAL: Sana/Gemma2 text encoders have KV-cache patterns where:
        - input_shapes: [[1, 506, 1024]] (real tensor)
        - output_shapes: [[0]] (FakeTensor capture error)

        For these cases, we MUST passthrough the input - the [0] output shape
        is a tracer bug, not the actual expected output.
        """
        output_shapes = attrs.get("output_shapes", [])
        input_shapes = attrs.get("input_shapes", [])
        expected_output_shape = output_shapes[0] if output_shapes else None
        expected_input_shape = input_shapes[0] if input_shapes else None

        # Case 1: Graph expected scalar (shape=[])
        if expected_output_shape == []:
            def lift_fresh_scalar(inp):
                if inp is None:
                    return None
                if not isinstance(inp, torch.Tensor) or inp.ndim == 0:
                    return inp
                # Multi-dim tensor but expected scalar - compute scale
                # This handles embedding *= sqrt(hidden_size) where tracer captured
                # the scalar but runtime produces the weight tensor instead.
                # UNIVERSAL: derive hidden_size from tensor shape, not a lookup table.
                if len(inp.shape) >= 2:
                    hidden_size = inp.shape[-1]
                    scale = math.sqrt(hidden_size)
                    return torch.tensor(scale, dtype=inp.dtype, device=inp.device)
                return torch.tensor(1.0, dtype=inp.dtype, device=inp.device)
            return lift_fresh_scalar

        # Case 2: Graph expected empty tensor (shape=[0])
        # Return empty tensor so downstream cat() will filter it out.
        # This matches sequential_dispatcher.py behavior (lines 187-192).
        if expected_output_shape == [0]:
            def lift_fresh_empty(inp):
                if inp is None:
                    return None
                if not isinstance(inp, torch.Tensor):
                    return inp
                return torch.empty(0, dtype=inp.dtype, device=inp.device)
            return lift_fresh_empty

        # Case 3: Normal pass-through
        def lift_fresh_passthrough(inp):
            if inp is None:
                return None
            return inp
        return lift_fresh_passthrough

    def _make_rsqrt(self) -> Callable:
        """Create stable rsqrt function that prevents division by zero."""
        def rsqrt_stable(inp):
            if isinstance(inp, torch.Tensor):
                clamped = torch.clamp(inp, min=1e-10)
                return torch.rsqrt(clamped)
            return torch.rsqrt(inp)
        return rsqrt_stable

    def _make_slice(self) -> Callable:
        """Create slice wrapper that converts tensor args to Python ints."""
        raw_op = torch.ops.aten.slice.Tensor  # type: ignore[attr-defined]

        def slice_wrapper(input_tensor, dim, start=None, end=None, step=1):
            dim = _tensor_to_int(dim)
            start = _tensor_to_int_or_none(start)
            end = _tensor_to_int_or_none(end)
            step = _tensor_to_int(step)
            return raw_op(input_tensor, dim, start, end, step)  # type: ignore[operator]
        return slice_wrapper

    def _make_narrow(self) -> Callable:
        """Create narrow wrapper that converts tensor args to Python ints."""
        raw_op = torch.ops.aten.narrow  # type: ignore[attr-defined]

        def narrow_wrapper(input_tensor, dim, start, length):
            dim = _tensor_to_int(dim)
            start = _tensor_to_int(start)
            length = _tensor_to_int(length)
            return raw_op(input_tensor, dim, start, length)  # type: ignore[operator]
        return narrow_wrapper

    def _make_select(self) -> Callable:
        """Create select wrapper that converts tensor args to Python ints."""
        raw_op = torch.ops.aten.select.int  # type: ignore[attr-defined]

        def select_wrapper(input_tensor, dim, index):
            dim = _tensor_to_int(dim)
            index = _tensor_to_int(index)
            return raw_op(input_tensor, dim, index)  # type: ignore[operator]
        return select_wrapper

    def _make_squeeze_unsqueeze(self, op_name: str) -> Callable:
        """Create squeeze/unsqueeze wrapper that converts dim to Python int."""
        if op_name == "squeeze":
            raw_op = torch.squeeze
        else:
            raw_op = torch.unsqueeze

        def dim_wrapper(input_tensor, dim=None):
            if dim is not None:
                dim = _tensor_to_int(dim)
                return raw_op(input_tensor, dim=dim)
            if op_name == "squeeze":
                return raw_op(input_tensor, dim=None)  # type: ignore[call-arg]
            else:
                # unsqueeze requires dim parameter
                raise ValueError("unsqueeze requires dim parameter")
        return dim_wrapper

    def _make_transpose(self) -> Callable:
        """Create transpose wrapper that converts dims to Python ints."""
        raw_op = torch.transpose

        def transpose_wrapper(input_tensor, dim0, dim1):
            dim0 = _tensor_to_int(dim0)
            dim1 = _tensor_to_int(dim1)
            return raw_op(input_tensor, dim0, dim1)
        return transpose_wrapper

    def _make_split_chunk(self, op_name: str) -> Callable:
        """Create split/chunk wrapper that converts args to Python ints."""
        raw_op = torch.split if op_name == "split" else torch.chunk

        def split_wrapper(input_tensor, split_size_or_sections, dim=0):
            if isinstance(split_size_or_sections, torch.Tensor):
                split_size_or_sections = int(split_size_or_sections.item())
            dim = _tensor_to_int(dim)
            return raw_op(input_tensor, split_size_or_sections, dim)
        return split_wrapper

    def _make_embedding(self) -> Callable:
        """Create embedding wrapper that ensures int64 indices.

        ATen signature: torch.embedding(weight, indices, ...)
        """
        raw_op = torch.embedding

        def embedding_wrapper(weight, indices, *args, **kwargs):
            # Ensure indices are int64 (not float)
            if isinstance(indices, torch.Tensor) and indices.is_floating_point():
                indices = indices.long()
            return raw_op(weight, indices, *args, **kwargs)
        return embedding_wrapper

    def _make_gather_index_select(self, op_name: str) -> Callable:
        """Create gather/index_select wrapper with int conversion."""
        if op_name == "gather":
            raw_op = torch.gather
        else:
            raw_op = torch.index_select

        def index_wrapper(input_tensor, dim, index, *args, **kwargs):
            dim = _tensor_to_int(dim)
            if isinstance(index, torch.Tensor) and index.is_floating_point():
                index = index.long()
            return raw_op(input_tensor, dim, index, *args, **kwargs)
        return index_wrapper

    def _make_scatter_ops(self, op_name: str) -> Callable:
        """Create scatter/scatter_add/index_add wrapper with int conversion."""
        if op_name == "scatter":
            raw_op = torch.scatter
        elif op_name == "scatter_add":
            raw_op = torch.scatter_add
        else:  # index_add
            raw_op = torch.index_add

        def scatter_wrapper(input_tensor, dim, index, src, *args, **kwargs):
            dim = _tensor_to_int(dim)
            if isinstance(index, torch.Tensor) and index.is_floating_point():
                index = index.long()
            return raw_op(input_tensor, dim, index, src, *args, **kwargs)
        return scatter_wrapper

    def _make_rms_norm(self, attrs: Dict[str, Any]) -> Callable:
        """
        Create RMS normalization function.

        Pattern-reassembled from: pow(x,2) → mean → add(eps) → rsqrt → mul(x) → [cast] → mul(weight)

        Calling convention: CompiledSequence calls op.func(*args, **kwargs)
        where args = (x_tensor, weight_tensor) and kwargs may include epsilon.
        Epsilon is captured in closure at compile time.
        """
        import torch
        epsilon = attrs.get("epsilon", attrs.get("kwargs", {}).get("epsilon", 1e-6))

        def _rms_norm_fn(x, weight, **kwargs):
            # Compute in fp32 for numerical stability (matches decomposed pattern)
            x_fp32 = x.to(torch.float32)
            variance = x_fp32.pow(2).mean(-1, keepdim=True)
            x_normed = x_fp32 * torch.rsqrt(variance + epsilon)
            return x_normed.to(weight.dtype) * weight

        return _rms_norm_fn

    def _make_attention(self, op_name: str, attrs: Dict[str, Any]) -> Callable:
        """
        Create scaled_dot_product_attention function with variant handling.

        Handles ATen SDPA variants with different return signatures:
        - _scaled_dot_product_flash_attention_for_cpu → (output, lse)
        - _scaled_dot_product_efficient_attention     → (output, lse, seed, offset)
        - _scaled_dot_product_flash_attention         → (output, lse, seed, offset)
        - scaled_dot_product_attention                → output

        NeuroBrix executes the op as traced. DtypeEngine handles dtype.
        """
        raw_name = op_name.replace("aten::", "")
        base_name = raw_name.split(".")[0]

        if base_name == "_scaled_dot_product_flash_attention_for_cpu":
            def flash_cpu_attention(q, k, v, dropout_p=0.0, is_causal=False, **kwargs):
                q, k, v = _align_qkv_dtypes(q, k, v)
                scale = kwargs.get("scale", None)
                attn_mask = _cast_attn_mask(kwargs.get("attn_mask", None), q)
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=_safe_dropout(dropout_p),
                    is_causal=_safe_is_causal(is_causal),
                    scale=scale,
                )
                lse = torch.zeros((q.shape[0], q.shape[1], q.shape[2]),
                                  device=q.device, dtype=q.dtype)
                return output, lse
            return flash_cpu_attention

        if base_name in ("_scaled_dot_product_efficient_attention",
                         "_scaled_dot_product_flash_attention"):
            def efficient_attention(q, k, v, attn_bias=None, compute_lse=False,
                                   dropout_p=0.0, is_causal=False, scale=None, *args):
                q, k, v = _align_qkv_dtypes(q, k, v)
                if not q.is_contiguous():
                    q = q.contiguous()
                if not k.is_contiguous():
                    k = k.contiguous()
                if not v.is_contiguous():
                    v = v.contiguous()
                attn_bias = _cast_attn_mask(attn_bias, q)
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_bias,
                    dropout_p=_safe_dropout(dropout_p),
                    is_causal=_safe_is_causal(is_causal),
                    scale=scale,
                )
                lse = torch.zeros((q.shape[0], q.shape[1], q.shape[2]),
                                  device=q.device, dtype=q.dtype)
                philox_seed = torch.tensor(0, device=q.device, dtype=torch.int64)
                philox_offset = torch.tensor(0, device=q.device, dtype=torch.int64)
                return output, lse, philox_seed, philox_offset
            return efficient_attention

        # Standard SDPA — handles pattern-reassembled ops + native SDPA
        def standard_attention(q, k, v, *args, **kwargs):
            q, k, v = _align_qkv_dtypes(q, k, v)
            # K may be transposed [B,H,D,S] from pattern reassembly — fix to [B,H,S,D]
            if k.ndim == 4 and q.ndim == 4 and k.shape[-2] != q.shape[-2]:
                k = k.transpose(-2, -1)

            is_causal = _safe_is_causal(kwargs.pop('is_causal', False))
            kwargs.pop('dropout_p', None)
            scale = kwargs.pop('scale', None)

            attn_mask = None
            if args and isinstance(args[0], torch.Tensor):
                attn_mask = args[0]
                seq_q, seq_k = q.shape[2], k.shape[2]
                if attn_mask.shape[-2] > seq_q or attn_mask.shape[-1] > seq_k:
                    attn_mask = attn_mask[..., :seq_q, :seq_k].contiguous()

            if attn_mask is not None:
                attn_mask = _cast_attn_mask(attn_mask, q)

            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                is_causal=is_causal, scale=scale,
            )
        return standard_attention

    def _make_upsample(self, op_name: str) -> Callable:
        """
        Create upsample function with multi-resolution fix.

        Graph contains hardcoded output_size from trace time. When scale factors
        are available, we recompute output_size from actual input dimensions.
        """
        if op_name == "upsample_nearest2d":
            raw_op = F.interpolate
            mode = "nearest"
        elif op_name == "upsample_bilinear2d":
            raw_op = F.interpolate
            mode = "bilinear"
        else:  # upsample_bicubic2d
            raw_op = F.interpolate
            mode = "bicubic"

        def upsample_wrapper(input_tensor, output_size=None, scales_h=None, scales_w=None, *args, **kwargs):
            # Multi-resolution fix: if scales are provided, recompute output_size
            if scales_h is not None and scales_w is not None:
                if isinstance(scales_h, (int, float)) and isinstance(scales_w, (int, float)):
                    h_in = input_tensor.shape[2]
                    w_in = input_tensor.shape[3]
                    output_size = (int(h_in * scales_h), int(w_in * scales_w))

            if output_size is not None:
                return raw_op(input_tensor, size=output_size, mode=mode,
                             align_corners=False if mode != "nearest" else None)
            else:
                # Fallback: use scale_factor
                return raw_op(input_tensor, scale_factor=(scales_h, scales_w), mode=mode,
                             align_corners=False if mode != "nearest" else None)
        return upsample_wrapper

    def _make_expand(self) -> Callable:
        """
        Create expand function with multi-resolution fix.

        Graph contains hardcoded expand sizes from trace time. When the tensor
        has non-singleton dimensions that don't match, use actual dimensions.
        """
        def expand_wrapper(input_tensor, size, *args, **kwargs):
            if isinstance(size, (list, tuple)) and len(size) == len(input_tensor.shape):
                new_size = list(size)
                for i, (actual, target) in enumerate(zip(input_tensor.shape, size)):
                    # expand can only broadcast from size 1
                    # If actual != 1 and actual != target, use actual
                    if actual != 1 and actual != target and target != -1:
                        new_size[i] = actual
                return input_tensor.expand(*new_size)
            return input_tensor.expand(*size)
        return expand_wrapper


    def _make_cat(self) -> Callable:
        """
        Create cat function with Gemma2 scalar tensor handling.

        Gemma2 creates scalar -inf tensors for attention masking, then tries
        to cat them. 0-dimensional tensors cannot be concatenated.
        """
        def cat_wrapper(tensors, dim=0, *args, **kwargs):
            if isinstance(tensors, (list, tuple)):
                # Filter out 0-dimensional and empty tensors
                valid_tensors = []
                for t in tensors:
                    if isinstance(t, torch.Tensor):
                        if t.ndim == 0 or t.numel() == 0:
                            continue
                    valid_tensors.append(t)

                if len(valid_tensors) == 0:
                    # All filtered - return empty tensor
                    device = tensors[0].device if tensors and isinstance(tensors[0], torch.Tensor) else 'cpu'
                    return torch.tensor([], device=device)
                elif len(valid_tensors) == 1:
                    return valid_tensors[0]
                else:
                    return torch.cat(valid_tensors, dim=dim)
            return torch.cat(tensors, dim=dim)
        return cat_wrapper

    def _make_view_reshape(self) -> Callable:
        """
        Create view function that falls back to reshape for non-contiguous tensors.

        CRITICAL FIX for KV cache: After SDPA, transpose returns non-contiguous
        tensors. The traced graph uses aten::view, but view fails on non-contiguous
        tensors. Using reshape instead handles both cases correctly.

        MoE FIX: When symbolic seq_len changes between trace and runtime, view/reshape
        may fail because numel differs from the product of the target shape dims.
        The fallback tries each position as -1 (not just the last) to infer the
        changing dimension. This handles MoE routing views where the dynamic dim
        can be in any position.
        """
        def view_or_reshape(tensor, shape, *args, **kwargs):
            # Try view first (faster, no copy)
            try:
                return tensor.view(shape)
            except RuntimeError:
                try:
                    # Fallback to reshape (handles non-contiguous)
                    return tensor.reshape(shape)
                except RuntimeError:
                    # Numel mismatch: tensor has trace-time dims, shape has runtime dims.
                    # Try each position as -1 to find the symbolic (changed) dimension.
                    if len(shape) >= 2:
                        for i in range(len(shape)):
                            trial = list(shape)
                            trial[i] = -1
                            try:
                                result = tensor.reshape(trial)
                                return result
                            except RuntimeError:
                                continue
                    raise
        return view_or_reshape

    def _get_standard_op(self, op_name: str) -> Callable:
        """
        Get standard PyTorch op function.

        Resolution priority:
        1. torch.ops.aten
        2. torch namespace
        3. torch.nn.functional
        """
        if op_name in self._op_cache:
            return self._op_cache[op_name]

        # Priority 1: torch.ops.aten
        if hasattr(torch.ops.aten, op_name):
            op = getattr(torch.ops.aten, op_name)
            self._op_cache[op_name] = op
            return op

        # Priority 2: torch namespace
        if hasattr(torch, op_name):
            op = getattr(torch, op_name)
            self._op_cache[op_name] = op
            return op

        # Priority 3: torch.nn.functional
        if hasattr(F, op_name):
            op = getattr(F, op_name)
            self._op_cache[op_name] = op
            return op

        raise AttributeError(
            f"[CompiledOpResolver] Op '{op_name}' not found in "
            f"torch.ops.aten, torch, or torch.nn.functional"
        )


# ============================================================================
# SINGLETON INSTANCE - For convenience
# ============================================================================

_resolver_instance: Optional[CompiledOpResolver] = None


def get_compiled_op_resolver(device: torch.device, dtype: torch.dtype) -> CompiledOpResolver:
    """Get or create the compiled op resolver singleton."""
    global _resolver_instance
    if _resolver_instance is None or _resolver_instance.device != device or _resolver_instance.dtype != dtype:
        _resolver_instance = CompiledOpResolver(device, dtype)
    return _resolver_instance
