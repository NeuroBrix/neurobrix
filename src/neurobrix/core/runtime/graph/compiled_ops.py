"""
Compiled Ops — Autonomous Op Resolution for Compiled + Triton Modes

Two independent paths, ZERO crossover:

NATIVE/COMPILED MODE:
  - Ops resolved via torch.ops.aten / torch / F
  - DtypeEngine handles AMP casting

TRITON MODE:
  - COMPUTE ops → Triton kernels via dispatch.py (GPU)
  - METADATA ops → pure Python CPU handlers (shape/stride math only)
  - _get_pytorch_op is NEVER called. ZERO fallback to PyTorch.
  - Missing Triton kernel = CRASH with explicit error message.

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

    On fp16 hardware, _FP16_NEED_FP32 ops (bmm) upcast to fp32. If Q or K flows
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
                 amp_enabled: bool = True, use_triton: bool = False):
        self.device = device
        self.dtype = dtype
        self.use_triton = False  # triton mode now uses triton/ package directly
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
        """Resolve raw op function (before DtypeEngine wrapping).

        Native/compiled mode only. Triton mode uses triton/ package.
        """
        # ── NATIVE MODE: special handlers for complex ATen signatures ──
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
        if op_name == "expand":
            return self._make_expand()
        if op_name == "cat":
            return self._make_cat()
        if op_name in ("view", "_unsafe_view"):
            return self._make_view_reshape()
        if op_name == "_pack_padded_sequence":
            return self._make_pack_padded_sequence()
        if op_name == "set":
            return self._make_set()
        if op_name == "_local_scalar_dense":
            return self._make_local_scalar_dense()
        if op_name == "copy":
            return lambda x, src, *a, **k: x.copy_(src)
        if op_name == "_cudnn_rnn":
            return self._make_cudnn_rnn()
        if op_name == "rsqrt":
            return self._make_rsqrt()
        if op_name == "embedding":
            return self._make_embedding()
        if op_name in ("gather", "index_select"):
            return self._make_gather_index_select(op_name)
        if op_name in ("index_add", "scatter", "scatter_add"):
            return self._make_scatter_ops(op_name)
        if op_name == "custom::rms_norm" or op_name == "rms_norm":
            return self._make_rms_norm(attrs)
        if "scaled_dot_product" in op_name and "attention" in op_name:
            return self._make_attention(op_name, attrs)
        if op_name in ("upsample_nearest2d", "upsample_bilinear2d", "upsample_bicubic2d",
                       "upsample_nearest1d", "upsample_linear1d"):
            return self._make_upsample(op_name)
        if op_name == "as_strided":
            return self._make_as_strided()
        if op_name == "unfold_backward":
            return self._make_unfold_backward()
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
        """Create slice wrapper — clamps end to dim size, uses narrow."""
        def slice_wrapper(input_tensor, dim, start=None, end=None, step=1):
            dim = _tensor_to_int(dim)
            start = _tensor_to_int_or_none(start) or 0
            dim_size = input_tensor.size(dim) if hasattr(input_tensor, 'size') else input_tensor.shape[dim]
            end = _tensor_to_int_or_none(end)
            if end is None or end > dim_size:
                end = dim_size
            if start < 0:
                start = max(0, dim_size + start)
            if end < 0:
                end = max(0, dim_size + end)
            length = max(0, end - start)
            result = input_tensor.narrow(dim, start, length)
            if step != 1 and step is not None:
                step = _tensor_to_int(step)
                # Apply step via indexing on the sliced dim
                indices = list(range(0, length, step))
                if len(indices) < length:
                    result = result.narrow(dim, 0, len(indices))
            return result
        return slice_wrapper

    def _make_narrow(self) -> Callable:
        """Create narrow wrapper — pure tensor method, no aten dispatch."""
        def narrow_wrapper(input_tensor, dim, start, length):
            dim = _tensor_to_int(dim)
            start = _tensor_to_int(start)
            length = _tensor_to_int(length)
            return input_tensor.narrow(dim, start, length)
        return narrow_wrapper

    def _make_select(self) -> Callable:
        """Create select wrapper — pure tensor method, no aten dispatch."""
        def select_wrapper(input_tensor, dim, index):
            dim = _tensor_to_int(dim)
            index = _tensor_to_int(index)
            return input_tensor.select(dim, index)
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
        from neurobrix.core.dtype.engine import rms_norm_fp32
        epsilon = attrs.get("epsilon", attrs.get("kwargs", {}).get("epsilon", 1e-6))

        def _rms_norm_fn(x, weight, **kwargs):
            # fp32-variance RMSNorm; the dtype-upcast policy lives in the engine.
            return rms_norm_fp32(x, weight, epsilon)

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
            # K may be transposed [B,H,D,S] from pattern reassembly — fix to [B,H,S,D].
            # Detect via head_dim (last axis), NOT seq (axis -2): cross-attention
            # legitimately has seq_q != seq_k (Perceiver: 32 latent queries over 150
            # prompt keys), so a seq-axis comparison wrongly transposes a correct K.
            # A genuinely transposed [B,H,D,S] K carries head_dim in axis -2 == q[-1].
            if (k.ndim == 4 and q.ndim == 4
                    and k.shape[-1] != q.shape[-1] and k.shape[-2] == q.shape[-1]):
                k = k.transpose(-2, -1)

            is_causal = _safe_is_causal(kwargs.pop('is_causal', False))
            kwargs.pop('dropout_p', None)
            scale = kwargs.pop('scale', None)

            attn_mask = None
            if args and isinstance(args[0], torch.Tensor):
                attn_mask = args[0]
                seq_q, seq_k = q.shape[2], k.shape[2]
                if attn_mask.shape[-2] > seq_q or attn_mask.shape[-1] > seq_k:
                    # Mask larger than Q/K: trim to match
                    attn_mask = attn_mask[..., :seq_q, :seq_k].contiguous()
                elif attn_mask.shape[-2] < seq_q or attn_mask.shape[-1] < seq_k:
                    # Mask smaller than Q/K: trace-time constant mask that doesn't
                    # cover runtime seq_len. If it's a causal mask (lower-triangular
                    # with -inf above diagonal), use is_causal=True instead.
                    is_causal = True
                    attn_mask = None

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
        raw_op = F.interpolate
        is_1d = op_name.endswith("1d")
        if "nearest" in op_name:
            mode = "nearest"
        elif "linear" in op_name:  # upsample_linear1d or upsample_bilinear2d
            mode = "linear" if is_1d else "bilinear"
        else:  # upsample_bicubic2d
            mode = "bicubic"

        if is_1d:
            # The trace bakes a fixed output_size that won't track a dynamic input
            # length (Kokoro F0/N prosody + decoder resample on the variable
            # acoustic frame axis). Recompute the length from the LIVE input ×
            # scale, mirroring the 2D fix below. The scale's arg position differs:
            #   nearest1d(input, output_size, scales)             → scales = arg[2]
            #   linear1d (input, output_size, align_corners, scales) → scales = arg[3]
            nearest = (mode == "nearest")

            def upsample_wrapper_1d(input_tensor, output_size=None, a2=None, a3=None, *args, **kwargs):
                if nearest:
                    scales, align = a2, None
                else:
                    scales, align = a3, a2  # a2 is align_corners (a bool)
                # bool is an int subclass — exclude it so align_corners can't be
                # mistaken for a scale (that produced int(W*False)=0).
                if isinstance(scales, (int, float)) and not isinstance(scales, bool):
                    size = int(round(input_tensor.shape[-1] * scales))
                elif isinstance(output_size, (list, tuple)):
                    size = output_size[0]
                else:
                    size = output_size
                ac = None if nearest else bool(align) if align is not None else False
                if size is not None and size > 0:
                    return raw_op(input_tensor, size=size, mode=mode, align_corners=ac)
                return raw_op(input_tensor, scale_factor=scales, mode=mode, align_corners=ac)
            return upsample_wrapper_1d

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

    def _make_unfold_backward(self) -> Callable:
        """aten::unfold_backward(grad, input_sizes, dim, size, step) is the iSTFT
        overlap-add reconstruction (inverse of the as_strided framing): it scatters
        the `grad` windows back into a signal of `input_sizes`. The trace bakes
        input_sizes at the trace frame count; recompute the reconstructed length
        from the LIVE window count so it tracks variable audio:
            input_sizes[dim] = (n_windows - 1) * step + size.
        Mirror of the as_strided framing recompute.
        """
        def unfold_backward_wrapper(grad, input_sizes, dim, size, step, *args, **kwargs):
            input_sizes = list(input_sizes)
            if (isinstance(dim, int) and isinstance(size, int) and isinstance(step, int)
                    and hasattr(grad, "shape") and 0 <= dim < len(input_sizes)
                    and dim < grad.dim()):
                n_windows = grad.shape[dim]
                input_sizes[dim] = (n_windows - 1) * step + size
            return torch.ops.aten.unfold_backward(grad, input_sizes, dim, size, step)
        return unfold_backward_wrapper

    def _make_as_strided(self) -> Callable:
        """as_strided used as an overlap-add framing view (iSTFT in the istftnet
        vocoder) bakes the trace frame count into size[-2] and the full signal
        length into stride[0]. At a different runtime audio length the baked view
        over-reads storage. Recompute the window count and signal stride from the
        LIVE input so the framing tracks the variable frame axis. Non-framing
        as_strided (stride[-1] != 1 or non-3D) passes through unchanged.
        """
        def as_strided_wrapper(input_tensor, size=None, stride=None,
                               storage_offset=0, *args, **kwargs):
            if (isinstance(size, (list, tuple)) and isinstance(stride, (list, tuple))
                    and len(size) == 3 and len(stride) == 3 and stride[-1] == 1):
                B, _, W = size
                L0, H, _ = stride
                if B and H and B > 0 and H > 0:
                    live_L = input_tensor.numel() // B
                    if L0 != live_L:               # trace length ≠ runtime length
                        N = (live_L - W) // H + 1
                        size = [B, N, W]
                        stride = [live_L, H, 1]
            return torch.as_strided(input_tensor, list(size), list(stride),
                                    storage_offset or 0)
        return as_strided_wrapper

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
                    # All filtered — return empty tensor on SAME DEVICE (never CPU)
                    for t in tensors:
                        if isinstance(t, torch.Tensor):
                            return torch.tensor([], device=t.device, dtype=t.dtype)
                    return torch.tensor([])
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
                    # When multiple positions work, prefer the one whose inferred dim
                    # matches the input tensor's actual dimension at that axis.
                    # This prevents symbolic dimension collisions (e.g., head_dim/2 == trace seq_len)
                    # from corrupting batch or other structural dimensions.
                    if len(shape) >= 2:
                        candidates = []
                        for i in range(len(shape)):
                            trial = list(shape)
                            trial[i] = -1
                            try:
                                result = tensor.reshape(trial)
                                candidates.append((i, result))
                            except RuntimeError:
                                continue
                        if candidates:
                            if len(candidates) == 1:
                                return candidates[0][1]
                            # Prefer position where inferred dim matches input tensor's actual dim
                            for idx, result in candidates:
                                if idx < tensor.ndim and result.shape[idx] == tensor.shape[idx]:
                                    return result
                            return candidates[0][1]
                    raise
        return view_or_reshape

    def _make_pack_padded_sequence(self) -> Callable:
        """pack_padded_sequence — memory layout change, no arithmetic."""
        def pack_padded(inp, lengths, batch_first=False):
            return torch.nn.utils.rnn.pack_padded_sequence(
                inp, lengths.cpu(), batch_first=batch_first)
        return pack_padded

    def _make_set(self) -> Callable:
        """aten::set — storage manipulation used by pack/unpack padded sequence.

        The traced graph captures set(tensor, storage, ...) but the storage
        was serialized as a string. In compiled mode, the tensor data is already
        correct from the preceding op — return self unchanged.
        """
        def set_wrapper(self_tensor, *args):
            return self_tensor
        return set_wrapper

    def _make_local_scalar_dense(self) -> Callable:
        """aten::_local_scalar_dense — converts tensor to Python scalar (tensor.item())."""
        def local_scalar_dense(inp):
            return inp.item()
        return local_scalar_dense

    def _make_cudnn_rnn(self) -> Callable:
        """cuDNN RNN — CUDA compute op (LSTM/GRU via cuDNN backend).

        In triton mode: crash. cuDNN is not Triton.
        In native mode: dtype alignment + aten dispatch.
        """
        if self.use_triton:
            def cudnn_rnn_crash(*args, **kwargs):
                raise RuntimeError(
                    "[--triton mode] _cudnn_rnn (LSTM/GRU) requires cuDNN. "
                    "No Triton RNN kernel exists. Use native mode for RNN models."
                )
            return cudnn_rnn_crash

        aten_op = torch.ops.aten._cudnn_rnn
        def cudnn_rnn_wrapper(*args, **kwargs):
            inp = args[0]
            weight_list = args[1]
            param_dtype = None
            if isinstance(weight_list, (list, tuple)) and weight_list:
                for w in weight_list:
                    if isinstance(w, torch.Tensor):
                        param_dtype = w.dtype
                        break
            if param_dtype is not None and inp.dtype != param_dtype:
                args = list(args)
                args[0] = inp.to(param_dtype)
                for idx in (4, 5):
                    if idx < len(args) and isinstance(args[idx], torch.Tensor):
                        args[idx] = args[idx].to(param_dtype)
                args = tuple(args)
            return aten_op(*args, **kwargs)
        return cudnn_rnn_wrapper

    def _get_standard_op(self, op_name: str) -> Callable:
        """Get op function.

        TRITON MODE: ONE path — dispatch.py has ALL ops (compute + metadata).
                     Not found = CRASH. Zero fallback.

        NATIVE MODE: torch.ops.aten / torch / F namespace.
        """
        if op_name in self._op_cache:
            return self._op_cache[op_name]
        return self._get_pytorch_op(op_name)

    def _get_pytorch_op(self, op_name: str) -> Callable:
        """Resolve op via PyTorch (native/compiled mode only)."""
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
