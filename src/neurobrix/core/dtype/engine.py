# core/dtype/engine.py
"""
DtypeEngine — AMP-Driven Dtype Engine for Runtime Execution.

Implements PyTorch's Automatic Mixed Precision (AMP) autocast rules:
  - FP32 ops: numerically sensitive ops upcast to fp32 (pow, rsqrt, softmax, ...)
  - FP16 ops: compute-heavy ops cast to compute_dtype (matmul, conv, ...)
  - Promote ops: mixed-dtype inputs promote to widest type
  - _to_copy: Prism-driven dtype remapping
  - Constants: convert to compute_dtype

Source of truth: PyTorch aten/src/ATen/autocast_mode.h
  AT_FORALL_FP32, AT_FORALL_FP32_SET_OPT_DTYPE,
  AT_FORALL_LOWER_PRECISION_FP, AT_FORALL_PROMOTE

PRISM IS THE MASTER: compute_dtype comes from Prism. We apply it.
ZERO SEMANTIC: No model/family knowledge. Only tensor dtype math.
"""
import torch
from typing import Callable, Optional, Dict, Any, FrozenSet


# Max representable value per dtype (for overflow clamping before downcast)
_DTYPE_MAX = {
    torch.float16: 65504.0,
    # bf16 has same exponent range as fp32, no clamping needed
}


def configure_fp16_matmul_precision():
    """
    Configure PyTorch for stable fp16 matrix multiplication.

    On V100 GPUs, fp16 matmul can produce NaN with large tensors due to
    reduced precision accumulation. This disables that optimization.

    Source: https://github.com/pytorch/pytorch/issues/45724
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


# Apply V100 matmul fix at module load time
configure_fp16_matmul_precision()


# ============================================================================
# AMP OP CLASSIFICATION
#
# From PyTorch aten/src/ATen/autocast_mode.h (AT_FORALL_* macros).
# These are the CUDA autocast rules — universally applicable.
#
# AUDIT STATUS (Feb 2026): Cross-referenced against PyTorch 2.10 source.
#   - AT_FORALL_FP32:                    100% match (all 50+ ops)
#   - AT_FORALL_FP32_SET_OPT_DTYPE:      100% match (all 9 ops)
#   - AT_FORALL_LOWER_PRECISION_FP:       97% match (30/32 ops)
#   - AT_FORALL_PROMOTE:                 100% match (all 11 ops)
#
# DEVIATIONS from PyTorch (intentional):
#   1. bmm: PyTorch=FP16, Ours=FP32.
#      Reason: T5-XXL decomposed attention (no 1/sqrt(d_k) scaling)
#      overflows fp16 → Inf → NaN. Affects Flex/PixArt text_encoders (48
#      bmm each). Sana bmm inputs are already fp32 → wrapper is a no-op.
#      DeepSeek/Janus/Qwen3 bmm at seq_len=1 → negligible cost. ~1ms/step.
#   2. SDPA ops: PyTorch=FP16, Ours=Excluded.
#      Reason: compiled_ops._make_attention() calls F.sdpa directly.
#      DtypeEngine must not double-wrap SDPA inputs.
#
# EXTENSIONS beyond PyTorch:
#   - polar, view_as_complex → FP32 (complex32 doesn't exist on CUDA)
#   - native_group_norm → FP32 (extra safety, covers native dispatch)
#   - AMP_OVERFLOW_PROTECT_OPS → add/sub post-compute clamp for fp16
#   - _upcast_matmul was planned but removed (see note near line 190)
#
# HARDWARE COMPATIBILITY:
#   - V100 (sm_70): fp16 compute, AMP for bf16 graphs
#   - A100 (sm_80): bf16/fp16, TF32 auto-enabled for fp32 ops
#   - H100 (sm_90): bf16/fp16, TF32 auto-enabled, FlashAttn v2
#   - All consumer GPUs (RTX 30xx/40xx): bf16/fp16, compatible
# ============================================================================

# Ops that MUST run in float32 for numerical stability.
# Combines AT_FORALL_FP32 + AT_FORALL_FP32_SET_OPT_DTYPE.
# Output stays in fp32 — downstream FP16 ops bring it back to compute_dtype.
AMP_FP32_OPS: FrozenSet[str] = frozenset({
    # --- AT_FORALL_FP32 (PyTorch 100% match) ---
    # Transcendental / exponential
    "acos", "asin", "cosh", "erfinv", "exp", "expm1",
    "log", "log10", "log2", "log1p", "reciprocal", "rsqrt",
    "sinh", "tan",
    # Power (all variants: Tensor_Scalar, Tensor_Tensor, Scalar)
    "pow",
    # Activation
    "softplus",
    # Normalization (PyTorch: layer_norm, native_layer_norm, group_norm)
    "layer_norm", "native_layer_norm", "group_norm", "native_group_norm",
    # Norms
    "frobenius_norm", "nuclear_norm", "cosine_similarity",
    # Loss functions (rare in inference, included for correctness)
    "poisson_nll_loss", "cosine_embedding_loss", "nll_loss", "nll_loss2d",
    "hinge_embedding_loss", "kl_div", "l1_loss", "smooth_l1_loss",
    "huber_loss", "mse_loss", "margin_ranking_loss",
    "multilabel_margin_loss", "soft_margin_loss", "triplet_margin_loss",
    "multi_margin_loss", "binary_cross_entropy_with_logits",
    # Distance
    "dist", "pdist", "cdist",
    # Complex number ops — EXTENSION (not in PyTorch autocast).
    # Require fp32 because complex32/fp16 doesn't exist on CUDA.
    # Used in RoPE implementations (DeepSeek-V2, etc.)
    "polar", "view_as_complex",
    # Other
    "renorm", "logsumexp",
    # Upsample (all variants)
    "upsample_nearest1d", "_upsample_nearest_exact1d",
    "upsample_nearest2d", "_upsample_nearest_exact2d",
    "upsample_nearest3d", "_upsample_nearest_exact3d",
    "upsample_linear1d", "upsample_bilinear2d", "_upsample_bilinear2d_aa",
    "upsample_trilinear3d", "upsample_bicubic2d", "_upsample_bicubic2d_aa",

    # --- AT_FORALL_FP32_SET_OPT_DTYPE (PyTorch 100% match) ---
    # Reductions (need fp32 accumulation to prevent overflow)
    "prod", "softmax", "_softmax", "log_softmax",
    "cumprod", "cumsum", "sum",
    # Norm variants
    "linalg_vector_norm", "linalg_matrix_norm",

    # --- DEVIATION: bmm (PyTorch classifies as LOWER_PRECISION_FP / fp16) ---
    # We force fp32 for T5-XXL decomposed attention stability.
    # T5 has no 1/sqrt(d_k) scaling → Q@K^T scores overflow fp16 (>65504).
    # Impact audit (8 models):
    #   T5 text_encoders (Flex/PixArt): 48 bmm each — PROTECTED from NaN
    #   Sana transformer: 40 bmm — graph casts to fp32 first, wrapper is no-op
    #   DeepSeek/Janus/Qwen3: 56-97 bmm — seq_len=1 at decode, ~0.01ms cost
    "bmm",
    # NOTE: add/sub are NOT here. See AMP_OVERFLOW_PROTECT_OPS below.
})

# Ops that should run in compute_dtype (fp16/bf16) for performance.
# From AT_FORALL_LOWER_PRECISION_FP.
# Inputs are cast DOWN to compute_dtype; output stays in compute_dtype.
AMP_FP16_OPS: FrozenSet[str] = frozenset({
    # Convolutions (PyTorch 100% match)
    "_convolution", "conv1d", "conv2d", "conv3d", "conv_tbc", "convolution",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    # Matrix multiply (PyTorch 100% match, except bmm — see AMP_FP32_OPS)
    "addmm", "addmv", "addr", "matmul", "einsum",
    "mm", "mv", "addbmm", "baddbmm",
    "linalg_vecdot", "linear", "chain_matmul", "linalg_multi_dot",
    # RNN cells (PyTorch 100% match)
    "_thnn_fused_lstm_cell", "_thnn_fused_gru_cell",
    "lstm_cell", "gru_cell", "rnn_tanh_cell", "rnn_relu_cell",
    # DEVIATION: SDPA ops EXCLUDED (PyTorch classifies as LOWER_PRECISION_FP).
    # compiled_ops._make_attention() calls F.sdpa directly.
    # DtypeEngine must not double-wrap SDPA inputs.
    # Activation (PyTorch 100% match)
    "prelu",
})

# Ops that promote to widest input type.
# From AT_FORALL_PROMOTE. (PyTorch 100% match — all 11 ops)
AMP_PROMOTE_OPS: FrozenSet[str] = frozenset({
    "addcdiv", "addcmul", "atan2", "bilinear", "cross",
    "dot", "vdot", "grid_sampler", "index_put",
    "tensordot", "scatter_add",
})


# EXTENSION: Overflow protection for fp16 residual accumulation.
# NOT in PyTorch autocast — unique to NeuroBrix inference engine.
#
# Unlike AMP_FP32_OPS (which upcast inputs and keep output in fp32, altering
# the precision chain), these ops run at native dtype and only intervene when
# overflow actually occurs: clamp Inf→±65504 AFTER computation.
#
# Why not just put add/sub in AMP_FP32_OPS?
# Because AMP_FP32_OPS changes the OUTPUT dtype to fp32, altering the entire
# downstream precision chain. For models that DON'T overflow (Janus, DeepSeek,
# Sana), this would force unnecessary fp32 computation in subsequent ops.
#
# Concrete case: T5-XXL (Flex text_encoder_2) residual accumulation across 24
# encoder layers reaches ±65504 (fp16 max) by layer ~20. The next residual add
# overflows → Inf → rms_norm(Inf) → NaN. Clamping at 65504 preserves the
# direction (post-rms_norm normalization is correct) while preventing NaN.
#
# Cost: ~0.01ms per op (GPU-only clamp, no sync). All hardware.
AMP_OVERFLOW_PROTECT_OPS: FrozenSet[str] = frozenset({
    "add", "sub",
})


# NOTE: _MATMUL_OPS and _upcast_matmul were removed (Feb 2026 audit).
# The feature (fp32 upcast for bf16→fp16 matmul on V100) was planned but never
# wired into compile_op(). All 8 models pass regression tests without matmul
# upcast. If bf16→fp16 matmul precision becomes an issue for future models,
# implement in compile_op() with _MATMUL_OPS check.


def _safe_downcast(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Cast tensor to target_dtype with overflow clamping if needed."""
    if not tensor.is_floating_point():
        return tensor
    max_val = _DTYPE_MAX.get(target_dtype)
    if max_val is not None and tensor.dtype != target_dtype:
        tensor = tensor.clamp(-max_val, max_val)
    return tensor.to(target_dtype)


# Dtype string to torch.dtype mapping
# Supports both "torch.float32" (from attrs.kwargs) and "float32" (from output_dtypes)
_DTYPE_MAP = {
    "torch.float32": torch.float32, "float32": torch.float32,
    "torch.float16": torch.float16, "float16": torch.float16,
    "torch.bfloat16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "torch.float64": torch.float64, "float64": torch.float64,
    "torch.int32": torch.int32, "int32": torch.int32,
    "torch.int64": torch.int64, "int64": torch.int64,
    "torch.int16": torch.int16, "int16": torch.int16,
    "torch.int8": torch.int8, "int8": torch.int8,
    "torch.uint8": torch.uint8, "uint8": torch.uint8,
    "torch.bool": torch.bool, "bool": torch.bool,
}


class DtypeEngine:
    """
    AMP-driven dtype engine.

    Prism decides compute_dtype. This engine applies PyTorch AMP autocast rules:
    1. FP32 ops: upcast inputs to fp32 (pow, rsqrt, softmax, sum, ...)
    2. FP16 ops: cast inputs to compute_dtype (matmul, conv, ...)
    3. Promote ops: promote to widest input dtype
    4. _to_copy: Prism-driven remapping
    5. Constants: convert to compute_dtype

    Output behavior matches PyTorch autocast:
    - FP32 ops output fp32 → dtype propagates until next FP16 op
    - FP16 ops output compute_dtype → brings the chain back to half-precision
    """

    def __init__(self, compute_dtype: Optional[torch.dtype], graph_dtype: Optional[torch.dtype] = None):
        self.compute_dtype = compute_dtype
        self.graph_dtype = graph_dtype

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def compile_op(self, op_type: str, func: Optional[Callable], attrs: Dict[str, Any]) -> Callable:
        """
        Compile-time entry point. Wraps ops with AMP autocast rules.

        Priority:
        1. _to_copy: always handled (dtype remapping)
        2. AMP FP32 ops: upcast inputs to fp32
        3. AMP FP16 ops: cast inputs to compute_dtype + inf-fix
        4. AMP Promote ops: promote to widest input dtype
        5. Everything else: pass through unchanged
        """
        op_name = op_type[6:] if op_type.startswith("aten::") else op_type

        # _to_copy always handled (required for dtype conversion)
        if op_type == "aten::_to_copy":
            return self._make_to_copy(attrs)

        assert func is not None, f"ZERO FALLBACK: func cannot be None for op {op_name}"

        # AMP rules only apply when compute_dtype is half-precision
        if self.compute_dtype in (torch.float16, torch.bfloat16):
            if op_name in AMP_FP32_OPS:
                # _softmax/_log_softmax have half_to_float that crashes if input
                # is already fp32 ("conversion is supported for Half type only").
                # Wrap with a guard that disables half_to_float when input is fp32.
                if op_name in ("_softmax", "_log_softmax"):
                    return self._make_safe_softmax(func)
                return self._make_fp32_wrapper(func)

            if op_name in AMP_FP16_OPS:
                return self._make_lower_precision_wrapper(func)

            if op_name in AMP_PROMOTE_OPS:
                return self._make_promote_wrapper(func)

            if op_name in AMP_OVERFLOW_PROTECT_OPS:
                return self._make_overflow_protect_wrapper(func)

        return func

    # ========================================================================
    # AMP WRAPPERS
    # ========================================================================

    def _make_safe_softmax(self, func: Callable) -> Callable:
        """
        Safe wrapper for _softmax/_log_softmax.

        These ops have a half_to_float parameter that handles fp16→fp32
        conversion internally. If input is already fp32 (e.g. from upstream
        bmm FP32 upcast), half_to_float=True would crash with:
            "conversion is supported for Half type only"

        This wrapper disables half_to_float when input is already fp32,
        and upcasts fp16 inputs when half_to_float=False.
        """
        def safe_softmax(inp, dim, half_to_float=False):
            if inp.dtype == torch.float32:
                # Already fp32 — disable half_to_float to prevent crash
                return func(inp, dim, False)
            if inp.dtype in (torch.float16, torch.bfloat16) and not half_to_float:
                # Half-precision input without auto-conversion — upcast manually
                return func(inp.float(), dim, False)
            # Standard path: fp16 input with half_to_float=True
            return func(inp, dim, half_to_float)
        return safe_softmax

    def _make_fp32_wrapper(self, func: Callable) -> Callable:
        """
        AMP FP32: Upcast float inputs to fp32 for numerical stability.

        Matches PyTorch autocast behavior for ops in AT_FORALL_FP32 and
        AT_FORALL_FP32_SET_OPT_DTYPE. These ops need fp32 precision to
        avoid overflow/underflow in half-precision.

        Output stays in fp32. Downstream FP16 ops (matmul, conv) will
        cast back to compute_dtype, creating the mixed-precision chain:

            pow(fp32) → mean(fp32) → rsqrt(fp32) → mul(fp32) → mm(fp16)

        Note: .contiguous() is required because .float()/.to() preserve
        channels_last format from upstream conv/upsample ops, but ops like
        native_group_norm require standard contiguous layout.
        """
        def fp32_func(*args, **kwargs):
            new_args = tuple(
                a.float().contiguous() if isinstance(a, torch.Tensor) and a.is_floating_point()
                    and a.dtype != torch.float32
                else (a.contiguous() if isinstance(a, torch.Tensor) and not a.is_contiguous() else a)
                for a in args
            )
            return func(*new_args, **kwargs)
        return fp32_func

    def _make_lower_precision_wrapper(self, func: Callable) -> Callable:
        """
        AMP FP16: Cast float inputs to compute_dtype for performance.

        Matches PyTorch autocast behavior for ops in AT_FORALL_LOWER_PRECISION_FP.
        These are compute-heavy ops (matmul, conv) that benefit from half-precision.

        For fp16 compute_dtype, applies overflow clamp to handle fp16 limits.
        Uses GPU-only clamp (no sync) instead of isinf().any() check.
        """
        assert self.compute_dtype is not None  # Guaranteed by caller check
        compute: torch.dtype = self.compute_dtype
        needs_inf_fix = (compute == torch.float16)

        def lower_precision_func(*args, **kwargs):
            new_args = tuple(
                _safe_downcast(a, compute) if isinstance(a, torch.Tensor) and a.is_floating_point()
                    and a.dtype != compute
                else a
                for a in args
            )
            result = func(*new_args, **kwargs)
            if needs_inf_fix and isinstance(result, torch.Tensor) and result.dtype == torch.float16:
                # GPU-only clamp: no sync (replaces isinf().any() which forced GPU→CPU sync)
                # Clamps ±inf to ±65504 while preserving NaN for error detection
                result = result.clamp(-65504.0, 65504.0)
            return result
        return lower_precision_func

    def _make_promote_wrapper(self, func: Callable) -> Callable:
        """
        AMP Promote: Promote to widest input type.

        Matches PyTorch autocast behavior for ops in AT_FORALL_PROMOTE.
        If all inputs are fp16, runs in fp16. If any input is fp32,
        promotes all to fp32.
        """
        def promote_func(*args, **kwargs):
            max_size = 0
            max_dtype = None
            for a in args:
                if isinstance(a, torch.Tensor) and a.is_floating_point():
                    if a.element_size() > max_size:
                        max_size = a.element_size()
                        max_dtype = a.dtype

            if max_dtype is not None and max_size > 2:
                new_args = tuple(
                    a.to(max_dtype) if isinstance(a, torch.Tensor) and a.is_floating_point()
                        and a.dtype != max_dtype
                    else a
                    for a in args
                )
                return func(*new_args, **kwargs)
            return func(*args, **kwargs)
        return promote_func

    def _make_overflow_protect_wrapper(self, func: Callable) -> Callable:
        """
        Overflow protection: clamp Inf→fp16_max AFTER computation.

        Unlike AMP_FP32 (which upcasts and keeps output in fp32, altering the
        entire downstream precision chain), this wrapper preserves the native
        dtype. It only intervenes when overflow actually occurs (Inf in fp16).

        This prevents the chain: fp16 add overflow → Inf → rms_norm(Inf) → NaN,
        which kills T5-XXL after ~20 layers of residual accumulation.

        Uses GPU-only clamp (no sync) — negligible cost (~0.01ms per op).
        """
        def overflow_safe(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, torch.Tensor) and result.dtype == torch.float16:
                # GPU-only clamp: no sync (replaces isinf().any() which forced GPU→CPU sync)
                result = result.clamp(-65504.0, 65504.0)
            return result
        return overflow_safe

    # ========================================================================
    # CONSTANT CONVERSION
    # ========================================================================

    def convert_constant(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a graph-embedded constant tensor to compute_dtype.

        Only converts constants whose dtype matches graph_dtype (the graph's
        "default" precision). Constants in a different dtype (e.g. fp32 inv_freq
        in an fp16 graph) are intentional and feed fp32 computation chains
        (RoPE, etc.) — these must be preserved.
        """
        if not tensor.is_floating_point():
            return tensor
        if self.compute_dtype is None:
            return tensor
        if tensor.dtype == self.compute_dtype:
            return tensor
        # Only convert if the constant matches graph_dtype
        if self.graph_dtype is not None and tensor.dtype != self.graph_dtype:
            return tensor  # Intentional different-precision constant
        return _safe_downcast(tensor, self.compute_dtype)

    # ========================================================================
    # INTERNAL — _to_copy compilation
    # ========================================================================

    def _make_to_copy(self, attrs: Dict[str, Any]) -> Callable:
        """
        Create _to_copy function with Prism override.

        PRISM OVERRIDE:
        - fp16/bf16 targets → remapped to compute_dtype
        - fp32 targets → PRESERVED (numerical stability: RMSNorm, RoPE, MoE routing)
        - Non-float targets (int64, bool) → preserved as-is
        """
        output_dtypes = attrs.get("output_dtypes", [])
        target_dtype_str = output_dtypes[0] if output_dtypes else None
        target_dtype = _DTYPE_MAP.get(target_dtype_str) if target_dtype_str else None

        # Prism override: remap fp16/bf16 targets, NEVER fp32
        if target_dtype in (torch.float16, torch.bfloat16):
            if self.compute_dtype is not None and target_dtype != self.compute_dtype:
                target_dtype = self.compute_dtype

        if target_dtype is not None:
            def to_copy_with_dtype(inp, **_kwargs):
                if inp is None:
                    return None
                if not isinstance(inp, torch.Tensor):
                    return inp
                # NEVER cast complex → real (discards imaginary part, corrupts RoPE)
                if inp.is_complex() and not target_dtype.is_complex:
                    return inp
                if inp.is_floating_point() and target_dtype in (torch.float16, torch.bfloat16):
                    return _safe_downcast(inp, target_dtype)
                return inp.to(target_dtype)
            return to_copy_with_dtype

        # Fallback: no explicit dtype from graph — use kwargs
        prism_dtype = self.compute_dtype
        def to_copy_passthrough(inp, **kwargs):
            if inp is None:
                return None
            if not isinstance(inp, torch.Tensor):
                return inp
            dtype = kwargs.get('dtype')
            if dtype is not None:
                # Same rule: only remap fp16/bf16, never fp32
                if prism_dtype is not None and dtype in (torch.float16, torch.bfloat16) and dtype != prism_dtype:
                    dtype = prism_dtype
                # NEVER cast complex → real (discards imaginary part, corrupts RoPE)
                if inp.is_complex() and not dtype.is_complex:
                    return inp
                if inp.is_floating_point() and dtype in (torch.float16, torch.bfloat16):
                    return _safe_downcast(inp, dtype)
                return inp.to(dtype)
            return inp
        return to_copy_passthrough
