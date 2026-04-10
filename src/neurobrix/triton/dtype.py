"""Triton DtypeEngine — AMP rules for triton mode. Zero torch dependency.

Same AMP logic as core/dtype/engine.py but uses NBXDtype instead of torch.dtype.
Wraps op functions with input casting for numerical stability.

Rules (from PyTorch AT_FORALL_FP32 / AT_FORALL_LOWER_PRECISION_FP):
  - FP32 ops: upcast inputs to fp32 (pow, rsqrt, softmax, layernorm, ...)
  - FP16 ops: cast inputs to compute_dtype (mm, conv, bmm, ...)
  - FP16_NEED_FP32: on fp16 hardware, mm/bmm/div/addmm need fp32
  - Promote ops: promote to widest input dtype
"""

from typing import Callable, FrozenSet

from neurobrix.kernels.nbx_tensor import NBXDtype


# ============================================================================
# AMP OP SETS — identical to core/dtype/engine.py
# ============================================================================

AMP_FP32_OPS: FrozenSet[str] = frozenset({
    "acos", "asin", "cosh", "erfinv", "exp", "expm1",
    "log", "log10", "log2", "log1p", "reciprocal", "rsqrt",
    "sinh", "tan", "pow", "softplus",
    "layer_norm", "native_layer_norm", "group_norm", "native_group_norm",
    "batch_norm", "native_batch_norm", "cudnn_batch_norm", "instance_norm",
    "frobenius_norm", "nuclear_norm", "cosine_similarity",
    "poisson_nll_loss", "cosine_embedding_loss", "nll_loss",
    "mse_loss", "smooth_l1_loss", "huber_loss",
    "polar", "view_as_complex",
    "renorm", "logsumexp",
    "upsample_nearest1d", "upsample_nearest2d", "upsample_nearest3d",
    "upsample_linear1d", "upsample_bilinear2d", "upsample_bicubic2d",
    "prod", "softmax", "_softmax", "log_softmax",
    "cumprod", "cumsum", "sum",
    "linalg_vector_norm", "linalg_matrix_norm",
})

AMP_FP16_OPS: FrozenSet[str] = frozenset({
    "_convolution", "conv1d", "conv2d", "conv3d", "convolution",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "addmm", "addmv", "addr", "matmul", "einsum",
    "mm", "mv", "bmm", "addbmm", "baddbmm",
    "linear", "prelu", "div",
})

# Triton kernels accumulate in fp32 internally (matmul_kernel line 43,
# fused_moe_kernel line 106, flash_attention line 78). Unlike cuBLAS
# (which needs allow_fp16_reduced_precision_reduction=False), Triton
# kernels ALWAYS use fp32 accumulators. So mm/bmm/addmm don't need
# input upcast to fp32 — fp16 in, fp32 accum, fp16 out.
# Keeping div for epsilon underflow (1e-15 → 0 in fp16).
_FP16_NEED_FP32: FrozenSet[str] = frozenset({"div"})

AMP_PROMOTE_OPS: FrozenSet[str] = frozenset({
    "addcdiv", "addcmul", "atan2", "bilinear", "cross",
    "dot", "vdot", "grid_sampler", "index_put",
    "scatter_add", "tensordot", "linalg_cross",
})

_FLOATING = frozenset({NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32, NBXDtype.float64})


def _is_float_tensor(a) -> bool:
    """Check if a is a floating-point tensor (NBXTensor or duck-typed)."""
    if not hasattr(a, 'is_floating_point'):
        return False
    return a.is_floating_point()


def _get_nbx_dtype(a) -> NBXDtype:
    """Get NBXDtype from tensor."""
    if hasattr(a, 'nbx_dtype'):
        return a.nbx_dtype
    return NBXDtype.float32


# ============================================================================
# TRITON DTYPE ENGINE
# ============================================================================

class TritonDtypeEngine:
    """AMP-driven dtype engine for triton mode. Zero torch dependency.

    Same logic as core/dtype/engine.py DtypeEngine but operates on
    NBXDtype instead of torch.dtype.
    """

    def __init__(self, compute_dtype: NBXDtype):
        self.compute_dtype = compute_dtype

    def wrap_op(self, op_name: str, func: Callable) -> Callable:
        """Wrap an op function with AMP casting rules.

        Args:
            op_name: bare op name (e.g., "mm", "pow", "add")
            func: the raw kernel wrapper function

        Returns:
            Wrapped function with dtype casting applied
        """
        # AMP only applies for half-precision compute
        if self.compute_dtype not in (NBXDtype.float16, NBXDtype.bfloat16):
            return func

        if op_name in AMP_FP32_OPS:
            return self._wrap_fp32(func)

        if op_name in AMP_FP16_OPS:
            if self.compute_dtype == NBXDtype.float16 and op_name in _FP16_NEED_FP32:
                return self._wrap_fp32(func)
            return self._wrap_lower_precision(func)

        if op_name in AMP_PROMOTE_OPS:
            return self._wrap_promote(func)

        return func

    def _wrap_fp32(self, func: Callable) -> Callable:
        """Upcast float inputs to fp32."""
        def fp32_func(*args, **kwargs):
            new_args = tuple(
                a.to(NBXDtype.float32).contiguous()
                    if _is_float_tensor(a) and _get_nbx_dtype(a) != NBXDtype.float32
                else (a.contiguous() if hasattr(a, 'contiguous') and hasattr(a, 'is_contiguous') and not a.is_contiguous() else a)
                for a in args
            )
            return func(*new_args, **kwargs)
        return fp32_func

    def _wrap_lower_precision(self, func: Callable) -> Callable:
        """Cast float inputs to compute_dtype."""
        compute = self.compute_dtype
        def lower_func(*args, **kwargs):
            new_args = tuple(
                a.to(compute) if _is_float_tensor(a) and _get_nbx_dtype(a) != compute
                else a
                for a in args
            )
            return func(*new_args, **kwargs)
        return lower_func

    def _wrap_promote(self, func: Callable) -> Callable:
        """Promote to widest input dtype."""
        def promote_func(*args, **kwargs):
            max_size = 0
            max_dtype = None
            for a in args:
                if _is_float_tensor(a):
                    esz = a.element_size()
                    if esz > max_size:
                        max_size = esz
                        max_dtype = _get_nbx_dtype(a)

            if max_dtype is not None and max_size > 2:
                new_args = tuple(
                    a.to(max_dtype) if _is_float_tensor(a) and _get_nbx_dtype(a) != max_dtype
                    else a
                    for a in args
                )
                return func(*new_args, **kwargs)
            return func(*args, **kwargs)
        return promote_func
