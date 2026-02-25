"""
ATen Operations Execution Classification.

Classifies ATen ops by their execution strategy:
- TRITON: Pure Triton kernel
- METADATA: PyTorch native ops (no custom kernel needed)
"""

from enum import Enum, auto
from typing import Dict


class OpExecution(Enum):
    """Classification of how an ATen op is executed."""

    TRITON = auto()     # Pure Triton kernel
    METADATA = auto()   # PyTorch native (shape/view ops, optimized ops)


# ATen op name -> execution type
# Covers all ops from PixArt-Sigma, Flux, SDXL, LLMs
#
# Classification categories:
# - TRITON: Use custom Triton kernel (must exist in kernels/ops/)
# - METADATA: Use PyTorch native (for optimized ops, view ops, or debugging)
#
ATEN_CLASSIFICATION: Dict[str, OpExecution] = {
    # =========================================================================
    # COMPUTE OPS - METADATA (PyTorch native)
    #
    # TRITON KERNEL TESTING (January 2026):
    # - Unit tests: mm, bmm, add, mul kernels produce IDENTICAL results to PyTorch
    # - Overflow handling: Triton kernels PREVENT inf by clamping (PyTorch overflows)
    # - Individual kernels work correctly with fp32 intermediate + clamp pattern
    #
    # FULL PIPELINE ISSUE:
    # - When many ops classified as TRITON, adapter dispatch causes divergence
    # - Many TRITON ops lack kernel mappings (softplus, pooling, loss functions)
    # - Keep as METADATA for stability until adapter issues are resolved
    #
    # TODO: Fix adapter dispatch to properly handle missing kernel fallback
    # =========================================================================

    # Matrix operations - METADATA (PyTorch cuBLAS is optimal)
    "aten::mm": OpExecution.TRITON,  # TEST: mm only
    "aten::bmm": OpExecution.METADATA,
    "aten::addmm": OpExecution.METADATA,
    "aten::linear": OpExecution.METADATA,
    "aten::matmul": OpExecution.METADATA,

    # Element-wise binary - TRITON (Adapter V2 handles allocation)
    "aten::add": OpExecution.TRITON,
    "aten::sub": OpExecution.TRITON,
    "aten::rsub": OpExecution.TRITON,
    "aten::mul": OpExecution.TRITON,
    "aten::div": OpExecution.TRITON,
    "aten::pow": OpExecution.TRITON,
    "aten::maximum": OpExecution.TRITON,
    "aten::minimum": OpExecution.TRITON,

    # Element-wise unary - TRITON (Adapter V2 handles allocation)
    "aten::abs": OpExecution.TRITON,
    "aten::neg": OpExecution.TRITON,
    "aten::exp": OpExecution.TRITON,
    "aten::log": OpExecution.TRITON,
    "aten::sqrt": OpExecution.TRITON,
    "aten::rsqrt": OpExecution.TRITON,
    "aten::sin": OpExecution.TRITON,
    "aten::cos": OpExecution.TRITON,
    "aten::erf": OpExecution.TRITON,
    "aten::reciprocal": OpExecution.TRITON,
    "aten::isinf": OpExecution.METADATA,
    "aten::isnan": OpExecution.METADATA,
    "aten::isfinite": OpExecution.METADATA,

    # =========================================================================
    # COMPUTE OPS - METADATA (PyTorch/cuDNN highly optimized, no benefit to Triton)
    # =========================================================================

    # Normalization - PyTorch handles these optimally
    "aten::layer_norm": OpExecution.METADATA,
    "aten::native_layer_norm": OpExecution.METADATA,
    "aten::group_norm": OpExecution.METADATA,
    "aten::native_group_norm": OpExecution.METADATA,
    "aten::rms_norm": OpExecution.TRITON,  # Simple enough for Triton
    "aten::batch_norm": OpExecution.TRITON,
    "aten::instance_norm": OpExecution.TRITON,

    # Activations - TRITON (Adapter V2 handles allocation)
    "aten::gelu": OpExecution.TRITON,
    "aten::silu": OpExecution.TRITON,
    "aten::relu": OpExecution.TRITON,
    "aten::sigmoid": OpExecution.TRITON,
    "aten::tanh": OpExecution.TRITON,
    "aten::softmax": OpExecution.METADATA,
    "aten::_softmax": OpExecution.METADATA,
    "aten::log_softmax": OpExecution.TRITON,
    "aten::softplus": OpExecution.TRITON,
    "aten::mish": OpExecution.TRITON,
    "aten::leaky_relu": OpExecution.TRITON,
    "aten::elu": OpExecution.TRITON,
    "aten::hardswish": OpExecution.TRITON,
    "aten::hardsigmoid": OpExecution.TRITON,

    # Reductions - PyTorch highly optimized
    "aten::sum": OpExecution.METADATA,
    "aten::mean": OpExecution.METADATA,
    "aten::max": OpExecution.METADATA,
    "aten::min": OpExecution.METADATA,
    "aten::prod": OpExecution.METADATA,
    "aten::var": OpExecution.METADATA,
    "aten::std": OpExecution.METADATA,
    "aten::argmax": OpExecution.METADATA,
    "aten::argmin": OpExecution.METADATA,
    "aten::any": OpExecution.METADATA,
    "aten::all": OpExecution.METADATA,
    "aten::cumsum": OpExecution.METADATA,
    "aten::cumprod": OpExecution.METADATA,

    # Comparison - PyTorch handles these well
    "aten::eq": OpExecution.METADATA,
    "aten::ne": OpExecution.METADATA,
    "aten::lt": OpExecution.METADATA,
    "aten::le": OpExecution.METADATA,
    "aten::gt": OpExecution.METADATA,
    "aten::ge": OpExecution.METADATA,
    "aten::where": OpExecution.METADATA,
    "aten::clamp": OpExecution.METADATA,

    # Dropout - inference is identity (no compute)
    "aten::dropout": OpExecution.METADATA,
    "aten::native_dropout": OpExecution.METADATA,

    # Embedding - gather from embedding table, PyTorch optimized
    "aten::embedding": OpExecution.METADATA,

    # =========================================================================
    # METADATA: Complex ops - PyTorch handles these optimally
    # =========================================================================

    # Attention - METADATA (PyTorch's Flash Attention / cuDNN is optimal)
    "aten::scaled_dot_product_attention": OpExecution.METADATA,
    "aten::_scaled_dot_product_attention": OpExecution.METADATA,
    "aten::_scaled_dot_product_efficient_attention": OpExecution.METADATA,
    "aten::_scaled_dot_product_flash_attention": OpExecution.METADATA,
    "aten::_scaled_dot_product_flash_attention_for_cpu": OpExecution.METADATA,
    "aten::multi_head_attention_forward": OpExecution.METADATA,

    # Convolutions - METADATA (cuDNN is highly optimized)
    "aten::conv2d": OpExecution.METADATA,
    "aten::conv1d": OpExecution.METADATA,
    "aten::conv_transpose2d": OpExecution.METADATA,
    "aten::depthwise_conv2d": OpExecution.METADATA,
    "aten::convolution": OpExecution.METADATA,

    # Loss functions
    "aten::cross_entropy_loss": OpExecution.TRITON,
    "aten::nll_loss": OpExecution.TRITON,
    "aten::mse_loss": OpExecution.TRITON,
    "aten::l1_loss": OpExecution.TRITON,
    "aten::binary_cross_entropy": OpExecution.TRITON,
    "aten::binary_cross_entropy_with_logits": OpExecution.TRITON,
    "aten::kl_div": OpExecution.TRITON,

    # Pooling
    "aten::max_pool2d": OpExecution.TRITON,
    "aten::avg_pool2d": OpExecution.TRITON,
    "aten::adaptive_avg_pool2d": OpExecution.TRITON,
    "aten::adaptive_max_pool2d": OpExecution.TRITON,

    # Interpolation/Upsampling - METADATA (F.interpolate handles these)
    "aten::upsample_nearest2d": OpExecution.METADATA,
    "aten::upsample_bilinear2d": OpExecution.METADATA,
    "aten::interpolate": OpExecution.METADATA,

    # Padding - METADATA (F.pad handles these)
    "aten::pad": OpExecution.METADATA,
    "aten::constant_pad_nd": OpExecution.METADATA,
    "aten::reflection_pad1d": OpExecution.METADATA,
    "aten::reflection_pad2d": OpExecution.METADATA,
    "aten::replication_pad1d": OpExecution.METADATA,
    "aten::replication_pad2d": OpExecution.METADATA,

    # One-hot encoding - COMPUTE OP
    "aten::one_hot": OpExecution.TRITON,

    # Sorting
    "aten::sort": OpExecution.TRITON,
    "aten::topk": OpExecution.TRITON,
    "aten::argsort": OpExecution.TRITON,

    # Scatter/Gather
    "aten::scatter": OpExecution.TRITON,
    "aten::scatter_add": OpExecution.TRITON,
    "aten::gather": OpExecution.TRITON,
    "aten::index_select": OpExecution.TRITON,
    "aten::index_add": OpExecution.TRITON,

    # =========================================================================
    # METADATA: No compute kernel needed
    # =========================================================================

    # View operations (just pointer math)
    "aten::view": OpExecution.METADATA,
    "aten::_unsafe_view": OpExecution.METADATA,  # PyTorch internal view
    "aten::reshape": OpExecution.METADATA,
    "aten::flatten": OpExecution.METADATA,
    "aten::squeeze": OpExecution.METADATA,
    "aten::unsqueeze": OpExecution.METADATA,
    "aten::permute": OpExecution.METADATA,
    "aten::transpose": OpExecution.METADATA,
    "aten::t": OpExecution.METADATA,
    "aten::contiguous": OpExecution.METADATA,
    "aten::expand": OpExecution.METADATA,
    "aten::expand_as": OpExecution.METADATA,
    "aten::repeat": OpExecution.METADATA,
    "aten::narrow": OpExecution.METADATA,
    "aten::slice": OpExecution.METADATA,
    "aten::select": OpExecution.METADATA,  # Select single element along dim
    "aten::split": OpExecution.METADATA,
    "aten::chunk": OpExecution.METADATA,
    "aten::unbind": OpExecution.METADATA,

    # Concatenation (memory copy, no compute)
    "aten::cat": OpExecution.METADATA,
    "aten::stack": OpExecution.METADATA,

    # Tensor creation
    "aten::zeros": OpExecution.METADATA,
    "aten::ones": OpExecution.METADATA,
    "aten::empty": OpExecution.METADATA,
    "aten::full": OpExecution.METADATA,
    "aten::scalar_tensor": OpExecution.METADATA,  # Create tensor from scalar
    "aten::zeros_like": OpExecution.METADATA,
    "aten::ones_like": OpExecution.METADATA,
    "aten::empty_like": OpExecution.METADATA,
    "aten::full_like": OpExecution.METADATA,
    "aten::arange": OpExecution.METADATA,
    "aten::linspace": OpExecution.METADATA,
    "aten::triu": OpExecution.TRITON,  # COMPUTE: element-wise mask
    "aten::tril": OpExecution.TRITON,  # COMPUTE: element-wise mask
    "aten::masked_fill": OpExecution.TRITON,  # COMPUTE: conditional assignment

    # Type conversion
    "aten::to": OpExecution.METADATA,
    "aten::_to_copy": OpExecution.METADATA,
    "aten::type_as": OpExecution.METADATA,
    "aten::float": OpExecution.METADATA,
    "aten::half": OpExecution.METADATA,
    "aten::bfloat16": OpExecution.METADATA,
    "aten::int": OpExecution.METADATA,
    "aten::long": OpExecution.METADATA,

    # Clone/copy
    "aten::clone": OpExecution.METADATA,
    "aten::copy": OpExecution.METADATA,
    # aten::copy_ removed (normalized to aten::copy)
    "aten::detach": OpExecution.METADATA,
    "aten::alias": OpExecution.METADATA,
    "aten::lift_fresh": OpExecution.METADATA,

    # Shape queries
    "aten::size": OpExecution.METADATA,
    "aten::dim": OpExecution.METADATA,
    "aten::numel": OpExecution.METADATA,
    "aten::stride": OpExecution.METADATA,
    "aten::is_contiguous": OpExecution.METADATA,
}


def get_execution_type(op_name: str) -> OpExecution:
    """
    Get the execution type for an ATen op.

    Args:
        op_name: ATen op name (with or without 'aten::' prefix)

    Returns:
        OpExecution enum value

    Raises:
        KeyError: If op not in classification
    """
    # Normalize: ensure 'aten::' prefix
    if not op_name.startswith("aten::"):
        op_name = f"aten::{op_name}"

    if op_name not in ATEN_CLASSIFICATION:
        raise KeyError(f"Unknown ATen op: {op_name}")

    return ATEN_CLASSIFICATION[op_name]


def is_triton_op(op_name: str) -> bool:
    """Check if op uses pure Triton kernel."""
    try:
        return get_execution_type(op_name) == OpExecution.TRITON
    except KeyError:
        return False


def is_metadata_op(op_name: str) -> bool:
    """Check if op is metadata-only (no kernel needed)."""
    try:
        return get_execution_type(op_name) == OpExecution.METADATA
    except KeyError:
        return False
