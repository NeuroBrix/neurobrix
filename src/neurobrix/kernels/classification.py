"""
ATen Operations Execution Classification.

Classifies ATen ops by their execution strategy:
- TRITON: Pure Triton kernel (exists in kernels/ops/ and dispatch.py)
- METADATA: No compute kernel needed (shape/view/memory/creation ops)

Auto-aligned with dispatch.py — any op in dispatch is TRITON.
"""

from enum import Enum, auto
from typing import Dict


class OpExecution(Enum):
    """Classification of how an ATen op is executed."""

    TRITON = auto()     # Pure Triton kernel
    METADATA = auto()   # PyTorch native (shape/view ops) or --triton metadata ops


# ============================================================================
# TRITON COMPUTE OPS — all have kernels in dispatch.py
# ============================================================================

_TRITON_OPS = {
    # Activations (14)
    "relu", "silu", "gelu", "sigmoid", "tanh", "hardsigmoid", "hardswish",
    "leaky_relu", "elu", "mish", "selu", "softplus", "celu", "log_sigmoid",

    # Unary math (16)
    "neg", "exp", "exp2", "sin", "cos", "tan", "rsqrt", "sqrt", "abs", "log",
    "reciprocal", "pow", "clamp", "clamp_min", "erf", "threshold",

    # Binary element-wise (13)
    "add", "mul", "div", "sub", "rsub", "where", "maximum", "minimum",
    "masked_fill", "glu", "lerp", "addcdiv", "addcmul",

    # Comparisons (7)
    "gt", "ge", "lt", "le", "eq", "ne",

    # Bitwise (5)
    "bitwise_and", "bitwise_or", "bitwise_not", "bitwise_left_shift", "bitwise_right_shift",

    # Logical (4)
    "logical_and", "logical_or", "logical_not", "logical_xor",

    # Boolean/check (3)
    "isfinite", "isinf", "isnan",

    # Utility (1)
    "nan_to_num",

    # Normalization (5)
    "native_layer_norm", "rms_norm", "native_group_norm", "group_norm", "batch_norm",

    # Softmax (5)
    "softmax", "_softmax", "_safe_softmax", "log_softmax", "_log_softmax",

    # Matmul/conv (9)
    "mm", "bmm", "addmm", "conv2d", "conv1d", "convolution",
    "baddbmm", "addmv", "conv_depthwise2d",

    # Vector ops (2)
    "dot", "mv",

    # Embedding (1)
    "embedding",

    # Reductions (15)
    "mean", "sum", "amax", "argmax", "argmin", "cumsum", "prod",
    "min", "max", "all", "any", "std", "var", "topk", "sort",

    # Triangular (2)
    "triu", "tril",

    # Spatial (6)
    "upsample_nearest2d", "upsample_nearest3d", "upsample_bilinear2d",
    "avg_pool2d", "max_pool2d", "adaptive_avg_pool2d",

    # Pixel shuffle (2)
    "pixel_shuffle", "pixel_unshuffle",

    # Indexing (6)
    "index_select", "gather", "scatter", "scatter_add", "scatter_reduce", "index_add",

    # Dropout (2)
    "dropout", "native_dropout",

    # Loss (3)
    "cross_entropy_loss", "mse_loss", "nll_loss",

    # Attention (1)
    "scaled_dot_product_attention",

    # Linear algebra (4)
    "linalg_vector_norm", "_weight_norm", "addr", "trace",

    # RoPE (1)
    "rope",

    # Audio/Video specific (3)
    "_weight_norm_interface", "repeat_interleave", "flip",
}

# ============================================================================
# METADATA OPS — no compute kernel needed
# ============================================================================

_METADATA_OPS = {
    # View/reshape (pointer math)
    "view", "_unsafe_view", "reshape", "flatten", "squeeze", "unsqueeze",
    "permute", "transpose", "t", "contiguous", "expand", "expand_as",
    "repeat", "narrow", "slice", "select", "split", "chunk", "unbind",

    # Concatenation (memory copy)
    "cat", "stack",

    # Tensor creation
    "zeros", "ones", "empty", "full", "scalar_tensor",
    "zeros_like", "ones_like", "empty_like", "full_like",
    "arange", "linspace",

    # Type conversion
    "to", "_to_copy", "type_as", "float", "half", "bfloat16", "int", "long",

    # Clone/copy/alias
    "clone", "copy", "detach", "alias", "lift_fresh",

    # Shape queries
    "size", "dim", "numel", "stride", "is_contiguous",

    # Attention (handled by special handler in compiled_ops.py)
    "_scaled_dot_product_attention", "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_flash_attention", "_scaled_dot_product_flash_attention_for_cpu",
    "multi_head_attention_forward",

    # Padding (handled by special handler or PyTorch)
    "pad", "constant_pad_nd",
    "reflection_pad1d", "reflection_pad2d",
    "replication_pad1d", "replication_pad2d",
}


# Build the classification dict
ATEN_CLASSIFICATION: Dict[str, OpExecution] = {}
for op in _TRITON_OPS:
    ATEN_CLASSIFICATION[f"aten::{op}"] = OpExecution.TRITON
# Also classify custom ops
ATEN_CLASSIFICATION["custom::rms_norm"] = OpExecution.TRITON
ATEN_CLASSIFICATION["custom::swiglu_fused"] = OpExecution.TRITON
ATEN_CLASSIFICATION["custom::rope_fused"] = OpExecution.TRITON
for op in _METADATA_OPS:
    ATEN_CLASSIFICATION[f"aten::{op}"] = OpExecution.METADATA


def get_execution_type(op_name: str) -> OpExecution:
    """Get the execution type for an ATen op."""
    if not op_name.startswith(("aten::", "custom::")):
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
