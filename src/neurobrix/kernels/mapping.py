"""
ATen to Kernel Operation Mapping.

Maps ATen operation names to their corresponding kernel operation names.
Used by the runtime to resolve and execute the appropriate Triton kernel.

The actual kernel files are discovered at runtime via the kernels.resolver module
which implements the tiers cascade (e.g., hopper -> ampere -> volta -> common).
"""

from typing import Dict, Optional, Callable, Any, List

from .classification import OpExecution, ATEN_CLASSIFICATION


# ATen op name -> kernel operation name (filename without .py)
# STANDARD: Snake Case
ATEN_TO_KERNEL: Dict[str, str] = {
    # Matrix Operations
    "aten::mm": "mm",
    "aten::bmm": "bmm",
    "aten::addmm": "mm",
    "aten::linear": "mm",
    "aten::matmul": "mm", 

    # Normalization
    "aten::layer_norm": "layernorm",
    "aten::native_layer_norm": "layernorm",
    "aten::group_norm": "groupnorm",
    "aten::native_group_norm": "groupnorm",
    "aten::rms_norm": "rms_norm",
    "aten::batch_norm": "batch_norm",
    "aten::instance_norm": "instance_norm",

    # Activations
    "aten::gelu": "gelu",
    "aten::gelu_": "gelu",
    "aten::silu": "silu",
    "aten::silu_": "silu",
    "aten::relu": "relu",
    "aten::relu_": "relu",
    "aten::sigmoid": "sigmoid",
    "aten::sigmoid_": "sigmoid",
    "aten::tanh": "tanh",
    "aten::tanh_": "tanh",
    "aten::softmax": "softmax",
    "aten::_softmax": "softmax",
    "aten::log_softmax": "log_softmax",
    "aten::leaky_relu": "leaky_relu",
    "aten::elu": "elu",

    # Element-wise Binary
    "aten::add": "add",
    "aten::add_": "add",
    "aten::sub": "sub",
    "aten::sub_": "sub",
    "aten::rsub": "sub",
    "aten::mul": "mul",
    "aten::mul_": "mul",
    "aten::div": "div",
    "aten::div_": "div",
    "aten::pow": "pow",
    "aten::pow_": "pow",
    "aten::maximum": "maximum",
    "aten::minimum": "minimum",
    "aten::eq": "eq",
    "aten::ne": "ne",
    "aten::lt": "lt",
    "aten::le": "le",
    "aten::gt": "gt",
    "aten::ge": "ge",
    "aten::where": "where",
    "aten::clamp": "clamp",

    # Element-wise Unary
    "aten::abs": "abs",
    "aten::neg": "neg",
    "aten::exp": "exp",
    "aten::log": "log",
    "aten::sqrt": "sqrt",
    "aten::rsqrt": "rsqrt",
    "aten::sin": "sin",
    "aten::cos": "cos",
    "aten::erf": "erf",
    "aten::reciprocal": "reciprocal",
    "aten::ceil": "ceil",
    "aten::floor": "floor",

    # Reductions
    "aten::sum": "reduce_sum",
    "aten::mean": "reduce_mean",
    "aten::max": "reduce_max",
    "aten::min": "reduce_min",
    "aten::prod": "reduce_prod",
    "aten::var": "var_mean",
    "aten::std": "std",
    "aten::argmax": "argmax",
    "aten::argmin": "argmin",
    "aten::any": "any",
    "aten::all": "all",
    "aten::cumsum": "cumsum",

    # Access / Shape
    "aten::embedding": "embedding",
    "aten::gather": "gather",
    "aten::scatter": "scatter",
    "aten::upsample_nearest2d": "resize",
    "aten::upsample_bilinear2d": "resize",
    "aten::interpolate": "resize",
    "aten::index_select": "index_select",
    "aten::masked_fill": "masked_fill",

    # Attention
    "aten::scaled_dot_product_attention": "scaled_dot_product_attention",
    "aten::_scaled_dot_product_attention": "scaled_dot_product_attention",
    "aten::_scaled_dot_product_flash_attention_for_cpu": "scaled_dot_product_attention",
    
    # Conv
    "aten::conv2d": "conv2d",
    "aten::convolution": "conv2d",
    "aten::conv1d": "conv1d",
    "aten::conv3d": "conv3d",
    "aten::conv_depthwise2d": "conv_depthwise2d",
}


def get_kernel_op_name(aten_op: str) -> Optional[str]:
    """Get the kernel operation name for an ATen op."""
    if not aten_op.startswith("aten::"):
        aten_op = f"aten::{aten_op}"

    if aten_op in ATEN_CLASSIFICATION:
        if ATEN_CLASSIFICATION[aten_op] == OpExecution.METADATA:
            return None

    return ATEN_TO_KERNEL.get(aten_op)


def run_aten_op(aten_op: str, hw: Any, family: str, *tensors, **attrs) -> Any:
    """Resolve and execute an ATen op via its Triton kernel."""
    kernel = get_kernel(aten_op, hw, family)
    if kernel is None:
        raise ValueError(f"Cannot execute METADATA op '{aten_op}' via kernel.")
    return kernel(*tensors, **attrs)


def list_supported_ops() -> Dict[str, OpExecution]:
    return dict(ATEN_CLASSIFICATION)


def list_triton_ops() -> list:
    return [op for op, exec_type in ATEN_CLASSIFICATION.items() if exec_type == OpExecution.TRITON]


def list_metadata_ops() -> list:
    return [op for op, exec_type in ATEN_CLASSIFICATION.items() if exec_type == OpExecution.METADATA]