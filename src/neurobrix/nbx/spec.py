"""
NBX Format Specification Constants

Universal operation mapping from ONNX to NBX format.
Follows the NBX Format Specification v1.0.
"""

from typing import Optional

# =============================================================================
# NBX FORMAT CONSTANTS
# =============================================================================

NBX_VERSION = "1.0.0"
NBX_MAGIC = b"NBX\x00"  # Magic bytes for .nbx files

# =============================================================================
# ONNX TO NBX OPERATION MAPPING
# =============================================================================

ONNX_TO_NBX_OPS = {
    # ---------------------------------------------------------------------
    # Tensor Operations
    # ---------------------------------------------------------------------
    "MatMul": "matmul",
    "Gemm": "linear",
    "Conv": "conv2d",
    "ConvTranspose": "conv_transpose2d",
    "BatchNormalization": "batch_norm",
    "LayerNormalization": "layer_norm",
    "GroupNormalization": "group_norm",
    "InstanceNormalization": "instance_norm",

    # ---------------------------------------------------------------------
    # Activations
    # ---------------------------------------------------------------------
    "Relu": "relu",
    "Gelu": "gelu",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
    "Softmax": "softmax",
    "LogSoftmax": "log_softmax",
    "LeakyRelu": "leaky_relu",
    "Elu": "elu",
    "Selu": "selu",
    "Silu": "silu",  # Also known as Swish
    "HardSigmoid": "hard_sigmoid",
    "HardSwish": "hard_swish",
    "Mish": "mish",
    "Softplus": "softplus",
    "Softsign": "softsign",
    "PRelu": "prelu",

    # ---------------------------------------------------------------------
    # Pooling
    # ---------------------------------------------------------------------
    "MaxPool": "max_pool2d",
    "AveragePool": "avg_pool2d",
    "GlobalMaxPool": "adaptive_max_pool2d",
    "GlobalAveragePool": "adaptive_avg_pool2d",

    # ---------------------------------------------------------------------
    # Element-wise Operations
    # ---------------------------------------------------------------------
    "Add": "add",
    "Sub": "sub",
    "Mul": "mul",
    "Div": "div",
    "Min": "element_min",  # Element-wise minimum (different from ReduceMin)
    "Max": "element_max",  # Element-wise maximum (different from ReduceMax)
    "Pow": "pow",
    "Sqrt": "sqrt",
    "Exp": "exp",
    "Log": "log",
    "Abs": "abs",
    "Neg": "neg",
    "Reciprocal": "reciprocal",
    "Floor": "floor",
    "Ceil": "ceil",
    "Round": "round",
    "Sign": "sign",
    "Sin": "sin",
    "Cos": "cos",
    "Tan": "tan",
    "Asin": "asin",
    "Acos": "acos",
    "Atan": "atan",
    "Sinh": "sinh",
    "Cosh": "cosh",
    "Asinh": "asinh",
    "Acosh": "acosh",
    "Atanh": "atanh",
    "Erf": "erf",

    # ---------------------------------------------------------------------
    # Reduction Operations
    # ---------------------------------------------------------------------
    "ReduceSum": "sum",
    "ReduceMean": "mean",
    "ReduceMax": "max",
    "ReduceMin": "min",
    "ReduceProd": "prod",
    "ReduceL1": "norm_l1",
    "ReduceL2": "norm_l2",
    "CumSum": "cumsum",

    # ---------------------------------------------------------------------
    # Matrix Operations
    # ---------------------------------------------------------------------
    "Trilu": "trilu",

    # ---------------------------------------------------------------------
    # Shape Operations
    # ---------------------------------------------------------------------
    "Reshape": "reshape",
    "Transpose": "permute",
    "Flatten": "flatten",
    "Squeeze": "squeeze",
    "Unsqueeze": "unsqueeze",
    "Concat": "cat",
    "Split": "split",
    "Slice": "slice",
    "Gather": "gather",
    "Scatter": "scatter",
    "ScatterElements": "scatter",
    "ScatterND": "scatter_nd",
    "GatherElements": "gather",
    "GatherND": "gather_nd",
    "Pad": "pad",
    "Tile": "tile",
    "Expand": "expand",
    "Shape": "shape",
    "Size": "size",
    "ConstantOfShape": "full",

    # ---------------------------------------------------------------------
    # Attention / Transformer
    # ---------------------------------------------------------------------
    "Attention": "scaled_dot_product_attention",
    "MultiHeadAttention": "multi_head_attention",

    # ---------------------------------------------------------------------
    # Embedding
    # ---------------------------------------------------------------------
    # NOTE: Gather is already mapped to "gather" above (line 115)
    # op_gather handles both embedding lookup (2D weight) and index selection (1D input)

    # ---------------------------------------------------------------------
    # Dropout / Regularization
    # ---------------------------------------------------------------------
    "Dropout": "dropout",

    # ---------------------------------------------------------------------
    # Comparison Operations
    # ---------------------------------------------------------------------
    "Equal": "eq",
    "Greater": "gt",
    "GreaterOrEqual": "ge",
    "Less": "lt",
    "LessOrEqual": "le",
    "Not": "logical_not",
    "And": "logical_and",
    "Or": "logical_or",
    "Xor": "logical_xor",
    "Where": "where",

    # ---------------------------------------------------------------------
    # Type Casting
    # ---------------------------------------------------------------------
    "Cast": "cast",
    "CastLike": "cast",

    # ---------------------------------------------------------------------
    # Special Operations
    # ---------------------------------------------------------------------
    "Constant": "constant",
    "Identity": "identity",
    "Clip": "clamp",
    "Range": "arange",
    "NonZero": "nonzero",
    "TopK": "topk",
    "ArgMax": "argmax",
    "ArgMin": "argmin",
    "Einsum": "einsum",

    # ---------------------------------------------------------------------
    # Image Operations
    # ---------------------------------------------------------------------
    "Resize": "interpolate",
    "Upsample": "interpolate",
    "GridSample": "grid_sample",

    # ---------------------------------------------------------------------
    # RNN Operations (for completeness)
    # ---------------------------------------------------------------------
    "LSTM": "lstm",
    "GRU": "gru",
    "RNN": "rnn",
}

# =============================================================================
# NBX DATA TYPES
# =============================================================================

NBX_DTYPES = {
    # ONNX TensorProto types to NBX dtype strings
    1: "float32",   # FLOAT
    2: "uint8",     # UINT8
    3: "int8",      # INT8
    4: "uint16",    # UINT16
    5: "int16",     # INT16
    6: "int32",     # INT32
    7: "int64",     # INT64
    8: "string",    # STRING
    9: "bool",      # BOOL
    10: "float16",  # FLOAT16
    11: "float64",  # DOUBLE
    12: "uint32",   # UINT32
    13: "uint64",   # UINT64
    14: "complex64",  # COMPLEX64
    15: "complex128", # COMPLEX128
    16: "bfloat16",   # BFLOAT16
}

# Reverse mapping: dtype string to ONNX type
NBX_DTYPE_TO_ONNX = {v: k for k, v in NBX_DTYPES.items()}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_nbx_op(onnx_op: str) -> str:
    """
    Convert ONNX operation name to NeuroBrix standard.

    Convention: lowercase, nothing else.

    Examples:
        LayerNormalization → layernormalization
        MatMul → matmul
        Less → less
        Concat → concat

    UNIVERSAL: Works for ANY future ONNX op without code changes.
    ZERO HARDCODE: No mapping table, just .lower()
    """
    return onnx_op.lower()


def get_nbx_dtype(onnx_dtype: int) -> str:
    """
    Convert ONNX data type enum to NBX dtype string.

    Args:
        onnx_dtype: ONNX TensorProto data type enum

    Returns:
        NBX dtype string (e.g., "float32", "int64")
    """
    return NBX_DTYPES.get(onnx_dtype, "float32")


def get_onnx_dtype(nbx_dtype: str) -> int:
    """
    Convert NBX dtype string to ONNX data type enum.

    Args:
        nbx_dtype: NBX dtype string (e.g., "float32")

    Returns:
        ONNX TensorProto data type enum
    """
    return NBX_DTYPE_TO_ONNX.get(nbx_dtype, 1)  # Default to FLOAT


# =============================================================================
# NBX TOPOLOGY SCHEMA KEYS
# =============================================================================

# Required keys in manifest.json
MANIFEST_REQUIRED_KEYS = [
    "format_version",
    "model_name",
    "created_at",
    "framework_source",
]

# Required keys in graph.json (TensorDAG format)
GRAPH_REQUIRED_KEYS = [
    "format",      # "tensor_dag"
    "version",     # "2.x"
    "ops",
    "tensors",
    "execution_order",
]

# Legacy alias
TOPOLOGY_REQUIRED_KEYS = GRAPH_REQUIRED_KEYS

# Valid node types
VALID_NODE_TYPES = [
    "input",
    "output",
    "module",
    "functional",
    "constant",
    "getattr",
]
