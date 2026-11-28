"""
NBX Format Specification Constants

Universal operation mapping from ONNX to NBX format.
Follows the NBX Format Specification v1.0.
"""

from typing import Any, Mapping, Optional, Tuple

# =============================================================================
# NBX FORMAT CONSTANTS
# =============================================================================

NBX_VERSION = "1.0.0"
NBX_MAGIC = b"NBX\x00"  # Magic bytes for .nbx files

# Container format version contract (manifest.json "nbx_container_version").
# Engine 0.7.0 strictly requires containers with nbx_container_version == "0.2"
# and refuses legacy (e.g. nbx_version: "0.1" or missing nbx_container_version)
# as well as incompatible older/newer container format versions.
MIN_SUPPORTED_NBX_CONTAINER_VERSION = "0.2"
MAX_SUPPORTED_NBX_CONTAINER_VERSION = "0.2"
REQUIRED_NBX_CONTAINER_VERSION = MIN_SUPPORTED_NBX_CONTAINER_VERSION


def parse_nbx_container_version(version: Any) -> Tuple[int, ...]:
    """Parse a dotted container format version string (e.g. "0.2") into an integer tuple.

    Trailing zero patch segments are normalized so "0.2.0" compares equal to "0.2",
    while "0.10" compares newer than "0.2". Non-string, empty, single-segment, or
    non-numeric values raise ValueError.
    """
    if not isinstance(version, str):
        raise ValueError(f"expected a version string, got {type(version).__name__}")
    raw = version.strip()
    parts = raw.split(".")
    if len(parts) < 2 or not all(p.isdigit() for p in parts):
        raise ValueError(
            f"expected dotted numeric version (e.g. {REQUIRED_NBX_CONTAINER_VERSION!r}), got {version!r}"
        )
    nums = [int(p) for p in parts]
    while len(nums) > 2 and nums[-1] == 0:
        nums.pop()
    return tuple(nums)


def refuse_unsupported_nbx_container_version(
    manifest: Optional[Mapping[str, Any]],
    container: str = "",
) -> str:
    """Refuse any container whose manifest nbx_container_version is missing, outdated, or too new.

    Raises RuntimeError with an actionable message naming the found version (including
    legacy nbx_version when nbx_container_version is absent), the supported version,
    and the update command. Returns the validated version string when supported.
    """
    has_mapping = isinstance(manifest, Mapping)
    version = manifest.get("nbx_container_version") if has_mapping else None
    legacy_version = manifest.get("nbx_version") if has_mapping else None
    raw_name = manifest.get("model_name") if has_mapping else None
    model_ref = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else "<model>"
    target = container or model_ref
    expected = (
        MIN_SUPPORTED_NBX_CONTAINER_VERSION
        if MIN_SUPPORTED_NBX_CONTAINER_VERSION == MAX_SUPPORTED_NBX_CONTAINER_VERSION
        else f"{MIN_SUPPORTED_NBX_CONTAINER_VERSION}..{MAX_SUPPORTED_NBX_CONTAINER_VERSION}"
    )

    if version is None:
        if legacy_version is not None:
            raise RuntimeError(
                f"NBX CONTAINER VERSION: container {target!r} uses outdated container format "
                f"(legacy nbx_version={legacy_version!r}, missing nbx_container_version); "
                f"NeuroBrix 0.7.0 requires nbx_container_version={expected!r}.\n"
                f"  FIX: Update / re-download the model container:\n"
                f"    neurobrix remove {model_ref} && neurobrix import <org>/{model_ref}"
            )
        raise RuntimeError(
            f"NBX CONTAINER VERSION: container {target!r} is missing required "
            f"nbx_container_version (outdated or incomplete format); "
            f"NeuroBrix 0.7.0 requires nbx_container_version={expected!r}.\n"
            f"  FIX: Update / re-download the model container:\n"
            f"    neurobrix remove {model_ref} && neurobrix import <org>/{model_ref}"
        )

    try:
        got = parse_nbx_container_version(version)
    except ValueError as exc:
        raise RuntimeError(
            f"NBX CONTAINER VERSION: container {target!r} has unreadable "
            f"nbx_container_version ({version!r}: {exc}); "
            f"NeuroBrix 0.7.0 requires nbx_container_version={expected!r}.\n"
            f"  FIX: Update / re-download the model container:\n"
            f"    neurobrix remove {model_ref} && neurobrix import <org>/{model_ref}"
        ) from None

    min_ver = parse_nbx_container_version(MIN_SUPPORTED_NBX_CONTAINER_VERSION)
    max_ver = parse_nbx_container_version(MAX_SUPPORTED_NBX_CONTAINER_VERSION)

    if got < min_ver:
        raise RuntimeError(
            f"NBX CONTAINER VERSION: container {target!r} uses outdated format "
            f"nbx_container_version {version!r}; NeuroBrix 0.7.0 requires "
            f"nbx_container_version >= {MIN_SUPPORTED_NBX_CONTAINER_VERSION} (supported: {expected}).\n"
            f"  FIX: Update / re-download the model container:\n"
            f"    neurobrix remove {model_ref} && neurobrix import <org>/{model_ref}"
        )

    if got > max_ver:
        raise RuntimeError(
            f"NBX CONTAINER VERSION: container {target!r} uses newer format "
            f"nbx_container_version {version!r}, which is not supported by NeuroBrix 0.7.0 "
            f"(supported: {expected}).\n"
            f"  FIX: Upgrade neurobrix (pip install -U neurobrix) to load containers "
            f"of format nbx_container_version {version!r}."
        )

    return version

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
