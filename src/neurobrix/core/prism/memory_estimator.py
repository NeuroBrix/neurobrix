"""
NeuroBrix Prism - Memory Estimator
Tensor memory calculation utilities.

Uses core.dtype for dtype constants (single source of truth).
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import from single source of truth
from neurobrix.core.dtype import BYTES_MAP as DTYPE_BYTES


from neurobrix.core.config.system import bytes_to_mb, bytes_to_gb


def _get_dtype_bytes() -> Dict[str, int]:
    """Get dtype bytes mapping."""
    return DTYPE_BYTES


def estimate_op_workspace_bytes(
    mode: str,
    op_type: str,
    input_shapes: List[List[int]],
    output_shapes: List[List[int]],
    dtype_bytes: int,
    vram_per_gpu_bytes: Optional[int] = None,
) -> int:
    """Bound for the workspace memory an op needs ON TOP of its output tensor.

    Modeled per op class. The cuDNN workspace for conv with implicit-gemm
    on large feature maps grows ~2 × im2col matrix bytes (input × kh × kw
    × out_h × out_w), but cuDNN itself caps it at the largest algorithm
    that fits in VRAM. We bound by the smaller of the theoretical 2× im2col
    and 0.75 × VRAM (R3-style: hardware-driven, no global hardcoded
    ceiling).

    Triton mode = ~0 workspace (kernels write direct to output, no im2col
    matrix materialized) → returns 0.

    Used by ActivationProfiler.overflow_ops to flag ops Prism must tile.
    """
    if mode in ("triton", "triton_sequential"):
        return 0  # Triton kernels stream, no extra workspace

    cl = op_type.split("::")[-1]

    if cl in ("convolution", "conv2d", "conv1d", "_convolution"):
        # cuDNN implicit-gemm on large feature maps:
        # im2col matrix = (N × out_h × out_w) × (in_c × kh × kw) bytes
        # Worst-case workspace ~ 2 × im2col, capped by 0.75 × VRAM.
        if not input_shapes or not output_shapes:
            return 0
        in_shape = input_shapes[0]    # Conv1D/2D/3D: [N, in_c, *spatial]
        weight_shape = input_shapes[1] if len(input_shapes) > 1 else None
        out_shape = output_shapes[0]  # [N, out_c, *spatial_out]
        # Rank-discriminated (3D=Conv1D, 4D=Conv2D, 5D=Conv3D video). The im2col
        # matrix is (N × prod(out_spatial)) × (in_c × prod(kernel_spatial)).
        # Branch on tensor rank, never on model family (R34).
        rank = len(in_shape)
        if (out_shape is None or weight_shape is None
                or len(out_shape) != rank or len(weight_shape) != rank
                or rank not in (3, 4, 5)):
            return 0
        n, in_c = in_shape[0], in_shape[1]
        out_spatial = 1
        kernel_spatial = 1
        for ax in range(2, rank):
            out_spatial *= out_shape[ax]
            kernel_spatial *= weight_shape[ax]
        im2col_bytes = n * out_spatial * in_c * kernel_spatial * dtype_bytes
        workspace_upper = 2 * im2col_bytes
        if vram_per_gpu_bytes is not None and vram_per_gpu_bytes > 0:
            workspace_upper = min(workspace_upper, int(0.75 * vram_per_gpu_bytes))
        return int(workspace_upper)

    if "scaled_dot_product_attention" in cl:
        # Softmax temp = scores [B, H, T_q, T_k] in fp32 (always promoted).
        if not input_shapes:
            return 0
        q_shape = input_shapes[0]
        k_shape = input_shapes[1] if len(input_shapes) > 1 else q_shape
        if len(q_shape) < 4 or len(k_shape) < 4:
            return 0
        b, h, t_q = q_shape[0], q_shape[1], q_shape[2]
        t_k = k_shape[2]
        # fp32 scores
        return int(b * h * t_q * t_k * 4)

    # upsample / element-wise / norms: kernels stream output, ~no workspace
    return 0


def compute_tensor_bytes(shape: List[int], dtype: str) -> int:
    """
    Compute memory for a tensor in bytes.

    ZERO HARDCODE: dtype_bytes from config/system.yml

    Args:
        shape: Tensor shape as list of integers
        dtype: Dtype string (e.g., "float16", "float32", "bfloat16")

    Returns:
        Size in bytes
    """
    dtype_bytes = _get_dtype_bytes()
    bytes_per_element = dtype_bytes.get(dtype, 4)  # Default to float32

    numel = 1
    for dim in shape:
        numel *= dim

    return numel * bytes_per_element


def compute_tensor_mb(shape: List[int], dtype: str) -> float:
    """Compute memory for a tensor in MB."""
    return bytes_to_mb(compute_tensor_bytes(shape, dtype))


def compute_tensor_gb(shape: List[int], dtype: str) -> float:
    """Compute memory for a tensor in GB."""
    return bytes_to_gb(compute_tensor_bytes(shape, dtype))


def compute_dtype_factor(source_dtype: str, target_dtype: str) -> float:
    """
    Compute memory multiplier for dtype conversion.

    Args:
        source_dtype: Original dtype (e.g., "float16")
        target_dtype: Target dtype (e.g., "float32")

    Returns:
        Multiplier (e.g., 2.0 for float16→float32)
    """
    dtype_bytes = _get_dtype_bytes()
    source_bytes = dtype_bytes.get(source_dtype, 2)
    target_bytes = dtype_bytes.get(target_dtype, 4)

    return target_bytes / source_bytes


def get_dtype_bytes_per_element(dtype: str) -> int:
    """Get bytes per element for a dtype."""
    dtype_bytes = _get_dtype_bytes()
    return dtype_bytes.get(dtype, 4)


@dataclass
class MemoryBreakdown:
    """Memory breakdown for a component."""
    component_name: str
    weight_bytes: int
    activation_bytes: int
    overhead_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.activation_bytes + self.overhead_bytes

    @property
    def total_mb(self) -> float:
        return bytes_to_mb(self.total_bytes)

    @property
    def total_gb(self) -> float:
        return bytes_to_gb(self.total_bytes)

    @property
    def weight_mb(self) -> float:
        return bytes_to_mb(self.weight_bytes)

    @property
    def activation_mb(self) -> float:
        return bytes_to_mb(self.activation_bytes)

    @property
    def overhead_mb(self) -> float:
        return bytes_to_mb(self.overhead_bytes)

    def __repr__(self) -> str:
        return (
            f"MemoryBreakdown({self.component_name}: "
            f"weights={self.weight_mb:.0f}MB, "
            f"activations={self.activation_mb:.0f}MB, "
            f"overhead={self.overhead_mb:.0f}MB, "
            f"total={self.total_mb:.0f}MB)"
        )
