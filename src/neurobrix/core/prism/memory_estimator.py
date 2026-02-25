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
