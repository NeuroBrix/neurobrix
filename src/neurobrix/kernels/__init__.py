"""
NeuroBrix Kernel System.

Architecture:
- dispatch.py: aten_op → Triton kernel wrapper (--triton mode)
- wrappers.py: PyTorch wrappers for Triton kernels
- ops/: Pure @triton.jit kernels (ZERO import torch)
- classification.py: Op classification (TRITON vs METADATA)
- metadata_ops.py: Shape/view ops (PyTorch native)

Usage:
    from neurobrix.kernels.dispatch import dispatch
    kernel = dispatch("aten::relu")  # Returns Triton wrapper or None
"""

from .classification import OpExecution, get_execution_type, ATEN_CLASSIFICATION
from .metadata_ops import execute_metadata_op


__all__ = [
    # Classification
    "OpExecution",
    "get_execution_type",
    "ATEN_CLASSIFICATION",
    # Metadata ops
    "execute_metadata_op",
]
