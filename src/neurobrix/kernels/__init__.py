"""
NeuroBrix Kernel System - Enterprise Grade.

Architecture:
- adapter.py: Universal ATen → Kernel translation (ENTRY POINT)
- registry.py: @register_kernel decorator
- resolver.py: Family/Vendor/Tier cascade resolution
- ops/: Pure Triton kernels (ZERO wrappers)

Usage:
    from kernels import KernelAdapter, get_kernel

    # Via adapter (recommended for graph execution)
    adapter = KernelAdapter(family="image", vendor="nvidia", arch="volta", device="cuda:0")
    result = adapter.launch("aten::add", [a, b], {"alpha": 1.0})

    # Direct kernel access
    kernel = get_kernel("add", family="image", vendor="nvidia", arch="volta")
    result = kernel(a, b, alpha=1.0)
"""

from .adapter import KernelAdapter
from .registry import register_kernel, KERNEL_REGISTRY, KernelMeta, list_kernels
from .resolver import get_kernel, run_op
from .classification import OpExecution, get_execution_type, ATEN_CLASSIFICATION
from .mapping import get_kernel_op_name, ATEN_TO_KERNEL
from .metadata_ops import execute_metadata_op

__all__ = [
    # Adapter (primary entry point)
    "KernelAdapter",
    # Registry
    "register_kernel",
    "KERNEL_REGISTRY",
    "KernelMeta",
    "list_kernels",
    # Resolver
    "get_kernel",
    "run_op",
    # Classification
    "OpExecution",
    "get_execution_type",
    "ATEN_CLASSIFICATION",
    # Mapping
    "get_kernel_op_name",
    "ATEN_TO_KERNEL",
    # Metadata ops
    "execute_metadata_op",
]
