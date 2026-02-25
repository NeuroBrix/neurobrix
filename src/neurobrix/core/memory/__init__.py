# core/memory/__init__.py
"""
Memory Management Module for NeuroBrix Runtime

Consolidates memory cleanup logic previously duplicated across:
- core/runtime/graph_executor.py
- core/runtime/strategies/base.py
- core/runtime/strategies/single_gpu.py

SINGLE SOURCE OF TRUTH for memory operations.
"""

from .manager import MemoryManager, unload_weights, cleanup_tensors

__all__ = [
    "MemoryManager",
    "unload_weights",
    "cleanup_tensors",
]
