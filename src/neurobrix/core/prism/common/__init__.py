# core/prism/common/__init__.py
"""
Prism Common Utilities

Components:
- DtypeResolver: Dtype resolution based on model and hardware
- BlockIndexExtractor: Extract block indices from shard filenames
- AllocationHelpers: Common allocation utilities
"""

from .dtype_resolver import DtypeResolver, resolve_dtype
from .block_index import BlockIndexExtractor, extract_block_index
from .allocation import (
    allocate_sequential_shards,
    calculate_dtype_multiplier,
    get_device_index,
)

__all__ = [
    # Dtype
    "DtypeResolver",
    "resolve_dtype",
    # Block Index
    "BlockIndexExtractor",
    "extract_block_index",
    # Allocation
    "allocate_sequential_shards",
    "calculate_dtype_multiplier",
    "get_device_index",
]
