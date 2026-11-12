"""
NeuroBrix I/O Module.

weight_loader.py: the NBX weight loading (from the NBX container/cache).
"""

# Memory management and config
# NBX weight loader (moved from core/runtime/)
from neurobrix.core.io.weight_loader import (
    WeightLoader,
    safe_dtype_convert,
)

__all__ = [
    "WeightLoader",
    "safe_dtype_convert",
]
