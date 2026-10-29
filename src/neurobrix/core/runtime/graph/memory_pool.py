"""
Memory Pool - Buffer reuse for tensor allocation.

Phase 2+3 Optimization: Instead of allocating new tensors for each op output,
reuse buffers from tensors that are no longer needed (dead tensors).

Key concepts:
- Liveness Analysis: Compute when each tensor is last used
- Memory Pool: Store freed buffers by (shape, dtype) for reuse
- Pre-warming: First run creates buffers, subsequent runs reuse them

ZERO HARDCODE: Shapes/dtypes come from DAG tensor metadata.
ZERO FALLBACK: Missing metadata raises explicit errors.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # R33: the ATen branch imports it; shared code only annotates
    import torch
from neurobrix.core.device_utils import device_empty_cache
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set


@dataclass
class LivenessInfo:
    """Liveness analysis results for a tensor."""
    tensor_id: str
    first_use: int  # First op index that uses this tensor
    last_use: int   # Last op index that uses this tensor
    shape: Tuple[int, ...]
    dtype: torch.dtype
    is_output: bool = False  # True if this is a graph output (never freed)
    is_view: bool = False    # True if this is a view (shares storage)


class MemoryPool:
    """
    Pool of reusable tensor buffers.

    Buffers are indexed by (shape, dtype) for exact match reuse.
    When a tensor dies, its buffer is returned to the pool.
    When allocating, check pool first before creating new tensor.
    """

    def __init__(self, device: str):
        self._device = device
        # Pool: (shape, dtype) -> list of available buffers
        self._pool: Dict[Tuple[Tuple[int, ...], torch.dtype], List[torch.Tensor]] = defaultdict(list)
        # Stats
        self._hits = 0
        self._misses = 0
        self._releases = 0

    def acquire(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Acquire a buffer from the pool or allocate new.

        Args:
            shape: Required tensor shape
            dtype: Required tensor dtype

        Returns:
            A tensor buffer (may contain garbage data - caller must fill)
        """
        key = (shape, dtype)

        if self._pool[key]:
            # Pool hit - reuse existing buffer
            self._hits += 1
            return self._pool[key].pop()

        # Pool miss - allocate new
        self._misses += 1
        import torch
        return torch.empty(shape, dtype=dtype, device=self._device)

    def release(self, tensor: torch.Tensor) -> None:
        """
        Return a buffer to the pool for reuse.

        Args:
            tensor: The tensor to release (must not be a view)
        """
        # Don't pool views - they share storage with source tensors
        if tensor.storage_offset() != 0 or not tensor.is_contiguous():
            return

        key = (tuple(tensor.shape), tensor.dtype)
        self._pool[key].append(tensor)
        self._releases += 1

    def clear(self) -> None:
        """Clear all pooled buffers."""
        self._pool.clear()
        device_empty_cache(self._device)

    def stats(self) -> Dict[str, int | float]:
        """Get pool statistics."""
        total_buffers = sum(len(buffers) for buffers in self._pool.values())
        return {
            "hits": self._hits,
            "misses": self._misses,
            "releases": self._releases,
            "hit_rate": self._hits / max(1, self._hits + self._misses),
            "pooled_buffers": total_buffers,
            "unique_shapes": len(self._pool),
        }


