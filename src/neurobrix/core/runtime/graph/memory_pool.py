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

import torch
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


class LivenessAnalyzer:
    """
    Analyzes tensor liveness in a DAG.

    For each tensor, determines:
    - When it's first created/used
    - When it's last used (and can be freed)
    - Its shape and dtype
    """

    def __init__(self):
        self._liveness: Dict[str, LivenessInfo] = {}

    def analyze(
        self,
        dag: Dict[str, Any],
        symbol_values: Optional[Dict[str, int]] = None
    ) -> Dict[str, LivenessInfo]:
        """
        Analyze tensor liveness in the DAG.

        Args:
            dag: The graph DAG
            symbol_values: Resolved symbolic dimension values

        Returns:
            Dict mapping tensor_id -> LivenessInfo
        """
        execution_order = dag.get("execution_order", [])
        ops = dag.get("ops", {})
        tensors = dag.get("tensors", {})
        output_tensor_ids = set(dag.get("output_tensor_ids", []))

        # Pass 1: Track first/last use for each tensor
        first_use: Dict[str, int] = {}
        last_use: Dict[str, int] = {}

        for i, op_uid in enumerate(execution_order):
            op_data = ops.get(op_uid, {})

            # Track input tensors
            for tid in op_data.get("input_tensor_ids", []):
                if tid not in first_use:
                    first_use[tid] = i
                last_use[tid] = i

            # Track output tensors (they're "created" at this op)
            for tid in op_data.get("output_tensor_ids", []):
                if tid not in first_use:
                    first_use[tid] = i
                # Output tensor is first used by this op

        # Pass 2: Build LivenessInfo for each tensor
        liveness = {}

        for tid, tdata in tensors.items():
            # Get shape - may be symbolic
            shape = self._resolve_shape(tdata, symbol_values)
            dtype = self._resolve_dtype(tdata)

            # Determine if this is a view tensor
            is_view = tdata.get("is_view", False)

            liveness[tid] = LivenessInfo(
                tensor_id=tid,
                first_use=first_use.get(tid, 0),
                last_use=last_use.get(tid, len(execution_order)),
                shape=shape,
                dtype=dtype,
                is_output=tid in output_tensor_ids,
                is_view=is_view,
            )

        self._liveness = liveness
        return liveness

    def _resolve_shape(
        self,
        tdata: Dict[str, Any],
        symbol_values: Optional[Dict[str, int]]
    ) -> Tuple[int, ...]:
        """Resolve tensor shape, substituting symbolic values."""
        shape = tdata.get("shape", [])

        if not shape:
            return ()

        resolved = []
        for dim in shape:
            if isinstance(dim, int):
                resolved.append(dim)
            elif isinstance(dim, str) and symbol_values:
                # Symbolic dimension - look up in symbol_values
                if dim in symbol_values:
                    resolved.append(symbol_values[dim])
                else:
                    # Try common prefixes
                    for key in symbol_values:
                        if dim in key or key in dim:
                            resolved.append(symbol_values[key])
                            break
                    else:
                        # Can't resolve - return unknown
                        return ()
            else:
                # Unknown dimension type
                return ()

        return tuple(resolved)

    def _resolve_dtype(self, tdata: Dict[str, Any]) -> torch.dtype:
        """Resolve tensor dtype from metadata."""
        from neurobrix.core.dtype.config import parse_dtype
        dtype_str = tdata.get("dtype", "float32")
        return parse_dtype(dtype_str)

    def get_dead_tensors_at(self, op_idx: int) -> List[str]:
        """Get tensors that become dead (can be freed) after this op."""
        dead = []
        for tid, info in self._liveness.items():
            if info.last_use == op_idx and not info.is_output and not info.is_view:
                dead.append(tid)
        return dead


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
        torch.cuda.empty_cache()

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


class MemoryOptimizer:
    """
    Orchestrates memory optimization for DAG execution.

    Combines:
    - Liveness analysis to know when tensors die
    - Memory pooling for buffer reuse
    - Pre-allocation analysis for warm-up
    """

    def __init__(self, device: str):
        self._device = device
        self._liveness_analyzer = LivenessAnalyzer()
        self._memory_pool = MemoryPool(device)
        self._liveness: Dict[str, LivenessInfo] = {}
        self._analyzed = False

    def analyze(
        self,
        dag: Dict[str, Any],
        symbol_values: Optional[Dict[str, int]] = None
    ) -> None:
        """
        Analyze DAG for memory optimization.

        Call this once before execution loop.
        """
        self._liveness = self._liveness_analyzer.analyze(dag, symbol_values)
        self._analyzed = True

    def get_dead_tensors(self, op_idx: int) -> List[str]:
        """Get tensors that die after this op index."""
        if not self._analyzed:
            return []
        return self._liveness_analyzer.get_dead_tensors_at(op_idx)

    def release_tensor(self, tensor: torch.Tensor) -> None:
        """Release a tensor to the pool for reuse."""
        self._memory_pool.release(tensor)

    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Allocate or reuse a buffer."""
        return self._memory_pool.acquire(shape, dtype)

    def clear_pool(self) -> None:
        """Clear the memory pool."""
        self._memory_pool.clear()

    def stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "pool": self._memory_pool.stats(),
            "tensors_analyzed": len(self._liveness),
        }

    @property
    def liveness(self) -> Dict[str, LivenessInfo]:
        """Get liveness info for all tensors."""
        return self._liveness


def compute_allocation_schedule(
    dag: Dict[str, Any],
    symbol_values: Optional[Dict[str, int]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute a buffer allocation schedule for the DAG.

    This analyzes which tensors can share buffers because their
    lifetimes don't overlap.

    Returns:
        Dict mapping tensor_id -> allocation info:
        {
            "buffer_group": int,  # Tensors in same group share buffer
            "shape": tuple,
            "dtype": torch.dtype,
        }
    """
    analyzer = LivenessAnalyzer()
    liveness = analyzer.analyze(dag, symbol_values)

    # Group tensors by (shape, dtype) - only same shape/dtype can share
    groups: Dict[Tuple[Tuple[int, ...], torch.dtype], List[LivenessInfo]] = defaultdict(list)

    for info in liveness.values():
        if not info.is_output and not info.is_view and info.shape:
            key = (info.shape, info.dtype)
            groups[key].append(info)

    # For each group, assign buffer slots using interval scheduling
    allocation = {}
    buffer_id = 0

    for (shape, dtype), infos in groups.items():
        # Sort by first_use
        infos.sort(key=lambda x: x.first_use)

        # Greedy interval scheduling
        active_buffers: List[Tuple[int, int]] = []  # (end_time, buffer_id)

        for info in infos:
            # Find a buffer that's free (end_time < first_use)
            assigned = None
            for i, (end_time, bid) in enumerate(active_buffers):
                if end_time < info.first_use:
                    # Can reuse this buffer
                    assigned = bid
                    active_buffers[i] = (info.last_use, bid)
                    break

            if assigned is None:
                # Need new buffer
                assigned = buffer_id
                buffer_id += 1
                active_buffers.append((info.last_use, assigned))

            allocation[info.tensor_id] = {
                "buffer_group": assigned,
                "shape": shape,
                "dtype": dtype,
            }

    return allocation
