"""Where tensors placed one after another in one block land, and how many bytes the block takes: the layout every arena
of the library allocates (`memory_pool.ComponentArena`, `memory_pool.HostPinnedArena`, an expert store's slot arena),
and the very figure a planner prices an arena at — one function, so the plan and the allocation cannot differ (M3,
docs/nbxtensor/COST.md section 2). Pure arithmetic: no device, torch-free, numpy-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

from nbxtensor.dtype.itemsize import itemsize
from nbxtensor.shape import numel

#: every tensor of an arena starts on this boundary (the widest vector load of any backend; CUDA's allocation granule
#: for a sub-allocation)
ARENA_ALIGNMENT = 256


def aligned(nbytes: int, alignment: int = ARENA_ALIGNMENT) -> int:
    """`nbytes` rounded up to `alignment`."""
    return -(-int(nbytes) // alignment) * alignment


def tensor_nbytes(shape: Sequence[int], dtype) -> int:
    """The bytes a dense tensor of `shape` holds at `dtype` (a name, a torch spelling or an NBXDtype, as `itemsize`)."""
    return numel(tuple(int(d) for d in shape)) * itemsize(dtype)


@dataclass(frozen=True)
class ArenaLayout:
    """`total_bytes`: the block; `offsets`: where each tensor starts in it, in placement order."""
    total_bytes: int
    offsets: Tuple[int, ...]


def arena_layout(sizes: Iterable[int], alignment: int = ARENA_ALIGNMENT) -> ArenaLayout:
    """The layout of tensors of `sizes` bytes (placement order) in one arena: each at the next `alignment` boundary, a
    zero-byte tensor taking no room — the bump allocation `ComponentArena.alloc` performs."""
    offsets, at = [], 0
    for n in sizes:
        offsets.append(at)
        at += aligned(n, alignment) if int(n) else 0
    return ArenaLayout(at, tuple(offsets))
