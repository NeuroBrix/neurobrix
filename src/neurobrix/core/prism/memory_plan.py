"""Offsets for tensors whose lifetimes are known before they run — ONE function for the Triton engine's static memory
plan (`triton/sequence.py`, under a bucket's plan) and for Prism's price of it (docs/design/triton-static-memory-plan.md).

A tensor is an interval of op indices [first, last] (inclusive: allocated before op `first` writes it, freed after op
`last` reads it) and a size. Two tensors may share bytes only when their intervals do not overlap. Greedy by size
(Pisarchyk & Lee, "Efficient Memory Management for Deep Neural Net Inference", 2020): the tensors are placed largest
first, each at the lowest offset where it overlaps, in bytes, no ALREADY placed tensor whose interval overlaps its own.
The workspace is the highest end. Pure Python: imported by Prism and by both Triton modes (R33).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

#: every offset a multiple of this many bytes (the allocators' own granule, `ComponentArena.ALIGNMENT`)
ALIGNMENT = 256


def _align(n: int) -> int:
    return (int(n) + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def assign_offsets(tensors: Sequence[Tuple[int, int, int]]) -> Tuple[List[int], int]:
    """(offset of each tensor, workspace bytes) for `tensors` = [(first op, last op, bytes)], in their order.
    Refuses an interval that ends before it starts, or a negative size, by name."""
    for i, (a, b, n) in enumerate(tensors):
        if b < a or n < 0:
            raise ValueError(f"memory plan: tensor {i} lives [{a}, {b}] with {n} bytes — not an interval")
    order = sorted(range(len(tensors)), key=lambda i: (-_align(tensors[i][2]), tensors[i][0], i))
    offsets = [0] * len(tensors)
    placed: List[int] = []
    peak = 0
    for i in order:
        a, b, n = tensors[i]
        size = _align(n)
        if size == 0:
            offsets[i] = 0
            continue
        # the byte ranges taken, over this tensor's lifetime, by those already placed
        taken = sorted((offsets[j], offsets[j] + _align(tensors[j][2])) for j in placed
                       if tensors[j][0] <= b and a <= tensors[j][1])
        at = 0
        for lo, hi in taken:
            if at + size <= lo:
                break
            at = max(at, hi)
        offsets[i] = at
        placed.append(i)
        peak = max(peak, at + size)
    return offsets, peak


def lower_bound(tensors: Sequence[Tuple[int, int, int]]) -> int:
    """The most bytes simultaneously alive at any op — no placement can use less."""
    if not tensors:
        return 0
    ops = sorted({a for a, _, _ in tensors} | {b for _, b, _ in tensors})
    return max(sum(_align(n) for a, b, n in tensors if a <= t <= b) for t in ops)
