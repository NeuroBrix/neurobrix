"""Shadow device memory: addresses that address nothing, distinct and aligned, handed out where an application runs the
library with no device in sight (the engine's launch census derives every kernel key a model forms without running it)
— moved from the engine's census, M2. The application substitutes `malloc` for the allocator's direct allocation
(`DeviceAllocator.malloc_direct`); nothing here is installed by itself, and a real run never records anything here.

Every allocation the shadow hands out is recorded AS a shadow: base -> nbytes, bases sorted. A consumer that must size
an allocation (the Metal pin sizes its whole-allocation wrap) asks `shadow_range` and gets an answer for a recorded
shadow and None for anything else, so an address nobody recorded keeps failing in every run, shadow or not. Measured
2026-09-24: the 76 MoE census failures were the pin asking the allocator about addresses only the shadow had handed
out. Torch-free.
"""
from __future__ import annotations

import bisect
from typing import Dict, List, Optional

_NEXT = [1 << 40]                       # addresses that address nothing, distinct and aligned
_RANGES: Dict[int, int] = {}
_BASES: List[int] = []


def malloc(nbytes: int, dev_idx: Optional[int] = None, pooled: bool = True) -> int:
    """`DeviceAllocator.malloc_direct` in a shadow, its signature kept whole (`pooled`: a weight arena asks for a block
    outside the pool; a shadow has no pool and answers the same address either way)."""
    ptr = _NEXT[0]
    _NEXT[0] += (int(nbytes) + 255) // 256 * 256 + 256
    _RANGES[ptr] = int(nbytes)
    _BASES.append(ptr)          # monotonic: appending keeps the list sorted
    return ptr


def shadow_range(addr: int) -> Optional[tuple]:
    """(base, nbytes) of the RECORDED shadow allocation containing `addr`, or None.

    None for every address this shadow did not hand out, including every address of a real
    run, where nothing is ever recorded here."""
    addr = int(addr)
    i = bisect.bisect_right(_BASES, addr) - 1
    if i < 0:
        return None
    base = _BASES[i]
    size = _RANGES[base]
    return (base, size) if addr < base + max(size, 1) else None
