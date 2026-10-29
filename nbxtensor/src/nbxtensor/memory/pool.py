"""The caching pool of the device allocator: its segments, their blocks, the free index — and the same rules replayed
as a pure function of an allocation sequence (`simulate`), which is how a planner prices what the allocator holds
beyond the tensors alive (M3, docs/nbxtensor/COST.md section 2: the 1.3 factor it replaces).

Every driver allocation the pool takes is a SEGMENT made of BLOCKS chained in address order (the method of torch's
caching allocator, S4a 2026-10-27). A request takes the smallest free block that holds it (rounded up to `align`) and
splits off the rest when it is at least `split_min`; a freed block merges with its free neighbours; only a segment wholly
free returns to the driver — over the parked cap, on a refused malloc, on a flush, and before the device grows (the
allocator flushes the pool before every new driver allocation, so the device holds what is alive and the free blocks of
segments something alive still pins). Mirrored, a block's size is reported to `on_block(ptr, size, dev)` (`size` None:
the block is gone) — the allocator keeps its per-pointer tables by it.

Torch-free, device-free: the driver is reached only through the callable a caller passes in (`flush`, `evict`).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

#: a request is rounded up to this many bytes before it is matched (the allocator's `_POOL_ALIGN`)
POOL_ALIGN = 512
#: the least remainder a block is split for; a smaller one stays with the block it was cut from
POOL_SPLIT_MIN = 1 << 20


def _no_report(ptr: int, size: Optional[int], dev: int) -> None:
    return None


class SegmentPool:
    """The free blocks of every device's segments, and the blocks in use inside them."""

    def __init__(self, align: int = POOL_ALIGN, split_min: int = POOL_SPLIT_MIN,
                 on_block: Callable[[int, Optional[int], int], None] = _no_report) -> None:
        self.align = int(align)
        self.split_min = int(split_min)
        self.on_block = on_block
        self.blocks: Dict[int, list] = {}            # ptr -> [size, prev ptr, next ptr, free, device]
        self.free: Dict[int, Dict[int, list]] = {}   # device -> {size: [ptr, ...]}  (the FREE blocks, by size)
        self.sizes: Dict[int, list] = {}             # device -> sorted sizes present in free[dev]
        self.alloc_size: Dict[int, int] = {}         # ptr -> the block's size, for the caller of the last take
        self.cached: Dict[int, int] = {}             # device -> bytes parked in free blocks
        self.cached_peak: Dict[int, int] = {}        # device -> peak of the above
        self.stats: Dict[str, int] = {"exact": 0, "fit": 0, "miss": 0, "slack_total": 0, "slack_max": 0,
                                      "flushes": 0, "flushed_bytes": 0, "evictions": 0, "evicted_bytes": 0}

    # -- the free index ------------------------------------------------------------------------------------------------

    def _index_add(self, dev: int, ptr: int, size: int) -> None:
        per = self.free.setdefault(dev, {})
        lst = per.get(size)
        if lst is None:
            per[size] = [ptr]
            bisect.insort(self.sizes.setdefault(dev, []), size)
        else:
            lst.append(ptr)

    def _index_remove(self, dev: int, ptr: int, size: int) -> None:
        per = self.free.get(dev, {})
        lst = per.get(size)
        lst.remove(ptr)
        if not lst:
            del per[size]
            sizes = self.sizes[dev]
            sizes.pop(bisect.bisect_left(sizes, size))

    # -- blocks --------------------------------------------------------------------------------------------------------

    def take(self, dev: int, nbytes: int) -> Optional[int]:
        """The smallest free block on `dev` that holds `nbytes` (rounded up to `align`), split when what is left is at
        least `split_min`; None on a miss. The block's size is left in `alloc_size[ptr]` for the caller (it may exceed
        the request by less than the split threshold)."""
        st = self.stats
        sizes = self.sizes.get(dev)
        if not sizes:
            st["miss"] += 1
            return None
        want = -(-int(nbytes) // self.align) * self.align
        k = bisect.bisect_left(sizes, want)
        if k >= len(sizes):
            st["miss"] += 1
            return None
        sz = sizes[k]
        ptr = self.free[dev][sz][-1]
        self._index_remove(dev, ptr, sz)
        blk = self.blocks[ptr]
        blk[3] = False
        if sz - want >= self.split_min:
            rest = ptr + want
            self.blocks[rest] = [sz - want, ptr, blk[2], True, dev]
            if blk[2] is not None:
                self.blocks[blk[2]][1] = rest
            blk[0], blk[2] = want, rest
            self.on_block(rest, sz - want, dev)
            self._index_add(dev, rest, sz - want)
            st["fit"] += 1
        else:
            st["exact" if sz == int(nbytes) else "fit"] += 1
            if sz > int(nbytes):
                st["slack_total"] += sz - int(nbytes)
                st["slack_max"] = max(st["slack_max"], sz - int(nbytes))
        self.alloc_size[ptr] = blk[0]
        self.cached[dev] = self.cached.get(dev, 0) - blk[0]
        return ptr

    def add_segment(self, dev: int, ptr: int, nbytes: int) -> None:
        """A driver allocation joins the pool as a one-block segment, in use."""
        self.blocks[ptr] = [int(nbytes), None, None, False, dev]

    def release(self, ptr: int) -> int:
        """`ptr`'s block is free: merged with its free neighbours and indexed; returns the bytes it adds to the pool's
        parked total (`cached` is the caller's to update, with its peak: `park`)."""
        blk = self.blocks[ptr]
        dev = blk[4]
        added = blk[0]
        blk[3] = True
        nxt = blk[2]
        if nxt is not None and self.blocks[nxt][3]:
            nb = self.blocks.pop(nxt)
            self._index_remove(dev, nxt, nb[0])
            self.on_block(nxt, None, dev)
            blk[0] += nb[0]
            blk[2] = nb[2]
            if nb[2] is not None:
                self.blocks[nb[2]][1] = ptr
        prv = blk[1]
        if prv is not None and self.blocks[prv][3]:
            pb = self.blocks[prv]
            self._index_remove(dev, prv, pb[0])
            pb[0] += blk[0]
            pb[2] = blk[2]
            if blk[2] is not None:
                self.blocks[blk[2]][1] = prv
            self.blocks.pop(ptr)
            self.on_block(ptr, None, dev)
            ptr, blk = prv, pb
        self.on_block(ptr, blk[0], dev)
        self._index_add(dev, ptr, blk[0])
        return added

    def park(self, dev: int, added: int) -> int:
        """`added` bytes more parked on `dev`; the new parked total (its peak kept)."""
        cached = self.cached.get(dev, 0) + added
        self.cached[dev] = cached
        if cached > self.cached_peak.get(dev, 0):
            self.cached_peak[dev] = cached
        return cached

    # -- segments ------------------------------------------------------------------------------------------------------

    def free_segments(self, dev: int) -> List[Tuple[int, int]]:
        """The wholly free segments on `dev`, largest first: (ptr, bytes)."""
        out = []
        for size, ptrs in self.free.get(dev, {}).items():
            for p in ptrs:
                b = self.blocks.get(p)
                if b is not None and b[1] is None and b[2] is None:
                    out.append((p, size))
        return sorted(out, key=lambda x: -x[1])

    def drop_segment(self, dev: int, ptr: int) -> int:
        """A WHOLLY free segment (its one block, free, unchained) leaves the pool; its bytes. The driver's release is
        the caller's."""
        blk = self.blocks.pop(ptr)
        self._index_remove(dev, ptr, blk[0])
        self.alloc_size.pop(ptr, None)
        return blk[0]

    def evict(self, dev: int, want: int, to_driver: Callable[[int, int], int]) -> int:
        """Wholly free segments on `dev` to the driver (`to_driver(dev, ptr)`), largest first, until at least `want`
        bytes are freed or none is left. Returns the bytes freed."""
        freed = 0
        st = self.stats
        for p, _size in self.free_segments(dev):
            if freed >= want:
                break
            freed += to_driver(dev, p)
            st["evictions"] += 1
        self.cached[dev] = max(0, self.cached.get(dev, 0) - freed)
        st["evicted_bytes"] += freed
        return freed

    def flush(self, devs: Iterable[int], to_driver: Callable[[int, int], int]) -> int:
        """Every wholly free segment of `devs` to the driver (a segment still holding a live block stays). Returns the
        bytes freed."""
        freed = 0
        self.stats["flushes"] += 1
        for d in list(devs):
            got = 0
            for p, _size in self.free_segments(d):
                got += to_driver(d, p)
            self.cached[d] = max(0, self.cached.get(d, 0) - got)
            freed += got
        self.stats["flushed_bytes"] += freed
        return freed


# -- the same rules, replayed -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Simulation:
    """What the allocator holds for a sequence of allocations and frees: `peak_bytes` from the driver (live blocks, the
    free blocks of segments something alive pins, and parked whole segments), `peak_live_bytes` of the tensors alive,
    `parked_bytes` the most parked at once."""
    peak_bytes: int
    peak_live_bytes: int
    parked_bytes: int


def simulate(events: Iterable[Tuple[str, object, int]], parked_cap: int = 0, align: int = POOL_ALIGN,
             split_min: int = POOL_SPLIT_MIN, pooled: bool = True) -> Simulation:
    """Replay `events` — ("alloc", key, nbytes) and ("free", key, 0), `key` any hashable naming one allocation — through
    the allocator's own pool (`SegmentPool`) and its own order: a request is served from the pool first; on a miss
    the pool's wholly free segments return to the driver before the device grows by the request; a freed block parks,
    and wholly free segments beyond `parked_cap` (0: unbounded) return to the driver. `pooled=False`: the pool is off
    (`NBX_ALLOC_POOL=0`), every free returns to the driver. One device; the driver is a byte counter."""
    pool = SegmentPool(align, split_min)
    held = live = peak = peak_live = parked = 0
    next_ptr = 1 << 40
    where: Dict[object, Tuple[int, int]] = {}         # key -> (ptr, requested bytes)
    seg_bytes: Dict[int, int] = {}

    def to_driver(dev: int, ptr: int) -> int:
        nonlocal held
        n = pool.drop_segment(dev, ptr)
        held -= seg_bytes.pop(ptr)
        return n

    for kind, key, nbytes in events:
        if kind == "alloc":
            n = int(nbytes)
            if n == 0:
                continue
            ptr = pool.take(0, n) if pooled else None
            if ptr is None:
                if pooled:
                    pool.flush([0], to_driver)
                ptr = next_ptr
                next_ptr += -(-n // align) * align + align
                seg_bytes[ptr] = n
                held += n
                if pooled:
                    pool.add_segment(0, ptr, n)
            else:
                pool.alloc_size.pop(ptr, None)
            where[key] = (ptr, n)
            live += n
            peak = max(peak, held)
            peak_live = max(peak_live, live)
        elif kind == "free":
            if key not in where:
                continue
            ptr, n = where.pop(key)
            live -= n
            if pooled:
                cached = pool.park(0, pool.release(ptr))
                parked = max(parked, cached)
                if parked_cap and cached > parked_cap:
                    pool.evict(0, cached - parked_cap, to_driver)
            else:
                held -= seg_bytes.pop(ptr)
        else:
            raise ValueError(f"simulate: an event is ('alloc' | 'free', key, nbytes), not {kind!r}")
    return Simulation(peak, peak_live, parked)
