"""The caching pool of the device allocator: its segments, their blocks, the free index — and the same rules replayed
as a pure function of an allocation sequence (`simulate`), which is how a planner prices what the allocator holds
beyond the tensors alive (M3, docs/nbxtensor/COST.md section 2: the 1.3 factor it replaces).

Every driver allocation the pool takes is a SEGMENT made of BLOCKS chained in address order (the method of torch's
caching allocator, S4a 2026-10-27), in one of two SIZE CLASSES: a request of at most `POOL_SMALL_BYTES` is served only
from small segments of `POOL_SMALL_SEGMENT_BYTES`, a larger one only from large segments grown to the request (torch's
kSmallSize / kSmallBuffer). A request takes the smallest free block of its class that holds it (rounded up to `align`)
and splits off the rest when it is at least `split_min` — in the large class only when what is left is at most the
request itself (the half rule), so a block is never pinned by something smaller than its remainder. A freed block
merges with its free neighbours; only a segment wholly free returns to the driver — over the parked cap, on a refused
malloc, on a flush. A wholly free segment stays parked across a growth: it is the next request of its size's block.

Why the classes and the half rule (Allegro triton on a V100-16GB, gate of 2026-11-01): with one class and a free split,
a 17 KB scalar took a freed 2.5 GB block and pinned its remainder alive, a 632 MB request did the same to the next —
4.4 GB of free blocks in pinned segments beside 9.7 GB alive refused a 2.5 GB gelu. The real trace replayed through
these rules (campaign 2026_11_01_gate_nbxtensor, trace_allegro/replay.py; also mochi-1-preview 32g and Kokoro) completes
with the least held of every pooled rule tried, and 164e89dc's flush of the pool before every growth — which only
emptied the small free segments the scalars would have used — is withdrawn with it.

Mirrored, a block's size is reported to `on_block(ptr, size, dev)` (`size` None: the block is gone) — the allocator
keeps its per-pointer tables by it.

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
#: the largest request of the small class (torch's caching allocator: kSmallSize)
POOL_SMALL_BYTES = 1 << 20
#: the segment a small request grows the device by, carved for the small requests that follow (kSmallBuffer)
POOL_SMALL_SEGMENT_BYTES = 1 << 21

LARGE, SMALL = 0, 1


def _no_report(ptr: int, size: Optional[int], dev: int) -> None:
    return None


class SegmentPool:
    """The free blocks of every device's segments, and the blocks in use inside them."""

    def __init__(self, align: int = POOL_ALIGN, split_min: int = POOL_SPLIT_MIN, small: int = POOL_SMALL_BYTES,
                 small_segment: int = POOL_SMALL_SEGMENT_BYTES,
                 on_block: Callable[[int, Optional[int], int], None] = _no_report) -> None:
        self.align = int(align)
        self.split_min = int(split_min)
        self.small = int(small)
        self.small_segment = int(small_segment)
        self.on_block = on_block
        self.blocks: Dict[int, list] = {}            # ptr -> [size, prev ptr, next ptr, free, device, class]
        self.free: Dict[int, Dict[int, list]] = {}   # device -> {size: [ptr, ...]}  (the FREE blocks, by size)
        self.sizes: Dict[int, list] = {}             # device -> sorted sizes present in free[dev]
        self._cls_free: Dict[Tuple[int, int], Dict[int, list]] = {}   # (device, class) -> {size: [ptr, ...]}
        self._cls_sizes: Dict[Tuple[int, int], list] = {}             # (device, class) -> sorted sizes
        self.alloc_size: Dict[int, int] = {}         # ptr -> the block's size, for the caller of the last take
        self.cached: Dict[int, int] = {}             # device -> bytes parked in free blocks
        self.cached_peak: Dict[int, int] = {}        # device -> peak of the above
        self.stats: Dict[str, int] = {"exact": 0, "fit": 0, "miss": 0, "slack_total": 0, "slack_max": 0,
                                      "flushes": 0, "flushed_bytes": 0, "evictions": 0, "evicted_bytes": 0}

    # -- the classes ---------------------------------------------------------------------------------------------------

    def cls(self, nbytes: int) -> int:
        """The size class of a request: SMALL up to `small` bytes, LARGE beyond."""
        return SMALL if int(nbytes) <= self.small else LARGE

    def grow_size(self, nbytes: int) -> int:
        """The bytes the device grows by for a request the pool cannot serve: a small segment for a small request, the
        request itself (rounded up to `align`) for a large one."""
        if self.cls(nbytes) == SMALL:
            return self.small_segment
        return -(-int(nbytes) // self.align) * self.align

    # -- the free index ------------------------------------------------------------------------------------------------

    @staticmethod
    def _add(index: dict, sizes: dict, key, ptr: int, size: int) -> None:
        per = index.setdefault(key, {})
        lst = per.get(size)
        if lst is None:
            per[size] = [ptr]
            bisect.insort(sizes.setdefault(key, []), size)
        else:
            lst.append(ptr)

    @staticmethod
    def _remove(index: dict, sizes: dict, key, ptr: int, size: int) -> None:
        per = index.get(key, {})
        lst = per.get(size)
        lst.remove(ptr)
        if not lst:
            del per[size]
            sz = sizes[key]
            sz.pop(bisect.bisect_left(sz, size))

    def _index_add(self, dev: int, ptr: int, size: int) -> None:
        self._add(self.free, self.sizes, dev, ptr, size)
        self._add(self._cls_free, self._cls_sizes, (dev, self.blocks[ptr][5]), ptr, size)

    def _index_remove(self, dev: int, ptr: int, size: int) -> None:
        self._remove(self.free, self.sizes, dev, ptr, size)
        self._remove(self._cls_free, self._cls_sizes, (dev, self.blocks[ptr][5]), ptr, size)

    # -- blocks --------------------------------------------------------------------------------------------------------

    def take(self, dev: int, nbytes: int) -> Optional[int]:
        """The smallest free block of the request's class on `dev` that holds `nbytes` (rounded up to `align`), split
        when what is left is at least `split_min` — and, in the large class, at most the request; None on a miss (no
        such block, or the only ones would be left with more free than the request pins). The block's size is left in
        `alloc_size[ptr]` for the caller (it may exceed the request by less than the split threshold)."""
        st = self.stats
        cls = self.cls(nbytes)
        sizes = self._cls_sizes.get((dev, cls))
        if not sizes:
            st["miss"] += 1
            return None
        want = -(-int(nbytes) // self.align) * self.align
        k = bisect.bisect_left(sizes, want)
        if k >= len(sizes):
            st["miss"] += 1
            return None
        sz = sizes[k]
        if cls == LARGE and sz - want >= self.split_min and sz - want > want:
            st["miss"] += 1                      # the half rule: a larger block would leave even more pinned
            return None
        ptr = self._cls_free[(dev, cls)][sz][-1]
        self._index_remove(dev, ptr, sz)
        blk = self.blocks[ptr]
        blk[3] = False
        if sz - want >= self.split_min:
            rest = ptr + want
            self.blocks[rest] = [sz - want, ptr, blk[2], True, dev, cls]
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

    def add_free_segment(self, dev: int, ptr: int, nbytes: int, request: int) -> None:
        """A driver allocation of `nbytes` grown for `request` (`grow_size`) joins the pool as a one-block FREE segment
        of the request's class, parked; the caller takes the request from it next."""
        cls = self.cls(request)
        self.blocks[ptr] = [int(nbytes), None, None, True, dev, cls]
        self._index_add(dev, ptr, int(nbytes))
        self.cached[dev] = self.cached.get(dev, 0) + int(nbytes)

    def release(self, ptr: int) -> int:
        """`ptr`'s block is free: merged with its free neighbours and indexed; returns the bytes it adds to the pool's
        parked total (`cached` is the caller's to update, with its peak: `park`)."""
        blk = self.blocks[ptr]
        dev = blk[4]
        added = blk[0]
        blk[3] = True
        nxt = blk[2]
        if nxt is not None and self.blocks[nxt][3]:
            self._index_remove(dev, nxt, self.blocks[nxt][0])
            nb = self.blocks.pop(nxt)
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
        blk = self.blocks[ptr]
        self._index_remove(dev, ptr, blk[0])
        self.blocks.pop(ptr)
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
    """Replay `events` — ("alloc", key, nbytes), ("arena", key, nbytes) and ("free", key, 0), `key` any hashable naming
    one allocation — through the allocator's own pool (`SegmentPool`) and its own order: a request is served from the
    pool first; on a miss the device grows by `grow_size` and the new segment is taken from; a freed block parks, and
    wholly free segments beyond `parked_cap` (0: unbounded) return to the driver. An "arena" is a block outside the
    pool (`malloc_cuda(pooled=False)`, a weight arena): straight from the driver, back to it whole when freed.
    `pooled=False`: the pool is off (`NBX_ALLOC_POOL=0`), every free returns to the driver. One device; the driver is a
    byte counter that never refuses (what a refusal flushes is the parked whole segments, counted in `parked_bytes`)."""
    pool = SegmentPool(align, split_min)
    held = live = peak = peak_live = parked = 0
    next_ptr = 1 << 40
    where: Dict[object, Tuple[int, int]] = {}         # key -> (ptr, requested bytes)
    seg_bytes: Dict[int, int] = {}
    arenas: set = set()                                # the ptrs of blocks outside the pool

    def to_driver(dev: int, ptr: int) -> int:
        nonlocal held
        n = pool.drop_segment(dev, ptr)
        held -= seg_bytes.pop(ptr)
        return n

    for kind, key, nbytes in events:
        if kind == "arena":
            n = int(nbytes)
            if n == 0:
                continue
            ptr = next_ptr
            next_ptr += -(-n // align) * align + align
            seg_bytes[ptr] = n
            arenas.add(ptr)
            held += n
            where[key] = (ptr, n)
            live += n
            peak = max(peak, held)
            peak_live = max(peak_live, live)
            continue
        if kind == "alloc":
            n = int(nbytes)
            if n == 0:
                continue
            ptr = pool.take(0, n) if pooled else None
            if ptr is None:
                grown = pool.grow_size(n) if pooled else n
                seg = next_ptr
                next_ptr += grown + align
                seg_bytes[seg] = grown
                held += grown
                if pooled:
                    pool.add_free_segment(0, seg, grown, n)
                    ptr = pool.take(0, n)
                else:
                    ptr = seg
            pool.alloc_size.pop(ptr, None) if pooled else None
            where[key] = (ptr, n)
            live += n
            peak = max(peak, held)
            peak_live = max(peak_live, live)
        elif kind == "free":
            if key not in where:
                continue
            ptr, n = where.pop(key)
            live -= n
            if ptr in arenas:
                arenas.discard(ptr)
                held -= seg_bytes.pop(ptr)
            elif pooled:
                cached = pool.park(0, pool.release(ptr))
                parked = max(parked, cached)
                if parked_cap and cached > parked_cap:
                    pool.evict(0, cached - parked_cap, to_driver)
            else:
                held -= seg_bytes.pop(ptr)
        else:
            raise ValueError(f"simulate: an event is ('alloc' | 'free', key, nbytes), not {kind!r}")
    return Simulation(peak, peak_live, parked)
