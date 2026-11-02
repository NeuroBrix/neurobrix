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
        #: device -> its expandable ranges (`Range`), the large class's address space where the device maps chunks
        #: (`memory.vmm`); none: the large class grows by driver segments as the small one does
        self.ranges: Dict[int, List["Range"]] = {}

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
        ptr = self._cls_free[(dev, cls)][sz][-1]
        if (cls == LARGE and sz - want >= self.split_min and sz - want > want
                and not (self.ranges and _range_of(self, dev, ptr) is not None)):
            # the half rule: a larger block would leave even more pinned — in a driver segment; in an expandable range
            # a remainder pins no page, its whole chunks are unmapped at a reclaim (R16)
            st["miss"] += 1
            return None
        self._index_remove(dev, ptr, sz)
        blk = self.blocks[ptr]
        blk[3] = False
        if sz - want >= self.split_min:
            rest = ptr + want
            self.blocks[rest] = [sz - want, ptr, blk[2], True, dev, cls]
            if blk[2] is not None:
                self.blocks[blk[2]][1] = rest
            blk[0], blk[2] = want, rest
            if self.ranges:
                r = _range_of(self, dev, ptr)
                if r is not None and r.last == ptr:
                    r.last = rest
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
        r = _range_of(self, dev, ptr) if self.ranges else None
        if nxt is not None and self.blocks[nxt][3]:
            self._index_remove(dev, nxt, self.blocks[nxt][0])
            nb = self.blocks.pop(nxt)
            if r is not None and r.last == nxt:
                r.last = ptr
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
            if r is not None and r.last == ptr:
                r.last = prv
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
                if b is not None and b[1] is None and b[2] is None and _range_of(self, dev, p) is None:
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


# -- expandable ranges (R16) --------------------------------------------------------------------------------------------

class Range:
    """One reserved virtual address range of a device: [base, base + size), chunks of `chunk` bytes mapped into it as
    the pool grows (`mapped`: chunk indices), `top` the offset the next growth starts at, `last` the block ending at
    `top` (None while empty). The pool's large blocks of the range chain through it in address order, so a free block
    coalesces across growth boundaries; a chunk lying whole inside a free block can be unmapped and its physical
    memory mapped again elsewhere — the remainders a live neighbour pins give their pages back (R16)."""

    def __init__(self, base: int, size: int, chunk: int) -> None:
        self.base, self.size, self.chunk = int(base), int(size), int(chunk)
        self.top = 0
        self.last: Optional[int] = None
        self.mapped: set = set()

    def holds(self, ptr: int) -> bool:
        return self.base <= ptr < self.base + self.size

    def chunks_touching(self, lo: int, hi: int) -> range:
        return range((lo - self.base) // self.chunk, -(-(hi - self.base) // self.chunk))

    def chunks_within(self, lo: int, hi: int) -> range:
        return range(-(-(lo - self.base) // self.chunk), (hi - self.base) // self.chunk)

    def chunk_va(self, i: int) -> int:
        return self.base + i * self.chunk

    @property
    def mapped_bytes(self) -> int:
        return len(self.mapped) * self.chunk


def _range_of(pool: "SegmentPool", dev: int, ptr: int) -> Optional[Range]:
    for r in pool.ranges.get(dev, ()):
        if r.holds(ptr):
            return r
    return None


def range_add(pool: "SegmentPool", dev: int, base: int, size: int, chunk: int) -> Range:
    """A reserved range joins the device's large class."""
    r = Range(base, size, chunk)
    pool.ranges.setdefault(dev, []).append(r)
    return r


def range_extend(pool: "SegmentPool", dev: int, nbytes: int) -> Optional[Tuple[Range, int, int]]:
    """Grow the device's last range by `nbytes` (rounded to the pool's align) at its top: the new bytes join the
    block ending at the top when it is free (one block, coalesced), else become a free block chained after it.
    Returns (range, lo, hi) — the caller maps the chunks touching [lo, hi) not yet mapped — or None when no range of
    the device has the room (the caller reserves another)."""
    n = -(-int(nbytes) // pool.align) * pool.align
    rs = pool.ranges.get(dev) or []
    r = rs[-1] if rs else None
    if r is None or r.top + n > r.size:
        return None
    lo = r.base + r.top
    last = r.last
    if last is not None and pool.blocks[last][3]:
        blk = pool.blocks[last]
        pool._index_remove(dev, last, blk[0])
        blk[0] += n
        pool._index_add(dev, last, blk[0])
        pool.on_block(last, blk[0], dev)
    else:
        pool.blocks[lo] = [n, last, None, True, dev, LARGE]
        if last is not None:
            pool.blocks[last][2] = lo
        pool._index_add(dev, lo, n)
        pool.on_block(lo, n, dev)
        r.last = lo
    r.top += n
    pool.cached[dev] = pool.cached.get(dev, 0) + n
    return r, lo, lo + n


def range_unmapped(pool: "SegmentPool", dev: int, ptr: int, nbytes: int) -> List[Tuple[Range, int]]:
    """The (range, chunk) pairs a block [ptr, ptr + nbytes) touches that are not mapped (unmapped by a reclaim, or
    the growth's own new chunks)."""
    r = _range_of(pool, dev, ptr)
    if r is None:
        return []
    return [(r, i) for i in r.chunks_touching(ptr, ptr + nbytes) if i not in r.mapped]


def unmap_over_cap(pool: "SegmentPool", dev: int, parked_cap: int, unmap) -> int:
    """Unmap reclaimable chunks (`range_reclaimable`) while the bytes mapped under free blocks exceed `parked_cap`;
    `unmap(range, chunk)` releases each; the bytes unmapped."""
    out = 0
    pairs = range_reclaimable(pool, dev)
    mapped_free = len(pairs) * (pairs[0][0].chunk if pairs else 0)
    for r, i in pairs:
        if mapped_free <= parked_cap:
            break
        unmap(r, i)
        r.mapped.discard(i)
        mapped_free -= r.chunk
        out += r.chunk
    return out


def range_reclaimable(pool: "SegmentPool", dev: int) -> List[Tuple[Range, int]]:
    """The mapped chunks lying WHOLE inside free blocks of the device's ranges: what an unmap gives back without
    touching a live byte."""
    out = []
    for r in pool.ranges.get(dev, ()):
        for ptr, blk in pool.blocks.items():
            if blk[3] and blk[4] == dev and r.holds(ptr):
                out.extend((r, i) for i in r.chunks_within(ptr, ptr + blk[0]) if i in r.mapped)
    return out


# -- the same rules, replayed -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Simulation:
    """What the allocator holds for a sequence of allocations and frees: `peak_bytes` from the driver (live blocks, the
    free blocks of segments something alive pins, and parked whole segments), `peak_live_bytes` of the tensors alive,
    `parked_bytes` the most parked at once."""
    peak_bytes: int
    peak_live_bytes: int
    parked_bytes: int
    #: requests the card of `cap` bytes refused even after the reclaim (0 without a cap)
    refusals: int = 0


def simulate(events: Iterable[Tuple[str, object, int]], parked_cap: int = 0, align: int = POOL_ALIGN,
             split_min: int = POOL_SPLIT_MIN, pooled: bool = True, vmm_chunk: Optional[int] = None,
             cap: Optional[int] = None) -> Simulation:
    """Replay `events` — ("alloc", key, nbytes), ("arena", key, nbytes) and ("free", key, 0), `key` any hashable naming
    one allocation — through the allocator's own pool (`SegmentPool`) and its own order: a request is served from the
    pool first; on a miss the device grows by `grow_size` and the new segment is taken from; a freed block parks, and
    wholly free segments beyond `parked_cap` (0: unbounded) return to the driver. An "arena" is a block outside the
    pool (`malloc_cuda(pooled=False)`, a weight arena): straight from the driver, back to it whole when freed.
    `pooled=False`: the pool is off (`NBX_ALLOC_POOL=0`), every free returns to the driver. One device; the driver is a
    byte counter that never refuses (what a refusal flushes is the parked whole segments, counted in `parked_bytes`).

    `vmm_chunk`: the device maps chunks of that many bytes into an expandable range for the large class (R16,
    `memory.vmm`): a large miss extends the range and maps the chunks it touches, a block taken over unmapped chunks
    maps them, and a refusal (`cap`: the card's bytes; None: never) unmaps every mapped chunk lying whole inside a free
    block before it is retried — the reclaim the allocator makes, after one synchronisation."""
    pool = SegmentPool(align, split_min)
    held = live = peak = peak_live = parked = refusals = 0
    rng = range_add(pool, 0, 1 << 44, 1 << 46, int(vmm_chunk)) if (pooled and vmm_chunk) else None

    def map_chunks(pairs) -> bool:
        """Map the (range, chunk) pairs, reclaiming once when the card would refuse; False when it still refuses."""
        nonlocal held
        need = len(pairs) * rng.chunk
        if cap is not None and held + need > cap:
            pool.flush([0], to_driver)
            for r, i in range_reclaimable(pool, 0):
                r.mapped.discard(i)
                held -= r.chunk
            if held + need > cap:
                return False
        for r, i in pairs:
            r.mapped.add(i)
        held += need
        return True
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
            if rng is not None and pool.cls(n) == LARGE:
                ptr = pool.take(0, n)
                if ptr is None:
                    _r, lo, hi = range_extend(pool, 0, pool.grow_size(n))
                    ptr = pool.take(0, n)
                ok = map_chunks(range_unmapped(pool, 0, ptr, pool.alloc_size.get(ptr, n)))
                pool.alloc_size.pop(ptr, None)
                if not ok:
                    refusals += 1
                    pool.park(0, pool.release(ptr))
                    continue
                where[key] = (ptr, n)
                live += n
                peak = max(peak, held)
                peak_live = max(peak_live, live)
                continue
            ptr = pool.take(0, n) if pooled else None
            if ptr is None:
                grown = pool.grow_size(n) if pooled else n
                if cap is not None and held + grown > cap and pooled:
                    pool.flush([0], to_driver)                 # the refusal's flush, then the retry
                    if rng is not None:
                        for r, i in range_reclaimable(pool, 0):
                            r.mapped.discard(i)
                            held -= r.chunk
                if cap is not None and held + grown > cap:
                    refusals += 1
                    continue
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
                    if rng is not None:
                        unmap_over_cap(pool, 0, parked_cap, lambda r, i: None)
                        held = sum(seg_bytes.values()) + rng.mapped_bytes
            else:
                held -= seg_bytes.pop(ptr)
        else:
            raise ValueError(f"simulate: an event is ('alloc' | 'free', key, nbytes), not {kind!r}")
    return Simulation(peak, peak_live, parked, refusals)
