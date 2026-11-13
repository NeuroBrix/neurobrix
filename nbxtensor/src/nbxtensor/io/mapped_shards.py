"""Weight files mapped once each, the host views over spans inside them, and the pages a view's span gives back.

A server that keeps some weights in their files (routed experts on a unified device) maps each shard once and builds
host views over the stored bytes; nothing is read until a copy touches the pages. Once a view's bytes sit in a device
slot, its span's pages are given back to the kernel (`madvise(MADV_DONTNEED)` over the page-aligned range): a private,
never written mapping re-reads the file on the next touch, and the copy's pages no longer count against the machine
(the Mac, 2026-10-14, a 30B mixture of experts with 55 GB of mapped experts: the process flat at 8.6 GB while free
memory fell and swap grew 7 GB, until the pages were given back).

A view is found by the address of its first byte: whoever copies it may hand back the same bytes through another
tensor object, never other bytes.
"""
from __future__ import annotations

import mmap
from typing import Dict, Tuple


def drop_pages(m: mmap.mmap, offset: int, nbytes: int) -> Tuple[int, int]:
    """The page-aligned range around `nbytes` at `offset` in the mapping given back to the kernel; returns (start,
    length). A platform whose mmap cannot advise is refused by name, never left to swap."""
    if not hasattr(m, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
        raise RuntimeError("ZERO FALLBACK: this platform's mmap carries no madvise/MADV_DONTNEED — a mapped weight's "
                           "pages cannot be given back after a copy")
    start = offset - offset % mmap.PAGESIZE
    length = (offset + nbytes) - start
    m.madvise(mmap.MADV_DONTNEED, start, length)
    return start, length


class ShardMaps:
    """Each shard mapped once for the views' lifetime, and the spans its views were built over."""

    def __init__(self, access: int = mmap.ACCESS_COPY):
        # ACCESS_COPY (private, copy-on-write, never written) where the views must be writable objects (a torch
        # tensor over the buffer); ACCESS_READ (shared, read-only) where the GPU reads the pages in place: a no-copy
        # Metal buffer over a PRIVATE mapping made copy-on-write copies of every page it bound (8 GB bound in 5.5 MB
        # pieces: the compressor 0.3 -> 11.7 GB; the same over a read-only mapping: flat, the Mac 2026-10-20)
        self._maps: Dict[str, mmap.mmap] = {}
        self._access = access
        self._spans: Dict[int, Tuple[mmap.mmap, int, int]] = {}

    def map(self, shard: str) -> mmap.mmap:
        m = self._maps.get(shard)
        if m is None:
            with open(shard, "rb") as f:
                m = mmap.mmap(f.fileno(), 0, access=self._access)
            self._maps[shard] = m
        return m

    def __len__(self) -> int:
        return len(self._maps)

    def note(self, view, shard: str, offset: int, nbytes: int) -> None:
        """`view` (any tensor with `data_ptr()`) is over `nbytes` at `offset` in `shard`: its pages can be given back."""
        self._spans[int(view.data_ptr())] = (self.map(shard), int(offset), int(nbytes))

    def give_back(self, view) -> Tuple[int, int]:
        """The pages of the span `view` was noted over, given back (`drop_pages`). A tensor over no noted span is
        refused by name: nothing to give back, and nothing should have been copied from it."""
        span = self._spans.get(int(view.data_ptr()))
        if span is None:
            raise RuntimeError("ZERO FALLBACK: a copied host tensor is no view of a mapped weight file — nothing to "
                               "give back, and nothing should have been copied from it")
        return drop_pages(*span)
