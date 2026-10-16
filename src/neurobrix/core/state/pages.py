"""The state engine's pages — the shared, device-free half of step 2 (docs/design/state-engine.md §4.3 and §4.5).

State is held in fixed-size pages granted as a request grows, never at max_len. This module owns the bookkeeping both
engines share: which page holds which position (PageTable), who holds a page (reference counts, copy-on-write for a CFG
pair's shared prompt and beam prefixes), and when a released page may be granted again (PagePool's release queue).
The raw allocation behind the pages and the kernels that read through the table are per engine; nothing here touches
a device or imports torch (R33: the Triton branch imports this too).

Release is a queue event when it has to be (§3.3 point 3). In a fixed pool whose pages are written only by GPU work on
the same queue that reads them, a page granted again is written after the earlier reads in queue order, and release is
immediate (CUDA; both Metal engines as built, 2026-10-16). A device or path where a page is written outside that order
(by the host, or by a copy on another queue) declares `async_release`: a page whose last holder let it go waits in the
release queue under the fence the engine names, and returns to the free list only when the engine reports that fence
completed.
"""
from __future__ import annotations

from array import array
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class StateExhausted(RuntimeError):
    pass


class PagePool:
    """`num_pages` fixed pages of `tokens_per_page` positions. Grants pages, counts their holders, and frees them through
    the release queue."""

    def __init__(self, num_pages: int, tokens_per_page: int, async_release: bool):
        if num_pages <= 0 or tokens_per_page <= 0:
            raise ValueError(f"PagePool: {num_pages} pages of {tokens_per_page} positions")
        self.num_pages = int(num_pages)
        self.tokens_per_page = int(tokens_per_page)
        self.async_release = bool(async_release)
        self._free: List[int] = list(range(self.num_pages - 1, -1, -1))   # pop() grants page 0 first
        self._refs: List[int] = [0] * self.num_pages
        self._pending: List[Tuple[int, int]] = []                         # (fence, page), fences ascending
        self._completed_fence = -1

    @property
    def free_pages(self) -> int:
        return len(self._free)

    @property
    def pending_pages(self) -> int:
        return len(self._pending)

    def refs(self, page: int) -> int:
        return self._refs[page]

    def grant(self, n: int) -> List[int]:
        """`n` pages, each held once. Refused by name, granting nothing, when the free list cannot cover them; a page
        still waiting for its fence is not free."""
        if n > len(self._free):
            raise StateExhausted(
                f"ZERO FALLBACK: the state pool has {len(self._free)} free pages of {self.tokens_per_page} positions, "
                f"{n} asked ({len(self._pending)} wait for their release fence, {self.num_pages} in all)")
        pages = [self._free.pop() for _ in range(n)]
        for p in pages:
            self._refs[p] = 1
        return pages

    def grow(self, n: int) -> range:
        """`n` more pages, appended free at the end (ids num_pages .. num_pages+n-1): the engine has extended its raw
        allocation by as many pages (the compiled engine's pool growing toward its ceiling). Returns the new ids."""
        if n <= 0:
            raise ValueError(f"PagePool.grow: {n} pages")
        first = self.num_pages
        self.num_pages += int(n)
        self._refs.extend([0] * int(n))
        self._free[:0] = list(range(self.num_pages - 1, first - 1, -1))   # granted after the pages already free
        return range(first, self.num_pages)

    def hold(self, page: int) -> None:
        if self._refs[page] <= 0:
            raise RuntimeError(f"PagePool.hold: page {page} is not granted")
        self._refs[page] += 1

    def release(self, page: int, fence: Optional[int] = None) -> None:
        """One holder lets `page` go. When it was the last, the page is free at once on a synchronous device; on an
        asynchronous one it waits under `fence` (required: the engine's id of the last command that used the page)."""
        if self._refs[page] <= 0:
            raise RuntimeError(f"PagePool.release: page {page} is not granted")
        self._refs[page] -= 1
        if self._refs[page]:
            return
        if not self.async_release:
            self._free.append(page)
            return
        if fence is None:
            raise RuntimeError(f"PagePool.release: page {page} released on an asynchronous device without a fence")
        if fence <= self._completed_fence:
            self._free.append(page)
        else:
            self._pending.append((int(fence), page))

    def fence_completed(self, fence: int) -> int:
        """The engine reports every command up to `fence` completed: the pages waiting under it are free. Returns how
        many came back."""
        self._completed_fence = max(self._completed_fence, int(fence))
        back = [p for f, p in self._pending if f <= self._completed_fence]
        self._pending = [(f, p) for f, p in self._pending if f > self._completed_fence]
        self._free.extend(back)
        return len(back)


class PageTable:
    """One sequence's pages: position i lives in page `pages[i // tokens_per_page]` at slot `i % tokens_per_page`."""

    def __init__(self, pool: PagePool):
        self.pool = pool
        self.pages: List[int] = []
        self.length = 0

    def reserve(self, new_length: int) -> List[int]:
        """Grow to hold `new_length` positions, granting only the pages it lacks. Returns the pages granted."""
        tpp = self.pool.tokens_per_page
        need = -(-int(new_length) // tpp) - len(self.pages)
        granted = self.pool.grant(need) if need > 0 else []
        self.pages.extend(granted)
        self.length = max(self.length, int(new_length))
        return granted

    def slot(self, position: int) -> Tuple[int, int]:
        if not 0 <= position < self.length:
            raise IndexError(f"PageTable.slot: position {position} outside 0..{self.length - 1}")
        tpp = self.pool.tokens_per_page
        return self.pages[position // tpp], position % tpp

    def fork(self) -> "PageTable":
        """A second sequence on the same positions (a CFG pair's prompt, a beam): the pages are shared, each held once
        more, and copied only when one side writes into a shared page (`writable`)."""
        t = PageTable(self.pool)
        t.pages = list(self.pages)
        t.length = self.length
        for p in t.pages:
            self.pool.hold(p)
        return t

    def writable(self, position: int, fence: Optional[int] = None) -> Optional[Tuple[int, int]]:
        """Make the page holding `position` this sequence's own before a write. None when it already is; else
        (source page, new page): the engine copies the source's filled slots into the new page, and this table
        lets the source go (under `fence` on an asynchronous device)."""
        index = position // self.pool.tokens_per_page
        page = self.pages[index]
        if self.pool.refs(page) == 1:
            return None
        new = self.pool.grant(1)[0]
        self.pages[index] = new
        self.pool.release(page, fence)
        return page, new

    def free(self, fence: Optional[int] = None) -> None:
        for p in self.pages:
            self.pool.release(p, fence)
        self.pages, self.length = [], 0

    def slots(self, positions: Iterable[int]) -> array:
        """The flat slot of each position, page x tokens_per_page + slot (int32): where an append writes."""
        tpp = self.pool.tokens_per_page
        out = array("i")
        for pos in positions:
            page, s = self.slot(pos)
            out.append(page * tpp + s)
        return out

    def table(self) -> Dict[str, object]:
        """What a kernel reads: the page ids in position order, the positions held and the page size."""
        return {"pages": list(self.pages), "length": self.length, "tokens_per_page": self.pool.tokens_per_page}


def block_table(tables: Sequence[PageTable]) -> Tuple[array, int]:
    """What the paged attention read takes for a batch: one row per sequence of its page ids in position order, rows
    padded with -1 to the longest (int32, row-major), and the row width."""
    width = max((len(t.pages) for t in tables), default=0)
    out = array("i")
    for t in tables:
        out.extend(t.pages)
        out.extend([-1] * (width - len(t.pages)))
    return out, width
