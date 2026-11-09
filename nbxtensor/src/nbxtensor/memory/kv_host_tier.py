"""The KV cache's host tier — the shared, device-free half (design: the Mac 2026-11-09, reviewed by the Dell 19:05;
reste_apres_064 point 10).

One daemon serves several conversations: a conversation that leaves the card keeps its whole pages in host memory and,
when it comes back, they are copied back instead of recomputed; a prefix several conversations share (a system prompt, a
tool list) is held once. This module is the bookkeeping both engines share, the standard library only (as pages.py):

- `PrefixIndex`: the whole pages a cache holds, as a tree of page-aligned id blocks (a node = its parent and the ids of
  its page: an exact key, no hash to collide — vLLM's block hash chain, SGLang's radix tree). Each node says where its
  page lives (a device page of the PagePool, or a host page), how many running requests hold it, and when it was last
  used. A request's lookup returns the longest held prefix; its eviction chooses the least recently used LEAVES that no
  request holds, device pages first offloaded to the host, host pages then dropped.
- `HostPagePool`: which host pages are free. Each host page holds ONE page index of EVERY layer's K and V contiguous
  (vLLM's offloading connector: per-layer blocks of a few KB transfer 4-5x slower than one block of 0.5-2 MB).

The raw host memory (pinned on CUDA: DeviceAllocator.malloc_host_pinned; pageable on Metal, outside the GPU working set)
and the copies (one gather of a page over every layer into a staging buffer, one device-to-host copy) are per engine.
Exactness: a page enters only whole, its bytes are copied, and the match is by ids (kv_session.py): a restored prefix is
exactly what a device-held one is. The partial last page is never indexed; it is prefilled again.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

DEVICE, HOST = "device", "host"


class HostPagePool:
    """`capacity` host pages: which are free. Refused by name when none is (the caller drops a host leaf first)."""

    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError(f"HostPagePool: {capacity} pages")
        self.capacity = int(capacity)
        self._free: List[int] = list(range(self.capacity - 1, -1, -1))

    @property
    def free_pages(self) -> int:
        return len(self._free)

    def take(self) -> int:
        if not self._free:
            raise RuntimeError(f"ZERO FALLBACK: the KV host tier's {self.capacity} pages are all held")
        return self._free.pop()

    def grow(self, n: int) -> None:
        """`n` more pages (ids capacity .. capacity+n-1), free: the host memory grew by a slice."""
        first = self.capacity
        self.capacity += int(n)
        self._free[:0] = list(range(self.capacity - 1, first - 1, -1))

    def shrink(self, n: int) -> bool:
        """The last `n` pages removed when all are free (their slice is freed); False, nothing removed, otherwise."""
        tail = set(range(self.capacity - int(n), self.capacity))
        if n <= 0 or not tail <= set(self._free):
            return False
        self._free = [p for p in self._free if p not in tail]
        self.capacity -= int(n)
        return True

    def give(self, page: int) -> None:
        if not 0 <= page < self.capacity or page in self._free:
            raise RuntimeError(f"HostPagePool.give: host page {page} is not held")
        self._free.append(page)


def sequence_pages(tables, batch: int) -> List[int]:
    """The device pages of a serving cache's one sequence, in position order (a cache of the tier protocol)."""
    if batch != 1:
        raise RuntimeError(f"ZERO FALLBACK: the KV host tier serves one sequence; the cache holds {batch}")
    return list(tables[0].pages)


def host_tier_pages(budget_bytes: int, page_bytes_all_layers: int) -> int:
    """How many host pages a budget holds: whole pages only; 0 (the tier off) below one page."""
    if page_bytes_all_layers <= 0:
        raise ValueError(f"host_tier_pages: a page of {page_bytes_all_layers} bytes")
    return max(0, int(budget_bytes) // int(page_bytes_all_layers))


class _Node:
    __slots__ = ("parent", "ids", "where", "page", "refs", "stamp", "children")

    def __init__(self, parent: Optional["_Node"], ids: Tuple[int, ...], where: str, page: int, stamp: int):
        self.parent, self.ids, self.where, self.page, self.refs, self.stamp = parent, ids, where, page, 0, stamp
        self.children: Dict[Tuple[int, ...], "_Node"] = {}

    def chain(self) -> List["_Node"]:
        out, n = [], self
        while n is not None and n.ids:
            out.append(n)
            n = n.parent
        return out[::-1]


class PrefixIndex:
    """The whole pages of `tokens_per_page` positions one engine cache holds, device or host (see the module)."""

    def __init__(self, tokens_per_page: int):
        if tokens_per_page <= 0:
            raise ValueError(f"PrefixIndex: pages of {tokens_per_page} positions")
        self.tokens_per_page = int(tokens_per_page)
        self._root = _Node(None, (), DEVICE, -1, 0)
        self._clock = 0

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _blocks(self, ids: Sequence[int]) -> Iterator[Tuple[int, ...]]:
        tpp = self.tokens_per_page
        for i in range(len(ids) // tpp):
            yield tuple(int(t) for t in ids[i * tpp:(i + 1) * tpp])

    def _nodes(self) -> Iterator[_Node]:
        stack = list(self._root.children.values())
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children.values())

    def match(self, ids: Sequence[int]) -> List[_Node]:
        """The longest chain of whole pages of `ids` the index holds, device or host, in order; each is stamped used."""
        out, node = [], self._root
        for block in self._blocks(ids):
            node = node.children.get(block)
            if node is None:
                break
            node.stamp = self._tick()
            out.append(node)
        return out

    def insert(self, ids: Sequence[int], device_pages: Sequence[int]) -> List[_Node]:
        """Register the whole pages of `ids` held in `device_pages` (one per whole page, in order). A page already
        indexed is kept (its node returned); a new one becomes a device node. Returns the chain."""
        blocks = list(self._blocks(ids))
        if len(device_pages) < len(blocks):
            raise ValueError(f"PrefixIndex.insert: {len(blocks)} whole pages of ids, {len(device_pages)} device pages")
        out, node = [], self._root
        for block, page in zip(blocks, device_pages):
            child = node.children.get(block)
            if child is None:
                child = node.children[block] = _Node(node, block, DEVICE, int(page), self._tick())
            else:
                child.stamp = self._tick()
            out.append(child)
            node = child
        return out

    @staticmethod
    def hold(nodes: Sequence[_Node]) -> None:
        for n in nodes:
            n.refs += 1

    @staticmethod
    def let_go(nodes: Sequence[_Node]) -> None:
        for n in nodes:
            if n.refs <= 0:
                raise RuntimeError("PrefixIndex.let_go: a page no request holds")
            n.refs -= 1

    def _leaves(self, where: str) -> List[_Node]:
        """Nodes on `where` that no request holds and whose children all live farther down the tiers (a device node
        whose children are on the host or gone, a host node with no child left): evicting one never strands a child
        whose parent is gone. Least recently used first."""
        def free_below(n: _Node) -> bool:
            return all(c.where in (HOST, "moving") for c in n.children.values()) if where == DEVICE else not n.children
        return sorted((n for n in self._nodes() if n.where == where and n.refs == 0 and free_below(n)),
                      key=lambda n: n.stamp)

    def device_victims(self, n: int) -> List[_Node]:
        """Up to `n` device nodes to offload, least recently used leaves first (a parent becomes a leaf once its
        children moved: chosen in the same call)."""
        out: List[_Node] = []
        while len(out) < n:
            leaves = [x for x in self._leaves(DEVICE) if x not in out]
            if not leaves:
                break
            out.append(leaves[0])
            leaves[0].where = "moving"              # its parent sees it gone while the caller copies
        for x in out:
            x.where = DEVICE
        return out

    def offloaded(self, node: _Node, host_page: int) -> int:
        """`node`'s page now lives in `host_page`; returns the device page it left (free for the caller to reuse)."""
        if node.where != DEVICE:
            raise RuntimeError(f"PrefixIndex.offloaded: the node is on {node.where}")
        left, node.where, node.page = node.page, HOST, int(host_page)
        return left

    def restored(self, node: _Node, device_page: int) -> int:
        """`node`'s page is back in `device_page`; returns the host page it left."""
        if node.where != HOST:
            raise RuntimeError(f"PrefixIndex.restored: the node is on {node.where}")
        left, node.where, node.page = node.page, DEVICE, int(device_page)
        node.stamp = self._tick()
        return left

    def host_victims(self, n: int) -> List[_Node]:
        """Up to `n` host leaves to drop, least recently used first."""
        return self._leaves(HOST)[:n]

    def drop(self, node: _Node) -> int:
        """Forget a host leaf; returns its host page."""
        if node.where != HOST or node.children or node.refs:
            raise RuntimeError("PrefixIndex.drop: only a host leaf no request holds is dropped")
        del node.parent.children[node.ids]
        return node.page

    def forget_device(self, node: _Node) -> int:
        """Forget a device leaf (its device page is reused without an offload: the tier is off or full); returns it."""
        if node.where != DEVICE or node.children or node.refs:
            raise RuntimeError("PrefixIndex.forget_device: only a device leaf no request holds is forgotten")
        del node.parent.children[node.ids]
        return node.page

    def counts(self) -> Dict[str, int]:
        c = {DEVICE: 0, HOST: 0}
        for n in self._nodes():
            c[n.where] += 1
        return c
