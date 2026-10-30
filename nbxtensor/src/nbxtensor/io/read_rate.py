"""How fast a machine reads its weight files, by number of parallel reads — the measurement behind a hardware profile's
I/O worker count (`measure`). The reads are the loaders' own: pieces of `PIECE` bytes through `os.preadv` from a pool
of threads (`nbxtensor.io.safetensors.read_bytes`). Each trial evicts the files' cached pages first and bypasses the
cache while it reads (macOS `msync(MS_INVALIDATE)` + `F_NOCACHE`; Linux `posix_fadvise(DONTNEED)`), so the rate is the
storage path's (a disk, a RAID, an NFS or RDMA mount), never the page cache's."""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

__all__ = ["PIECE", "pieces", "trial", "choose", "measure", "workers_for"]

PIECE = 16 << 20                       # the loaders' read piece


def _no_cache(fd: int) -> None:
    if sys.platform == "darwin":
        import fcntl
        fcntl.fcntl(fd, 48, 1)         # F_NOCACHE: this descriptor's reads add nothing to the unified buffer cache
    elif hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def _evict(path: str, size: int) -> None:
    """The file's pages out of the cache before a trial: F_NOCACHE alone still SERVES pages cached earlier (measured
    2026-10-30 on the Mac's NFS cache: 19 064 MiB/s cached, 32 MiB/s after this). macOS: msync(MS_INVALIDATE) on a
    shared read-only mapping; Linux: posix_fadvise(DONTNEED)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        if sys.platform == "darwin":
            import ctypes
            import ctypes.util
            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            libc.mmap.restype = ctypes.c_void_p
            libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_longlong]
            libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            addr = libc.mmap(None, size, 1, 1, fd, 0)                    # PROT_READ, MAP_SHARED
            if libc.msync(addr, size, 2) != 0:                           # MS_INVALIDATE
                raise OSError(ctypes.get_errno(), f"msync(MS_INVALIDATE) {path}")
            libc.munmap(addr, size)
        elif hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def pieces(files, max_bytes: int):
    """(path, offset, bytes) pieces of the loaders' size over `files`, in order, until `max_bytes`; and their total."""
    out, total = [], 0
    for path in files:
        size = os.path.getsize(path)
        for off in range(0, size, PIECE):
            n = min(PIECE, size - off)
            out.append((path, off, n))
            total += n
            if total >= max_bytes:
                return out, total
    return out, total


def trial(pieces, workers: int) -> float:
    """Bytes per second reading `pieces` with `workers` threads, the cache bypassed."""
    for path in {p for p, _, _ in pieces}:
        _evict(path, os.path.getsize(path))
    fds = {}
    for path, _, _ in pieces:
        if path not in fds:
            fd = os.open(path, os.O_RDONLY)
            _no_cache(fd)
            fds[path] = fd
    bufs = [bytearray(PIECE) for _ in range(workers)]

    def read(i):
        path, off, n = pieces[i]
        buf = memoryview(bufs[i % workers])[:n]
        got = 0
        while got < n:
            got += os.preadv(fds[path], [buf[got:]], off + got)
        return n

    t = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        total = sum(pool.map(read, range(len(pieces))))
    dt = time.perf_counter() - t
    for fd in fds.values():
        os.close(fd)
    return total / dt


def choose(rates, within: float) -> int:
    """The smallest worker count whose rate is within `within` of the best: a worker past it holds a shard in host
    memory for nothing."""
    best = max(rates.values())
    return min(c for c, r in rates.items() if r >= (1 - within) * best)


def measure(files: Sequence[str], max_bytes: int, max_workers: int, within: float = 0.05) -> Dict[str, object]:
    """Rates (bytes/s) at 1, 2, 4, ... `max_workers` parallel reads of `files` (at most `max_bytes` per trial), the
    worker count `choose` takes, and whether the storage was steady: the one-worker trial is run first and last, and a
    gap past `within` between them makes `steady` False — such a measure is not to be recorded."""
    ps, total = pieces(list(files), max_bytes)
    counts: List[int] = []
    n = 1
    while n <= max_workers:
        counts.append(n)
        n *= 2
    rates = {c: trial(ps, c) for c in counts}
    last = trial(ps, 1)
    steady = abs(last - rates[1]) <= within * max(last, rates[1])
    return {"bytes": total, "rates": rates, "workers": choose(rates, within), "steady": steady, "last_one_worker": last}


def workers_for(directory, max_bytes: int = 256 << 20, max_workers: Optional[int] = None,
                pattern: str = "*/components/*/weights/*.safetensors") -> Optional[int]:
    """The parallel reads to keep in flight on `directory`'s storage: `measure` over its largest files matching
    `pattern` (a weight cache's shards), else over a scratch file written there and removed. None when two measures
    in a row are not steady (another reader on the storage): nothing is returned that was not measured."""
    from pathlib import Path
    root = Path(directory)
    files = sorted(root.glob(pattern), key=lambda p: -p.stat().st_size)[:4]
    scratch = None
    if not files:
        scratch = root / f".read_rate_{os.getpid()}"
        with open(scratch, "wb") as f:
            for _ in range(max(1, max_bytes // PIECE)):
                f.write(os.urandom(PIECE))
        files = [scratch]
    try:
        for _ in range(2):
            m = measure([str(p) for p in files], max_bytes, max_workers or os.cpu_count() or 1)
            if m["steady"]:
                return int(m["workers"])
        return None
    finally:
        if scratch is not None:
            scratch.unlink(missing_ok=True)

