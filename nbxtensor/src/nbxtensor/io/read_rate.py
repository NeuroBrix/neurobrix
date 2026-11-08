"""How fast a machine reads its weight files, by number of parallel reads — the measurement behind a hardware profile's
I/O worker count (`measure`). The reads are the loaders' own: pieces of `PIECE` bytes through `os.preadv` from a pool
of threads (`nbxtensor.io.safetensors.read_bytes`). Each trial evicts the files' cached pages first and bypasses the
cache while it reads (macOS `msync(MS_INVALIDATE)` + `F_NOCACHE`; Linux `posix_fadvise(DONTNEED)`), so the rate is the
storage path's (a disk, a RAID, an NFS or RDMA mount), never the page cache's."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

__all__ = ["PIECE", "WITNESS_BYTES", "START_BYTES", "mount_of", "pieces", "trial", "choose", "measure", "workers_for",
           "can_bypass_the_cache"]

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


def can_bypass_the_cache() -> bool:
    """Whether this system lets a measure read past its page cache (macOS F_NOCACHE + msync, Linux posix_fadvise);
    Windows offers neither through Python, and a rate read from the cache is not the storage's."""
    return sys.platform == "darwin" or hasattr(os, "posix_fadvise")


def trial(pieces, workers: int) -> float:
    """Bytes per second reading `pieces` with `workers` threads, the cache bypassed."""
    from nbxtensor.io.safetensors import _read_at
    for path in {p for p, _, _ in pieces}:
        _evict(path, os.path.getsize(path))
    fds = {}
    for path, _, _ in pieces:
        if path not in fds:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            _no_cache(fd)
            fds[path] = fd
    bufs = [bytearray(PIECE) for _ in range(workers)]

    def read(i):
        path, off, n = pieces[i]
        buf = memoryview(bufs[i % workers])[:n]
        got = 0
        while got < n:
            got += _read_at(fds[path], buf[got:], off + got)
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


def read_seconds(directory, nbytes: int, sample: int = 256 << 20) -> float:
    """Seconds this storage takes to read `nbytes`, at the rate one read of up to `sample` bytes of the weight files
    under `directory` measures now (one worker, the page cache bypassed)."""
    ps, _total = pieces(sorted(str(f) for f in Path(directory).rglob("*.safetensors")), sample)
    return nbytes / trial(ps, 1)


WITNESS_BYTES = 1 << 30                # the witness file written when the storage holds no weight file yet
START_BYTES = 2 << 30                  # the first trial: one shard's read, at most this


def mount_of(path) -> str:
    """The mount point the file `path` lives on: the storage a read of it goes through."""
    p = os.path.realpath(path)
    while not os.path.ismount(p):
        p = os.path.dirname(p)
    return p


def workers_for(directory, witness_dir=None, max_workers: Optional[int] = None,
                pattern: str = "*/components/*/weights/*.safetensors") -> Dict[str, object]:
    """The parallel reads to keep in flight on the storage a weight cache is read from, MEASURED there:
    {"workers": n, "storage": the mount point read, "read": "shards" or "witness"}.

    The files read are `directory`'s largest matching `pattern` (the cache's shards). When it holds none yet, a witness
    of `WITNESS_BYTES` is written in `witness_dir` (a local scratch directory the caller names, on the local cache's
    storage when there is one) and removed — NEVER in `directory`, which may be a cache shared over a network; with no
    `witness_dir`, refused by name. The answer says which storage it read. Counts 1, 2, 4, ... up to `max_workers`
    (the machine's `os.cpu_count()` when not given).

    The first trial reads one shard (the largest file, at most `START_BYTES`): a loader reads whole shards, and a
    shorter trial undervalues parallel reads (64 MiB picked 4 on the rack's RDMA cache, 2 GiB 8/16/16 — the count
    the planner prices host memory by is the one a load sees). The trials then grow while the measure is not stable:
    at a given size, up to three measures; when two of them are steady (`measure`'s first/last one-worker check) the
    answer is their median — one draw fluctuates and a plan prices the count. Otherwise the trial size is quadrupled,
    up to everything the files hold. When no size gives a stable measure, refused by name: never a silent None, never
    a default."""
    from pathlib import Path
    if not can_bypass_the_cache():
        raise RuntimeError(f"ZERO FALLBACK: no uncached read-rate measure on this system ({sys.platform}: no way to "
                           f"read past the page cache) — no worker count is assumed")
    root = Path(directory)
    max_workers = max_workers or os.cpu_count() or 1
    files = [str(p) for p in sorted(root.glob(pattern), key=lambda p: -p.stat().st_size)[:4]]
    witness = None
    if not files:
        if witness_dir is None:
            raise RuntimeError(f"ZERO FALLBACK: {root} holds no weight file to measure its reads on, and no local "
                               f"directory was named for a witness file (it is never written into the cache itself)")
        Path(witness_dir).mkdir(parents=True, exist_ok=True)
        witness = Path(witness_dir) / f".read_rate_witness_{os.getpid()}"
        with open(witness, "wb") as f:
            for _ in range(WITNESS_BYTES // PIECE):
                f.write(os.urandom(PIECE))
        files = [str(witness)]
    try:
        available = sum(os.path.getsize(p) for p in files)
        # A loader reads whole shards: a trial shorter than one is a rate no load sees.
        size, tried = max(4 * PIECE, min(max(os.path.getsize(p) for p in files), START_BYTES)), []
        while True:
            size = min(size, available)
            picks = []
            for _ in range(3):
                m = measure(files, size, max_workers)
                if m["steady"]:
                    picks.append(int(m["workers"]))
                if len(picks) == 2:
                    break
            tried.append(f"{size >> 20} MiB: {len(picks)}/3 steady")
            if len(picks) >= 2:
                n = picks[0] if picks[0] == picks[1] else _median_of_three(files, size, max_workers, picks)
                return {"workers": n, "storage": mount_of(files[0]), "read": "witness" if witness else "shards"}
            if size >= available:
                raise RuntimeError(
                    f"ZERO FALLBACK: no stable read-rate measure on {mount_of(files[0])} up to {available >> 20} MiB per "
                    f"trial ({'; '.join(tried)}): another reader on the storage? measure again on a quiet machine "
                    f"(tools/measure_io_workers.py) — no worker count is assumed")
            size *= 4
    finally:
        if witness is not None:
            witness.unlink(missing_ok=True)


def _median_of_three(files, size, max_workers, picks):
    """Two steady draws disagree: a third steady one decides (their median)."""
    for _ in range(3):
        m = measure(files, size, max_workers)
        if m["steady"]:
            return sorted(picks + [int(m["workers"])])[1]
    return sorted(picks)[0]
