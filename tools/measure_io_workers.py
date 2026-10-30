#!/usr/bin/env python3
"""How many parallel reads a machine's weight loaders should keep in flight — MEASURED, for the hardware profile's
`cpu.io_workers` (rule 1: a fact of the machine, never a number written by hand).

    python tools/measure_io_workers.py --files '<cache>/<model>/components/*/weights/*.safetensors' --max-gb 2

The loaders read a tensor's bytes with `os.preadv` from several threads (`nbxtensor.io.safetensors.read_bytes`), and
the rate that matters is the one through the machine's own storage path (an NVMe, a RAID, an NFS or RDMA mount). Each
trial reads the same bytes — real shard files, cut in the loaders' 16 MiB pieces — with N workers, the page cache
evicted before each trial and bypassed during it (macOS `msync(MS_INVALIDATE)` + `F_NOCACHE`; Linux
`posix_fadvise(DONTNEED)`), and reports the
rate. The answer is the SMALLEST worker count within `--within` (5 %) of the best rate: a worker beyond it holds a
shard in host memory for nothing (Prism's host estimate counts one per worker).

A trial whose rate is far above the storage path's (the cache was not bypassed) is said, never believed: the first and
the last trial at one worker are both run, and a gap past `--within` between them is reported.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

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


def _pieces(files, max_bytes):
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--files", required=True, help="glob of the shard files to read (the machine's real weights)")
    ap.add_argument("--max-gb", type=float, default=2.0, help="bytes read per trial")
    ap.add_argument("--max-workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--within", type=float, default=0.05)
    a = ap.parse_args(argv)
    files = sorted(glob.glob(a.files))
    if not files:
        print(f"no file matches {a.files!r}", file=sys.stderr)
        return 2
    pieces, total = _pieces(files, int(a.max_gb * (1 << 30)))
    counts, n = [], 1
    while n <= a.max_workers:
        counts.append(n)
        n *= 2
    print(f"[io_workers] {total / 2**30:.2f} GiB per trial from {len(files)} file(s), pieces of {PIECE >> 20} MiB")
    rates = {}
    first = trial(pieces, 1)
    for c in counts:
        rates[c] = trial(pieces, c) if c != 1 else first
        print(f"[io_workers] {c:>3} worker(s): {rates[c] / 2**20:8.1f} MiB/s")
    last = trial(pieces, 1)
    if abs(last - first) > a.within * max(first, last):
        print(f"[io_workers] WARNING: one worker read {first / 2**20:.0f} then {last / 2**20:.0f} MiB/s — the cache was "
              f"not bypassed or the storage is not steady; do not record this measure")
    best = max(rates.values())
    pick = choose(rates, a.within)
    print(f"cpu:\n  io_workers: {pick}   # measured {time.strftime('%Y-%m-%d')}: {rates[pick] / 2**20:.0f} MiB/s, "
          f"best {best / 2**20:.0f} MiB/s ({max(rates, key=rates.get)} workers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
