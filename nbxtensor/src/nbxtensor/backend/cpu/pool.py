"""The CPU uses every core it is given (Hocine 2026-10-29 18:11): one thread pool, and a loop that cuts a large array
into contiguous pieces run in parallel, on the model of ggml.

numpy's element-wise ops, reductions and conversions run on one core; their inner loops release the GIL. So an op over
a large array is cut into contiguous pieces, each piece run by a pool thread with the same numpy call on its slice:
the pieces run on as many cores as the pool holds. Under a size threshold the threads cost more than they save, and
the op runs inline; the threshold is MEASURED (inline against split, on this machine, per kind of work), never written
by hand.

- `threads()`: the pool's size. Default: what this library measures of the machine — on Apple silicon its performance
  cores (`hw.perflevel0.physicalcpu`: an efficiency core in a parallel region holds every other thread at its barrier),
  else the physical cores the process may run on (Linux: its CPU affinity and `/sys` topology; macOS:
  `hw.physicalcpu`). An application sets its own (`set_threads`; NeuroBrix: its hardware profile's compute threads).
- `parallel_for(n, run, kind)`: `run(start, stop)` over contiguous pieces of `[0, n)`, inline under the measured
  threshold of `kind`; returns the pieces' results in order.
- the BLAS numpy calls: as many threads as the pool outside it, one per piece inside `parallel_for` — no
  oversubscription (`backend.cpu.blas`).
- `numa_nodes()`: the NUMA nodes and their CPUs (Linux `/sys/devices/system/node`), `{}` where the machine has none to
  report.
"""
from __future__ import annotations

import concurrent.futures
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["threads", "set_threads", "measured_threads", "numa_nodes", "pieces", "parallel_for",
           "split_threshold", "set_split_threshold"]

_LOCK = threading.Lock()
_STATE: Dict[str, object] = {}          # "threads": int set by an application; "pool": the executor; "pool_size"
_THRESHOLDS: Dict[str, int] = {}        # kind -> elements under which the work runs inline


def _sysctl_int(name: str) -> Optional[int]:
    r = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True)
    out = r.stdout.strip()
    return int(out) if r.returncode == 0 and out.isdigit() else None


def _linux_physical_cores() -> Optional[int]:
    allowed = os.sched_getaffinity(0)
    cores = set()
    for cpu in allowed:
        topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            cores.add((int((topo / "physical_package_id").read_text()), int((topo / "core_id").read_text())))
        except (OSError, ValueError):
            return None
    return len(cores) or None


def measured_threads() -> int:
    """The compute cores this machine gives this process, as the library reads them (see the module's docstring).
    Refused by name where no reading exists: the count is never guessed."""
    system = platform.system()
    if system == "Darwin":
        n = _sysctl_int("hw.perflevel0.physicalcpu") or _sysctl_int("hw.physicalcpu")
    elif system == "Linux":
        n = _linux_physical_cores()
    else:
        n = None
    if not n:
        raise RuntimeError(f"ZERO FALLBACK: nbxtensor reads no physical core count on {system!r}; set the pool's size "
                           f"(nbxtensor.backend.cpu.pool.set_threads)")
    return int(n)


def numa_nodes() -> Dict[int, Tuple[int, ...]]:
    """{node: CPUs} from Linux's `/sys/devices/system/node`; `{}` where the machine reports none."""
    root = Path("/sys/devices/system/node")
    nodes: Dict[int, Tuple[int, ...]] = {}
    for d in sorted(root.glob("node[0-9]*")) if root.is_dir() else ():
        cpus: List[int] = []
        for part in (d / "cpulist").read_text().strip().split(","):
            if part:
                lo, _, hi = part.partition("-")
                cpus.extend(range(int(lo), int(hi or lo) + 1))
        nodes[int(d.name[4:])] = tuple(cpus)
    return nodes


def threads() -> int:
    """The pool's size: the application's (`set_threads`), else the library's measurement of the machine."""
    n = _STATE.get("threads")
    return int(n) if n is not None else measured_threads()


def set_threads(n: Optional[int]) -> None:
    """Set the pool's size (an int >= 1); `None` restores the library's measurement. The running pool is replaced."""
    if n is not None and (not isinstance(n, int) or n < 1):
        raise ValueError(f"set_threads: {n!r} — a pool holds one thread or more")
    with _LOCK:
        if n is None:
            _STATE.pop("threads", None)
        else:
            _STATE["threads"] = n
        pool = _STATE.pop("pool", None)
        _STATE.pop("pool_size", None)
    if pool is not None:
        pool.shutdown(wait=True)
    from nbxtensor.backend.cpu.blas import blas, set_blas_threads
    if blas()[0] == "openblas":                       # the process's BLAS count follows the pool's size
        set_blas_threads(threads())


def _pool(size: int) -> concurrent.futures.ThreadPoolExecutor:
    with _LOCK:
        if _STATE.get("pool_size") != size:
            old = _STATE.pop("pool", None)
            if old is not None:
                old.shutdown(wait=False)
            _STATE["pool"] = concurrent.futures.ThreadPoolExecutor(max_workers=size, thread_name_prefix="nbx-cpu",
                                                                   initializer=_one_blas_thread_here)
            _STATE["pool_size"] = size
        return _STATE["pool"]


def _one_blas_thread_here() -> None:
    """A pool thread runs its BLAS single-threaded (Accelerate's setting is per thread; OpenBLAS's count is the
    process's and is lowered by `parallel_for` itself)."""
    from nbxtensor.backend.cpu.blas import blas, set_blas_threads
    if blas()[0] == "accelerate":
        set_blas_threads(1)


def pieces(n: int, parts: int) -> List[Tuple[int, int]]:
    """`[0, n)` cut into at most `parts` contiguous, non-empty pieces of sizes differing by one at most, in order."""
    if n < 0 or parts < 1:
        raise ValueError(f"pieces: n={n}, parts={parts}")
    parts = min(parts, n) or 1
    base, extra = divmod(n, parts)
    out, start = [], 0
    for i in range(parts):
        stop = start + base + (1 if i < extra else 0)
        if stop > start:
            out.append((start, stop))
        start = stop
    return out


def split_threshold(kind: str, probe: Optional[Callable[[int], Callable[[int, int], object]]] = None) -> int:
    """Elements of work of `kind` under which a split costs more than it saves — set (`set_split_threshold`) or
    MEASURED once per process with `probe(n)` (which builds the work on `n` elements: a callable of (start, stop)),
    by timing inline against split at growing sizes. A kind never set nor probed is refused by name."""
    if kind in _THRESHOLDS:
        return _THRESHOLDS[kind]
    if probe is None:
        raise KeyError(f"split_threshold: kind {kind!r} has no threshold set and no probe to measure it")
    size = threads()
    n = 1 << 12
    while n <= 1 << 26:
        run = probe(n)
        inline = _best_of(lambda: run(0, n))
        split = _best_of(lambda: [f.result() for f in [_pool(size).submit(run, a, b) for a, b in pieces(n, size)]])
        if split < inline:
            _THRESHOLDS[kind] = n
            return n
        n <<= 1
    _THRESHOLDS[kind] = n          # no size measured faster split: work this large or smaller runs inline
    return n


def _best_of(fn, repeat: int = 5) -> float:
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def set_split_threshold(kind: str, n: Optional[int]) -> None:
    """Set (or with `None` forget, to measure again) the threshold of `kind`."""
    if n is None:
        _THRESHOLDS.pop(kind, None)
    elif not isinstance(n, int) or n < 1:
        raise ValueError(f"set_split_threshold: {n!r}")
    else:
        _THRESHOLDS[kind] = n


def parallel_for(n: int, run: Callable[[int, int], object], kind: str,
                 probe: Optional[Callable[[int], Callable[[int, int], object]]] = None) -> Sequence[object]:
    """`run(start, stop)` over contiguous pieces of `[0, n)`: inline (one piece) under `kind`'s threshold or on a pool
    of one thread, else one piece per pool thread. The pieces' results, in order."""
    size = threads()
    if size == 1 or n < split_threshold(kind, probe):
        return [run(0, n)]
    from nbxtensor.backend.cpu.blas import blas, set_blas_threads
    process_wide = blas()[0] == "openblas"           # a count for the whole process: 1 while the pieces run
    if process_wide:
        set_blas_threads(1)
    try:
        futures = [_pool(size).submit(run, a, b) for a, b in pieces(n, size)]
        return [f.result() for f in futures]
    finally:
        if process_wide:
            set_blas_threads(size)
