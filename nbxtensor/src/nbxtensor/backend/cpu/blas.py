"""The thread count of the BLAS numpy calls, from the pool's size, with no oversubscription (DESIGN rule 14).

numpy's matrix products run in the BLAS it was built with, which has its own threads. Two rules:

- outside the pool, the BLAS uses as many threads as the pool would (`pool.threads()`: the machine's compute cores);
- inside `parallel_for`, each piece already runs on a pool thread, so the BLAS a piece calls runs single-threaded:
  pool threads x BLAS threads would put more threads on the cores than there are cores.

Through the BLAS's own symbols, found in the process (no new dependency):

- Apple's Accelerate (`BLASSetThreading`, macOS 15+): a setting PER THREAD, multi-threaded (Accelerate chooses how
  many) or single-threaded — the pool's threads are set single-threaded when they start;
- OpenBLAS (numpy's wheels on Linux; `openblas_set_num_threads` / `scipy_openblas_set_num_threads64_`): a count for
  the PROCESS — set to 1 while `parallel_for` runs, the pool's size otherwise.

A BLAS none of these names is refused by name when its threads are set: the count is never assumed.
"""
from __future__ import annotations

import ctypes
import functools
import platform
from pathlib import Path
from typing import Optional, Tuple

__all__ = ["blas", "set_blas_threads", "blas_threads_of_this_thread"]

_ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib"
_OPENBLAS_SET = ("scipy_openblas_set_num_threads64_", "scipy_openblas_set_num_threads", "openblas_set_num_threads64_",
                 "openblas_set_num_threads")
_OPENBLAS_GET = ("scipy_openblas_get_num_threads64_", "scipy_openblas_get_num_threads", "openblas_get_num_threads64_",
                 "openblas_get_num_threads")


def _loaded_openblas() -> Optional[str]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return None
    for line in maps.read_text().splitlines():
        path = line.split()[-1] if line.split() else ""
        if "openblas" in Path(path).name.lower():
            return path
    return None


@functools.lru_cache(maxsize=1)
def blas() -> Tuple[str, object]:
    """(kind, library) of the BLAS numpy runs in this process: ("accelerate", libBLAS) or ("openblas", the loaded
    library); refused by name otherwise."""
    import numpy  # noqa: F401 — numpy loads its BLAS; the library is the one already in the process
    if platform.system() == "Darwin":
        # a system library lives in the dyld shared cache, not as a file on disk: it is loaded, never stat-ed
        try:
            lib = ctypes.CDLL(_ACCELERATE)
        except OSError:
            lib = None
        if lib is not None and hasattr(lib, "BLASSetThreading") and hasattr(lib, "BLASGetThreading"):
            return "accelerate", lib
    path = _loaded_openblas()
    if path is not None:
        lib = ctypes.CDLL(path)
        if any(hasattr(lib, s) for s in _OPENBLAS_SET):
            return "openblas", lib
    raise RuntimeError("ZERO FALLBACK: nbxtensor knows no thread control of this process's BLAS (Accelerate's "
                       "BLASSetThreading, OpenBLAS's openblas_set_num_threads); its threads are not set")


def _symbol(lib, names):
    for name in names:
        if hasattr(lib, name):
            return getattr(lib, name)
    raise RuntimeError(f"ZERO FALLBACK: the BLAS exports none of {names}")


def set_blas_threads(n: int) -> None:
    """The BLAS threads: `n` (OpenBLAS: for the process); Accelerate takes no count: 1 is single-threaded for the
    calling thread, more is Accelerate's own multi-threading for it."""
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"set_blas_threads: {n!r} — one thread or more")
    kind, lib = blas()
    if kind == "accelerate":
        fn = lib.BLASSetThreading
        fn.argtypes, fn.restype = [ctypes.c_uint], ctypes.c_int
        if fn(0 if n > 1 else 1) != 0:            # BLAS_THREADING_MULTI_THREADED = 0, _SINGLE_THREADED = 1
            raise RuntimeError("Accelerate refused BLASSetThreading on this platform (macOS 15 or later needed)")
        return
    fn = _symbol(lib, _OPENBLAS_SET)
    fn.argtypes, fn.restype = [ctypes.c_int], None
    fn(n)


def blas_threads_of_this_thread() -> Optional[int]:
    """1 when the BLAS runs single-threaded for the calling thread, else its count (OpenBLAS), or None for
    Accelerate's own multi-threading (it states no count)."""
    kind, lib = blas()
    if kind == "accelerate":
        fn = lib.BLASGetThreading
        fn.restype = ctypes.c_uint
        return 1 if fn() == 1 else None
    fn = _symbol(lib, _OPENBLAS_GET)
    fn.restype = ctypes.c_int
    return int(fn())
