"""The device allocator: allocation, the pool and its parked blocks, the free queues, peer copies and the allocator's
own accounting (live bytes, peaks), with the malloc call-site trace (NBX_MALLOC_TRACE). Cut out of tensor.py (M2, 2026-10-29),
unchanged."""

from __future__ import annotations
from nbxtensor.backend import faults as _faults

import bisect
import ctypes
import os
import sys
from typing import Dict, List, Optional, Tuple

from nbxtensor.backend import runtime as _rt



# ============================================================================
# MALLOC CALL-SITE LOGGER — env-gated, zero cost when disabled
# ============================================================================
#
# When NBX_MALLOC_TRACE=<path> is set, every DeviceAllocator.malloc_cuda
# records one row and every free_cuda records the matching "F" row. The
# buffer is flushed to <path> at interpreter exit via atexit. When unset,
# the allocator pays a single module-level `is None` check per call — no
# frame walk, no list append.
#
# How to use:
#   NBX_MALLOC_TRACE=/tmp/run.tsv  neurobrix run --model ... --triton
#
# Row format (tab-separated):
#   event_id \t M|F \t ptr \t nbytes \t <file:line func>   (an F row carries the
#                                                           engine frames that freed it, innermost first)
#
# Recipes:
#   - Total allocation volume by site:
#       awk '$2=="M"{by[$5]+=$4} END{for(k in by)print by[k],k}' run.tsv \
#         | sort -rn | head
#   - Live-set composition at peak: walk events, track ptr→nbytes live
#     map, record the set at each new peak, then aggregate by site.
#     This is the discriminator between "leak" (unfreed allocs), "peak
#     concurrency" (tensors retained by a deferred-free mechanism), and
#     "fragmentation" (total freed > 80% but driver rejects next alloc).
#     In April 2026 this tool killed a wrong hypothesis in ~30 min on
#     PixArt triton (confirmed _deferred accumulation as root cause,
#     not workspace leak as initially hypothesised). Kept permanent —
#     same discriminator work will be needed for Qwen3 long-context,
#     Sana 4Kpx, video upscalers.
# ============================================================================

_MALLOC_TRACE_FILE: Optional[str] = os.environ.get("NBX_MALLOC_TRACE") or None
_MALLOC_TRACE_EVENTS: List[Tuple[int, str, int, int, str]] = []  # (eid, kind, ptr, nbytes, site)
_MALLOC_TRACE_COUNTER: List[int] = [0]


def _extract_neurobrix_site(max_frames: int = 12) -> str:
    """Return nearest neurobrix/ frame as 'file:line func', skipping this file.

    The logger only fires inside DeviceAllocator — we want the CALLER of the
    allocator (wrappers.py, dispatch.py, sequence.py, ...) not the allocator
    itself. Walk the stack until the first frame whose filename contains
    'neurobrix/' AND is NOT nbx_tensor.py.
    """
    f = sys._getframe(2)  # skip this fn + malloc_cuda/free_cuda
    n = 0
    while f is not None and n < max_frames:
        fname = f.f_code.co_filename
        if "neurobrix" in fname and not fname.endswith("/nbx_tensor.py"):
            rel = fname.split("neurobrix/")[-1]
            return f"{rel}:{f.f_lineno} {f.f_code.co_name}"
        f = f.f_back
        n += 1
    return "<unknown>"


def _record_malloc_site(ptr: int, nbytes: int, dev: int) -> None:
    eid = _MALLOC_TRACE_COUNTER[0]
    _MALLOC_TRACE_COUNTER[0] = eid + 1
    site = _extract_neurobrix_site()
    _MALLOC_TRACE_EVENTS.append((eid, "M", ptr, nbytes, site))


def _extract_neurobrix_chain(max_frames: int = 40, keep: int = 8) -> str:
    """The engine frames of the caller, innermost first, ' < '-joined — for
    a FREE row. A block freed by a finalizer names `free()` as its site,
    which says nothing; the frames above it name what dropped the last
    reference (Ming's arenas freed under live weights, 2026-09-16)."""
    f = sys._getframe(2)
    n = 0
    out = []
    while f is not None and n < max_frames and len(out) < keep:
        fname = f.f_code.co_filename
        if "neurobrix" in fname and not fname.endswith("/nbx_tensor.py"):
            rel = fname.split("neurobrix/")[-1]
            out.append(f"{rel}:{f.f_lineno} {f.f_code.co_name}")
        f = f.f_back
        n += 1
    return " < ".join(out) or "<unknown>"


def _record_free_site(ptr: int, nbytes: int) -> None:
    eid = _MALLOC_TRACE_COUNTER[0]
    _MALLOC_TRACE_COUNTER[0] = eid + 1
    _MALLOC_TRACE_EVENTS.append((eid, "F", ptr, nbytes, _extract_neurobrix_chain()))


def _flush_malloc_trace() -> None:
    path = _MALLOC_TRACE_FILE
    if path is None or not _MALLOC_TRACE_EVENTS:
        return
    try:
        with open(path, "w") as fh:
            for eid, kind, ptr, nbytes, site in _MALLOC_TRACE_EVENTS:
                fh.write(f"{eid}\t{kind}\t{ptr}\t{nbytes}\t{site}\n")
    except Exception:
        pass


import atexit as _atexit
_atexit.register(_flush_malloc_trace)

# The floor watchdog stops a run with SIGTERM (5 s grace before SIGKILL),
# and Python's default SIGTERM action skips atexit — so a floor-stopped
# traced run died with an EMPTY trace (measured: x8@1024, 2026-09-21).
# When, and only when, the trace is armed, convert SIGTERM to SystemExit
# so the flush runs inside the grace window. Unarmed runs keep the
# default signal behavior untouched.
if _MALLOC_TRACE_FILE:
    import signal as _signal_mt

    def _flush_on_term(_signum, _frame):
        raise SystemExit(143)

    try:
        _signal_mt.signal(_signal_mt.SIGTERM, _flush_on_term)
    except (ValueError, OSError):
        pass                       # non-main thread or exotic host: trace



class ImpossibleExtentError(RuntimeError):
    """An op would produce a tensor with an extent that cannot exist (<= 0 where the op defines
    a length, or a rank the op does not produce). Raised at the op, with its input shapes and its
    arithmetic, instead of letting an impossible tensor travel on and fail far away. The graph
    executors add the op uid and the bound symbol values."""


class DeviceOOMError(RuntimeError):
    """The device allocator could not honor a request: the driver refused
    the allocation after the deferred-free drain and the single retry.
    Typed so a caller that names its own key in the refusal (the chunked
    SDPA prefill re-raises it with op, shape, rows and bytes asked) catches
    exactly this and nothing else — every other RuntimeError keeps propagating.

    It also CARRIES the figures it prints, as attributes. The refusal already
    computes requested / live / pool-cached / driver-free in order to write the
    message; a caller that wants to reshape the work rather than give up then had
    to parse them back out of English. They are the input a controller needs to
    re-enter the placement cascade at the op-level tiling rung with the real free
    figure at the moment of failure instead of the estimate made before the run
    (`docs/reference/adaptive-memory-a-runtime-controller.md`, addition 3).

    Every field is bytes, and `None` where the runtime could not answer — never 0,
    which is a legitimate reading and would be acted on as one.
    """

    def __init__(self, message: str, *, requested: Optional[int] = None,
                 device_idx: Optional[int] = None, live: Optional[int] = None,
                 pool_cached: Optional[int] = None, pool_blocks: Optional[int] = None,
                 driver_free: Optional[int] = None, driver_total: Optional[int] = None):
        super().__init__(message)
        self.requested = requested
        self.device_idx = device_idx
        self.live = live
        self.pool_cached = pool_cached
        self.pool_blocks = pool_blocks
        self.driver_free = driver_free
        self.driver_total = driver_total

    @property
    def shortfall(self) -> Optional[int]:
        """How many bytes the request was short by, or None if the driver did
        not report its free figure. This is the number a reshape has to close."""
        if self.requested is None or self.driver_free is None:
            return None
        return max(0, self.requested - self.driver_free)

class AllocatorInterceptor:
    """What stands in front of the allocator's five device operations (allocate, free, copy, set, synchronise) while
    it is installed (`DeviceAllocator.install_interceptor`): each method receives the DIRECT operation it stands in
    front of (`DeviceAllocator.malloc_direct`, ...) and decides whether and how to call it. A replay recording carves
    its run's allocations from a slab and records the copies it must reproduce (`memory.slab.SlabInterceptor`).

    The defaults pass straight through; a subclass overrides only what it changes. One interceptor at a time: two
    would each believe it sees every allocation."""

    def malloc(self, direct, nbytes: int, dev_idx: Optional[int], pooled: bool) -> int:
        return direct(nbytes, dev_idx, pooled)

    def free(self, direct, ptr: int):
        return direct(ptr)

    def memcpy(self, direct, dst: int, src: int, nbytes: int, kind: int):
        return direct(dst, src, nbytes, kind)

    def memset(self, direct, ptr: int, value: int, nbytes: int):
        return direct(ptr, value, nbytes)

    def sync(self, direct):
        return direct()


class DeviceAllocator:
    """GPU + pinned-host memory allocator via raw runtime API.

    Tracks running byte counters per device and for pinned host memory
    so diagnostics (especially for zero3 CPU offload) don't need to
    round-trip through cudaMemGetInfo or torch.cuda.memory_allocated —
    both of which are unavailable or incorrect for NBXTensor.
    """

    # Running counters — populated by every malloc_cuda/free_cuda and
    # malloc_host_pinned/free_host_pinned. Class-level so the accounting
    # is process-wide. Not thread-safe; NeuroBrix runtime is
    # single-threaded per component by convention.
    _cuda_live_bytes: Dict[int, int] = {}            # device_idx -> live
    _cuda_peak_bytes: Dict[int, int] = {}            # device_idx -> peak
    _cuda_ptr_size: Dict[int, int] = {}              # ptr -> nbytes (for free accounting)
    _range_bases: list = []                          # live allocation bases, sorted (holds())
    _range_size: dict = {}                           # base -> bytes, mirrors the two size tables
    _cuda_ptr_device: Dict[int, int] = {}            # ptr -> device_idx (REQUIRED for cudaFreeAsync correctness)

    # Which generation of the device runtime issued a pointer.
    #
    # A pointer is only meaningful to the runtime that issued it. Normally
    # there is exactly one for the life of the process and this is a constant
    # 0 that never costs a thought. When a runtime IS replaced, every address
    # it issued dies with it — and the free-list must not outlive it, or a
    # later allocation is served an address the current runtime never issued
    # and the first launch that touches it refuses a perfectly good tensor.
    #
    # Flushing the pool at the swap is not enough: a tensor still ALIVE at
    # that moment is freed afterwards, and that free parks its dead address
    # for reuse. The epoch is what the flush cannot see.
    _alloc_epoch: int = 0
    _cuda_ptr_epoch: Dict[int, int] = {}             # ptr -> the epoch that issued it
    _host_pinned_live_bytes: int = 0
    _host_pinned_peak_bytes: int = 0
    _host_pinned_ptr_size: Dict[int, int] = {}       # ptr -> nbytes

    # Phase 2 — caching free-list pool. Per-device map of nbytes → list of
    # cached free pointers. On free, the pointer is returned to the pool
    # instead of cudaFree (which would re-fragment the driver heap). On
    # malloc, the pool is checked first (exact size match, then smallest-
    # fit ≥ requested). On cudaMalloc OOM, the entire pool is flushed back
    # to the driver via cudaFree and the malloc is retried — this gives
    # the driver a chance to coalesce its internal heap before declaring
    # failure (analog to torch CachingAllocator's release-cached-blocks
    # path on OOM). Default ON since 2026-09-02 (P-PREFILL-TRITON lever 1:
    # the xlong triton prefill paid one cudaMalloc + one cudaFree per op
    # output, 82 s of host API time on a 172 s run; the pool cut the cold
    # run by 41 % with byte-identical output, gated by the full-zoo
    # battery + the warm-serve sweep). `NBX_ALLOC_POOL=0` disables it
    # (the differential for any allocator-suspected divergence).
    # P-SANA-4KPX-RUNTIME Phase 2 origin.
    _pool_free: Dict[int, Dict[int, list]] = {}      # device_idx -> {nbytes: [ptr, ...]}  (the FREE blocks, by size)
    # The pool SPLITS and COALESCES (S4a, 2026-10-27; the method of torch's caching allocator): every driver allocation
    # is a SEGMENT made of BLOCKS chained in address order; a request takes the smallest free block that holds it and
    # splits off the rest when it is at least `_POOL_SPLIT_MIN`; a freed block merges with its free neighbours; only a
    # segment wholly free returns to the driver (over the parked cap, on a refused malloc, on a flush). Exact-size
    # parking kept a freed 3 GB block from ever serving a 2 GB request: 2.6-3.7 GB parked beside 6-7 GB live on
    # mochi-1-preview triton at the 8 GB rung.
    _blocks: Dict[int, list] = {}                    # ptr -> [size, prev ptr, next ptr, free, device]
    _pool_sizes: Dict[int, list] = {}                # device_idx -> sorted sizes present in _pool_free[dev]
    _POOL_ALIGN = 512
    _POOL_SPLIT_MIN = 1 << 20
    _pool_alloc_size: Dict[int, int] = {}            # ptr -> ALLOCATED nbytes (may exceed requested when smallest-fit is bigger)
    _pool_enabled: bool = False                      # set by _maybe_init_pool() from env
    #: Callables `(dev) -> bytes released` the engine registers for memory it caches outside the pool (a recorded
    #: replay plan's slab, `triton/replay.release_idle_slabs`), called on a malloc refused after the pool's flush.
    _idle_reclaimers: list = []
    #: The `AllocatorInterceptor` in force, or None: the five public device operations go through it.
    _interceptor: Optional[AllocatorInterceptor] = None
    # Pool observability (NBX_ALLOC_STATS=1 prints it at exit): bytes
    # parked in the free-list per device (+ peak), hit/miss counters,
    # and the smallest-fit SLACK (allocated − requested) — the pool's
    # own contribution to the driver-level watermark.
    _pool_cached_bytes: Dict[int, int] = {}          # device_idx -> bytes parked in the free-list
    _pool_cached_peak: Dict[int, int] = {}           # device_idx -> peak of the above
    _pool_stats: Dict[str, int] = {"exact": 0, "fit": 0, "miss": 0,
                                   "slack_total": 0, "slack_max": 0,
                                   "flushes": 0, "flushed_bytes": 0,
                                   "evictions": 0, "evicted_bytes": 0}
    # Parked-bytes cap per device (bytes; 0 = unbounded), resolved once per
    # device from the arch yml `memory.alloc_pool_parked_cap_fraction` ×
    # the device's total memory (ONE cudaMemGetInfo at the first free on
    # that device, never in the hot path afterwards).
    _pool_cap_bytes: Dict[int, int] = {}
    # A rung imposed on the plan (`NBX_PRISM_BUDGET_MB`: the census, the rung proofs) is the card the plan was sized
    # for, and the allocator holds to it as to a card of that size: a malloc that would carry the bytes it holds from
    # the driver (live and pooled) past the rung less the runtime's own context is refused as such a card refuses it,
    # and takes the same path (the pool flushed, the malloc retried, then a named OOM). device -> bytes, None: no rung.
    # Measured 2026-10-28: mochi-1-preview triton at the 8 GB rung on a 32 GB card held 10.8 GB — 5.5-6.2 GB live and
    # 3.0-3.7 GB of pooled free blocks no refused malloc ever asked back.
    _rung_limit_bytes: Dict[int, Optional[int]] = {}
    _pool_slack_frac: Dict[int, float] = {}

    # Deferred-queue reclaimers (P-TRITON-LIVE-SET). The triton hot loops
    # register their deferred dead-tensor queues here. Since 2026-10-11 the
    # allocator's OWN malloc never calls them: Prism prices the queue
    # (`PrismSolver._deferred_drain_mb`), so a refused malloc is raised, not
    # rescued. The one reader left is `wrappers._autotune_headroom_guard`,
    # for the single device allocation the engine does not make and the plan
    # does not price — the Triton autotuner's benchmark buffer. A hook takes
    # (device_idx, nbytes requested) and returns the bytes it freed.
    _oom_reclaim_hooks: list = []

    @classmethod
    def register_oom_reclaim_hook(cls, fn) -> None:
        if fn not in cls._oom_reclaim_hooks:
            cls._oom_reclaim_hooks.append(fn)

    @classmethod
    def unregister_oom_reclaim_hook(cls, fn) -> None:
        try:
            cls._oom_reclaim_hooks.remove(fn)
        except ValueError:
            pass

    @staticmethod
    def set_device(device_id: int):
        """Set current GPU device for allocations (runtime API only).

        Uses only cudaSetDevice/hipSetDevice — does NOT touch the driver
        context. The driver context is managed solely by ensure_triton_device
        which is called just before Triton kernel launches.
        """
        ret = _rt.device_memory().set_device(device_id)
        if ret != 0:
            # An invalid ordinal (a device the process cannot see) must be
            # loud here, not reported by the next unrelated launch check in
            # the process as its own (2026-09-05: torch's, in the compiled
            # branch — "invalid device ordinal" on an fp16 op).
            DeviceAllocator.clear_last_error()
            raise RuntimeError(f"set_device({device_id}) refused by the runtime (error {ret}) — "
                               f"the process sees {DeviceAllocator.device_count()} device(s)")
        # C2 cache maintenance: a runtime-only switch to a device other
        # than the cached one breaks the "runtime AND Triton both point
        # at the cached idx" invariant — invalidate so the next
        # ensure_triton_device performs the full sync. Switching to the
        # cached device itself keeps the pair coherent (Triton's driver
        # device is untouched here).
        if device_id != _rt._cached_device_idx():
            _rt._DEVICE_CACHE.idx = None

    @staticmethod
    def get_device() -> int:
        """Get current GPU device."""
        return _rt.device_memory().get_device()[1]

    @staticmethod
    def peak_device_bytes():
        """The most one device held at once since the process started, as this allocator counted it from the driver
        (live driver bytes, the pool's cache included); None where the backend keeps no such count."""
        return max(DeviceAllocator._cuda_peak_bytes.values()) if DeviceAllocator._cuda_peak_bytes else None

    @staticmethod
    def device_count() -> int:
        """How many GPUs this process can see. 0 if there is no runtime.

        The backend table has declared `device_count` since the seam was
        written, but nothing exposed it: `most_free_device` counted inline and
        every other caller reached for torch. Callers that need to know
        whether multi-GPU behaviour is even testable need it too.
        """
        try:
            return max(0, _rt.device_memory().device_count()[1])
        except Exception:
            return 0

    @staticmethod
    def free_memory_mb(index: int) -> "Optional[float]":
        """The driver's FREE memory on device `index`, in MB, or None when the
        runtime cannot answer.

        The planner's door to the machine as it IS. `DeviceState.free_mb` was
        `capacity - used_mb` with `used_mb` the plan's own accounting — on a
        discrete card nothing ever read what another process holds, so a plan
        that fitted the card whole was accepted while a neighbour held 18 GB
        of it, and died at its first allocation (five of five runs, V100-32GB,
        2026-09-20). Same query as `most_free_device`, for ONE card, the
        previously-current device restored: cudaMemGetInfo / hipMemGetInfo via
        ctypes, no torch (R33).
        """
        try:
            if int(index) < 0 or int(index) >= DeviceAllocator.device_count():
                return None            # cudaSetDevice REPORTS a bad ordinal; it does not raise
            dm = _rt.device_memory()
            prev = DeviceAllocator.get_device()
            dm.set_device(int(index))
            _rc, free_b, _total_b = dm.mem_get_info()
            try:
                dm.set_device(prev)
            except Exception:
                pass
            return free_b / (1024 * 1024)
        except Exception:
            return None

    @staticmethod
    def most_free_device() -> int:
        """Return the GPU index with the most driver-free VRAM.

        Used by the triton weight loader to place a CPU-staged lazy
        component on the emptiest card (the loop GPU may have just
        unloaded). Pure runtime-API — cudaGetDeviceCount + cudaSetDevice +
        cudaMemGetInfo via ctypes; no torch (R33). The previously-current
        device is restored before returning. Falls back to device 0 on any
        query failure or a single-GPU host.
        """
        dm = _rt.device_memory()
        ndev = DeviceAllocator.device_count()
        if ndev <= 1:
            return 0
        prev = DeviceAllocator.get_device()
        best_idx, best_free = 0, -1
        for i in range(ndev):
            try:
                dm.set_device(i)
                _rc, free_b, _total_b = dm.mem_get_info()
                if free_b > best_free:
                    best_free, best_idx = free_b, i
            except Exception:
                continue
        try:
            dm.set_device(prev)
        except Exception:
            pass
        # C2: the probe loop above issued raw cudaSetDevice calls (and
        # the restore may have failed silently) — drop the fast-path
        # cache so the next ensure_triton_device re-syncs fully.
        _rt.invalidate_current_device_cache()
        return best_idx

    @staticmethod
    def visible_device_memory() -> list[tuple[int, int]]:
        """`[(ordinal, total_bytes)]` for every device THIS PROCESS can see.

        The ordinals come back usable as-is in `cuda:<i>`, because the whole
        walk goes through the GPU RUNTIME — so `CUDA_VISIBLE_DEVICES` applies
        to it, exactly as it applies to every allocation made afterwards.

        That is the entire point of the method. `nvidia-smi` answers the same
        question from NVML, which sits OUTSIDE the mask and always reports the
        whole board: a caller that picks an index from `nvidia-smi` and then
        allocates on it is reading one namespace and writing another, and the
        two agree only on an unpinned host. `tests/unit/kernels/
        test_prefill_determinism.py` did exactly that and asked for `cuda:2`
        under `CUDA_VISIBLE_DEVICES=0` (2026-09-17) — a red that had nothing
        to do with its subject, on a card that was simply not there.

        Pure runtime API through ctypes — cudaGetDeviceCount + cudaSetDevice +
        cudaMemGetInfo, no torch (R33). The previously-current device is
        restored and the fast-path cache invalidated, as in
        `most_free_device`, whose inline copy of this loop this replaces.
        Returns `[]` when there is no runtime or no device.
        """
        try:
            dm = _rt.device_memory()
        except Exception:
            return []
        ndev = DeviceAllocator.device_count()
        if ndev <= 0:
            return []
        prev = DeviceAllocator.get_device()
        out: list[tuple[int, int]] = []
        for i in range(ndev):
            try:
                dm.set_device(i)
                _rc, _free_b, total_b = dm.mem_get_info()
                out.append((i, total_b))
            except Exception:
                continue
        try:
            dm.set_device(prev)
        except Exception:
            pass
        _rt.invalidate_current_device_cache()
        return out

    @staticmethod
    def _maybe_init_pool() -> None:
        """One-shot read of the NBX_ALLOC_POOL env var (default on;
        `0` disables the pool). Cached on the class."""
        if not hasattr(DeviceAllocator, '_pool_enabled_init'):
            DeviceAllocator._pool_enabled = os.environ.get("NBX_ALLOC_POOL", "1") != "0"
            DeviceAllocator._pool_enabled_init = True
            if os.environ.get("NBX_ALLOC_STATS", "0") == "1":
                _atexit.register(DeviceAllocator.print_alloc_stats)

    @staticmethod
    def print_alloc_stats() -> None:
        """Driver-level watermark per device + the pool's own numbers
        (NBX_ALLOC_STATS=1, printed at exit). `peak_driver` is the peak of
        bytes held from the driver (pool-parked blocks included);
        `pool_peak` is the most the free-list ever parked; `slack` is the
        smallest-fit over-allocation (allocated − requested) summed over
        every fit hit, and its maximum for one hit."""
        st = DeviceAllocator._pool_stats
        devs = sorted(set(DeviceAllocator._cuda_peak_bytes) | set(DeviceAllocator._pool_cached_peak))
        for d in devs:
            print(f"[NBX_ALLOC_STATS] cuda:{d} peak_driver={DeviceAllocator._cuda_peak_bytes.get(d, 0) / 2**20:.0f}MB "
                  f"pool_peak={DeviceAllocator._pool_cached_peak.get(d, 0) / 2**20:.0f}MB "
                  f"pool_now={DeviceAllocator._pool_cached_bytes.get(d, 0) / 2**20:.0f}MB", flush=True)
        print(f"[NBX_ALLOC_STATS] pool={'on' if DeviceAllocator._pool_enabled else 'off'} "
              f"hits exact={st['exact']} fit={st['fit']} miss={st['miss']} "
              f"slack_total={st['slack_total'] / 2**20:.0f}MB slack_max={st['slack_max'] / 2**20:.0f}MB "
              f"flushes={st['flushes']} flushed={st['flushed_bytes'] / 2**20:.0f}MB "
              f"evictions={st['evictions']} evicted={st['evicted_bytes'] / 2**20:.0f}MB "
              f"cap={ {d: round(c / 2**20) for d, c in DeviceAllocator._pool_cap_bytes.items()} }MB", flush=True)

    @staticmethod
    def _pool_take(dev: int, nbytes: int) -> Optional[int]:
        """The smallest free block on `dev` that holds `nbytes` (rounded up to `_POOL_ALIGN`), split when what is
        left is at least `_POOL_SPLIT_MIN`; None on a miss. The block's size is left in `_pool_alloc_size[ptr]` for
        the caller (it may exceed the request by less than the split threshold)."""
        st = DeviceAllocator._pool_stats
        sizes = DeviceAllocator._pool_sizes.get(dev)
        if not sizes:
            st["miss"] += 1
            return None
        want = -(-int(nbytes) // DeviceAllocator._POOL_ALIGN) * DeviceAllocator._POOL_ALIGN
        k = bisect.bisect_left(sizes, want)
        if k >= len(sizes):
            st["miss"] += 1
            return None
        sz = sizes[k]
        ptr = DeviceAllocator._pool_free[dev][sz][-1]
        DeviceAllocator._free_index_remove(dev, ptr, sz)
        blk = DeviceAllocator._blocks[ptr]
        blk[3] = False
        if sz - want >= DeviceAllocator._POOL_SPLIT_MIN:
            rest = ptr + want
            DeviceAllocator._blocks[rest] = [sz - want, ptr, blk[2], True, dev]
            if blk[2] is not None:
                DeviceAllocator._blocks[blk[2]][1] = rest
            blk[0], blk[2] = want, rest
            DeviceAllocator._cuda_ptr_size[rest] = sz - want
            DeviceAllocator._cuda_ptr_device[rest] = dev
            DeviceAllocator._free_index_add(dev, rest, sz - want)
            st["fit"] += 1
        else:
            st["exact" if sz == int(nbytes) else "fit"] += 1
            if sz > int(nbytes):
                st["slack_total"] += sz - int(nbytes)
                st["slack_max"] = max(st["slack_max"], sz - int(nbytes))
        DeviceAllocator._pool_alloc_size[ptr] = blk[0]
        DeviceAllocator._pool_cached_bytes[dev] = DeviceAllocator._pool_cached_bytes.get(dev, 0) - blk[0]
        return ptr

    @staticmethod
    def _free_index_add(dev: int, ptr: int, size: int) -> None:
        per = DeviceAllocator._pool_free.setdefault(dev, {})
        lst = per.get(size)
        if lst is None:
            per[size] = [ptr]
            bisect.insort(DeviceAllocator._pool_sizes.setdefault(dev, []), size)
        else:
            lst.append(ptr)

    @staticmethod
    def _free_index_remove(dev: int, ptr: int, size: int) -> None:
        per = DeviceAllocator._pool_free.get(dev, {})
        lst = per.get(size)
        lst.remove(ptr)
        if not lst:
            del per[size]
            sizes = DeviceAllocator._pool_sizes[dev]
            sizes.pop(bisect.bisect_left(sizes, size))

    @staticmethod
    def _pool_segment_new(dev: int, ptr: int, nbytes: int) -> None:
        """A driver allocation joins the pool as a one-block segment, in use."""
        DeviceAllocator._blocks[ptr] = [int(nbytes), None, None, False, dev]

    @staticmethod
    def _pool_release_block(dev: int, ptr: int) -> int:
        """`ptr`'s block is free: merged with its free neighbours and indexed; returns the bytes it adds to the
        pool's parked total."""
        blk = DeviceAllocator._blocks[ptr]
        added = blk[0]
        blk[3] = True
        nxt = blk[2]
        if nxt is not None and DeviceAllocator._blocks[nxt][3]:
            nb = DeviceAllocator._blocks.pop(nxt)
            DeviceAllocator._free_index_remove(dev, nxt, nb[0])
            DeviceAllocator._cuda_ptr_size.pop(nxt, None)
            DeviceAllocator._cuda_ptr_device.pop(nxt, None)
            blk[0] += nb[0]
            blk[2] = nb[2]
            if nb[2] is not None:
                DeviceAllocator._blocks[nb[2]][1] = ptr
        prv = blk[1]
        if prv is not None and DeviceAllocator._blocks[prv][3]:
            pb = DeviceAllocator._blocks[prv]
            DeviceAllocator._free_index_remove(dev, prv, pb[0])
            pb[0] += blk[0]
            pb[2] = blk[2]
            if blk[2] is not None:
                DeviceAllocator._blocks[blk[2]][1] = prv
            DeviceAllocator._blocks.pop(ptr)
            DeviceAllocator._cuda_ptr_size.pop(ptr, None)
            DeviceAllocator._cuda_ptr_device.pop(ptr, None)
            ptr, blk = prv, pb
        DeviceAllocator._cuda_ptr_size[ptr] = blk[0]
        DeviceAllocator._cuda_ptr_device[ptr] = dev
        DeviceAllocator._free_index_add(dev, ptr, blk[0])
        return added

    @staticmethod
    def _pool_segment_free(dev: int, ptr: int) -> int:
        """Return a WHOLLY free segment (its one block, free, unchained) to the driver; its bytes."""
        blk = DeviceAllocator._blocks.pop(ptr)
        DeviceAllocator._free_index_remove(dev, ptr, blk[0])
        _rt.device_memory().free(ptr)
        DeviceAllocator._pool_alloc_size.pop(ptr, None)
        DeviceAllocator._cuda_ptr_size.pop(ptr, None)
        DeviceAllocator._range_del(ptr)
        DeviceAllocator._cuda_ptr_device.pop(ptr, None)
        DeviceAllocator._cuda_ptr_epoch.pop(ptr, None)
        DeviceAllocator._cuda_live_bytes[dev] = max(0, DeviceAllocator._cuda_live_bytes.get(dev, 0) - blk[0])
        return blk[0]

    @staticmethod
    def _free_segments(dev: int):
        """The wholly free segments on `dev`, largest first: (ptr, bytes)."""
        out = []
        for size, ptrs in DeviceAllocator._pool_free.get(dev, {}).items():
            for p in ptrs:
                b = DeviceAllocator._blocks.get(p)
                if b is not None and b[1] is None and b[2] is None:
                    out.append((p, size))
        return sorted(out, key=lambda x: -x[1])

    @staticmethod
    def _pool_fit_slack(dev: int) -> float:
        """The most a smallest-fit hit may hand out beyond its request, as a fraction of the request:
        `memory.alloc_pool_fit_slack_fraction` (arch yml), resolved once per device. The surplus of a
        larger block stays held while its tensor lives and no plan prices it: at 2x, HAT-S at 2048 px
        on a 16 GB V100 gave its 2 304 MB conv output the freed 4 096 MB attention mask's block, and the
        next 1 153 MB tile found no room the plan had left (2026-10-22). No profile yet, or no key:
        exact-size hits only (0.0, not cached until the profile exists)."""
        frac = DeviceAllocator._pool_slack_frac.get(dev)
        if frac is not None:
            return frac
        try:
            from nbxtensor.ops.wrappers import get_hardware_profile, _arch_memory_param
            prof = get_hardware_profile()
            if prof is None or not getattr(prof, "devices", None):
                return 0.0
            frac = max(0.0, float(_arch_memory_param("alloc_pool_fit_slack_fraction", 0.0)))
        except Exception:
            return 0.0
        DeviceAllocator._pool_slack_frac[dev] = frac
        return frac

    @staticmethod
    def _pool_parked_cap(dev: int) -> int:
        """Bytes the free-list may keep parked on `dev` (0 = unbounded):
        `memory.alloc_pool_parked_cap_fraction` (arch yml) × the device's
        total memory, resolved once per device. No profile / no key = 0."""
        cap = DeviceAllocator._pool_cap_bytes.get(dev)
        if cap is not None:
            return cap
        # "No hardware profile yet" (Prism has not stashed it) is NOT
        # "explicitly unbounded": return 0 for this call without caching, so
        # the cap resolves once the profile exists. The readers live one
        # layer up (kernels/wrappers.py); the import is function-scoped —
        # no import cycle at load — and the profile global is Prism's.
        try:
            from nbxtensor.ops.wrappers import get_hardware_profile, _arch_memory_param
            prof = get_hardware_profile()
            if prof is None or not getattr(prof, "devices", None):
                return 0
            frac = float(_arch_memory_param("alloc_pool_parked_cap_fraction", 0.0))
            cap = 0
            if frac > 0:
                total = DeviceAllocator.device_total_bytes(dev)
                # A rung imposed on the plan (`NBX_PRISM_BUDGET_MB`, the census and the rung proofs) is the device the
                # plan was sized for: the pool parks a fraction of IT, as it would on a card of that size — measured
                # 2026-10-27, mochi-1-preview triton at the 8 GB rung on a 32 GB card parked its 9 GB text encoder
                # arena (cap 16 GB) and peaked at 22.6 GB where a card of 8 GB would have freed it
                from nbxtensor.host import need as _need
                door = _need("imposed_rung_mb")()      # the application's imposed rung (nbxtensor.host)
                if door is not None and total > 0:
                    total = min(total, int(door) << 20)
                if total > 0:
                    cap = int(total * frac)
        except Exception:
            return 0
        DeviceAllocator._pool_cap_bytes[dev] = cap
        return cap

    @staticmethod
    def _pool_evict(dev: int, want: int) -> int:
        """Return wholly free segments on `dev` to the driver, largest first, until at least `want` bytes are freed
        or none is left. Returns the bytes freed."""
        freed = 0
        st = DeviceAllocator._pool_stats
        for p, size in DeviceAllocator._free_segments(dev):
            if freed >= want:
                break
            freed += DeviceAllocator._pool_segment_free(dev, p)
            st["evictions"] += 1
        DeviceAllocator._pool_cached_bytes[dev] = max(0, DeviceAllocator._pool_cached_bytes.get(dev, 0) - freed)
        st["evicted_bytes"] += freed
        return freed

    @staticmethod
    def _pool_flush(dev: Optional[int] = None) -> int:
        """Return every wholly free segment to the driver (a segment still holding a live block stays). Returns the
        bytes freed. Called on a refused malloc and from empty_cache_pool()."""
        freed = 0
        devs = [dev] if dev is not None else list(DeviceAllocator._pool_free.keys())
        DeviceAllocator._pool_stats["flushes"] += 1
        for d in devs:
            got = 0
            for p, _size in DeviceAllocator._free_segments(d):
                got += DeviceAllocator._pool_segment_free(d, p)
            DeviceAllocator._pool_cached_bytes[d] = max(0, DeviceAllocator._pool_cached_bytes.get(d, 0) - got)
            freed += got
        DeviceAllocator._pool_stats["flushed_bytes"] += freed
        return freed

    @staticmethod
    def empty_cache_pool() -> int:
        """Public API: release all pool-cached blocks back to the driver.
        Returns total bytes freed. Safe to call when pool is disabled
        (no-op). Useful between phases to defragment proactively."""
        DeviceAllocator._maybe_init_pool()
        if not DeviceAllocator._pool_enabled:
            return 0
        return DeviceAllocator._pool_flush()

    @staticmethod
    def new_allocation_epoch() -> int:
        """Declare that every address issued so far belongs to a runtime that
        is being replaced. Returns the new epoch.

        Call it AFTER releasing what can still be released (the parked pool)
        and BEFORE the new runtime exists. From here on, a free of an older
        pointer is a no-op rather than a park or a driver call.

        The engine proper never calls this: a process has one device runtime
        for its whole life. It exists because test processes legitimately
        replace one, and because the alternative — a free-list that outlives
        the runtime it belongs to — surfaced as an order-dependent numeric
        failure in an unrelated kernel, which is the worst way for it to be
        found.
        """
        DeviceAllocator._alloc_epoch += 1
        return DeviceAllocator._alloc_epoch

    @staticmethod
    def _rung_limit(dev: int) -> Optional[int]:
        """The bytes this process may hold from the driver on `dev` under an imposed rung: the rung less what the
        runtime holds outside the allocator's own blocks (its context), read once from the driver; None: no rung."""
        if dev in DeviceAllocator._rung_limit_bytes:
            return DeviceAllocator._rung_limit_bytes[dev]
        door = os.environ.get("NBX_PRISM_BUDGET_MB")
        limit = None
        if door:
            free_b, total_b = DeviceAllocator.mem_get_info(dev)
            context = max(0, int(total_b) - int(free_b) - DeviceAllocator._cuda_live_bytes.get(dev, 0))
            limit = max(0, (int(door) << 20) - context)
        DeviceAllocator._rung_limit_bytes[dev] = limit
        return limit

    @staticmethod
    def _past_budget(dm, dev: int, nbytes: int):
        """The backend's allocation-refused status when `nbytes` more would take this process past the budget its
        device grants (`DeviceMemory.budget`), for a backend whose driver does not refuse there itself
        (`supports("budget_refusal")`: Metal, whose buffers draw on the memory the whole machine shares; its host
        buffers draw on the same budget); None otherwise. The ONE account decides — the device keeps none."""
        if not dm.supports("budget_refusal"):
            return None
        rc, cap = dm.budget()
        if rc != 0:
            return rc
        held = DeviceAllocator._cuda_live_bytes.get(dev, 0) + DeviceAllocator._host_pinned_live_bytes
        return dm.ALLOC_REFUSED if held + int(nbytes) > int(cap) else None

    @staticmethod
    def _driver_malloc(ptr_obj, nbytes: int, dev: int) -> int:
        """The runtime's malloc, or the refusal a card of the imposed rung gives (cudaErrorMemoryAllocation, 2)."""
        limit = DeviceAllocator._rung_limit(dev)
        if limit is not None and DeviceAllocator._cuda_live_bytes.get(dev, 0) + int(nbytes) > limit:
            return 2
        dm = _rt.device_memory()
        refused = DeviceAllocator._past_budget(dm, dev, nbytes)
        if refused is not None:
            ptr_obj.value = None
            return refused
        rc, ptr = dm.malloc(nbytes)
        ptr_obj.value = ptr or None
        return rc

    @staticmethod
    def install_interceptor(interceptor: AllocatorInterceptor) -> None:
        """Put `interceptor` in front of the five device operations. Installing the one in force again is a no-op; a
        second, different one is refused by name (each would believe it sees every allocation)."""
        cur = DeviceAllocator._interceptor
        if cur is not None and cur is not interceptor:
            raise RuntimeError(
                f"an allocator interceptor is already installed ({type(cur).__module__}.{type(cur).__qualname__}); "
                f"{type(interceptor).__module__}.{type(interceptor).__qualname__} cannot stand beside it")
        DeviceAllocator._interceptor = interceptor

    @staticmethod
    def remove_interceptor(interceptor: AllocatorInterceptor) -> None:
        """Take `interceptor` away; refused by name when it is not the one in force."""
        if DeviceAllocator._interceptor is not interceptor:
            raise RuntimeError(f"{type(interceptor).__qualname__} is not the allocator interceptor in force")
        DeviceAllocator._interceptor = None

    @staticmethod
    def malloc_cuda(nbytes: int, dev_idx: Optional[int] = None, pooled: bool = True) -> int:
        """Allocate GPU memory: through the interceptor in force, else directly (`malloc_direct`)."""
        i = DeviceAllocator._interceptor
        if i is None:
            return DeviceAllocator.malloc_direct(nbytes, dev_idx, pooled)
        return i.malloc(DeviceAllocator.malloc_direct, nbytes, dev_idx, pooled)

    @staticmethod
    def malloc_direct(nbytes: int, dev_idx: Optional[int] = None, pooled: bool = True) -> int:
        """Allocate GPU memory.

        Default path: synchronous cudaMalloc. The stream-ordered allocator
        (cudaMallocAsync) was prototyped but disabled because its per-
        device memory pools don't grant cross-device peer access by
        default — freeing a cuda:N pointer from a Python-GC site while
        cuda:M is current produced illegal memory accesses on the multi-
        GPU pipeline_parallel path (Qwen3-30B --triton). The synchronous
        path combined with TritonSequence's _deferred batch-release is
        stable across single-GPU, zero3, and pipeline_parallel.

        Phase 2 path (NBX_ALLOC_POOL=1): satisfies requests from a per-
        device free-list pool first; on cudaMalloc OOM, flushes the pool
        back to the driver and retries (analog to torch CachingAllocator
        release_cached_blocks → retry). Mitigates fragmentation when
        many small/medium allocs precede a large alloc, which was the
        Sana 4Kpx VAE conv::62 8 GiB OOM pattern.

        dev_idx (C3): the CURRENT device index, when the caller already
        knows it — skips the cudaGetDevice round-trip used purely for
        pool/accounting bookkeeping. The caller MUST guarantee the
        runtime current device equals dev_idx (cudaMalloc allocates on
        the current device). None (default) keeps the internal query.

        pooled=False: a block that never enters the pool — taken from the driver (a refusal still flushes the pool and
        retries) and handed straight back to it when freed. A component's weight arena (`ComponentArena`): large and
        held for a whole component or segment; pooled, the small tensors that outlive it split it once freed and pin
        it whole (mochi-1-preview triton at the 8 GB rung, 2026-10-28: 3 553 MB of freed blocks held by ~19 MB of
        live tensors refused the transformer's 5.2 GB segment).
        """
        if nbytes == 0:
            return 0
        DeviceAllocator._maybe_init_pool()
        dev = dev_idx if dev_idx is not None else DeviceAllocator.get_device()

        # Pool fast path
        if DeviceAllocator._pool_enabled and pooled:
            ptr = DeviceAllocator._pool_take(dev, nbytes)
            if ptr is not None:
                # alloc size: exact-size hit reuses nbytes; smallest-fit
                # populated _pool_alloc_size with the bigger size.
                actual = DeviceAllocator._pool_alloc_size.pop(ptr, nbytes)
                DeviceAllocator._cuda_ptr_size[ptr] = actual
                DeviceAllocator._cuda_ptr_device[ptr] = dev      # its segment's range stays registered
                DeviceAllocator._cuda_ptr_epoch[ptr] = DeviceAllocator._alloc_epoch
                if _MALLOC_TRACE_FILE is not None:
                    _record_malloc_site(ptr, actual, dev)
                # No live-byte change: pool blocks stay counted as live
                # for the lifetime of the segment (pool == still allocated
                # from the driver's POV). Peak unchanged.
                return ptr

        # The device never GROWS while the pool parks freed blocks: they go back to the driver first, so the device
        # holds what is alive, not what is alive plus what was once freed and split (the S4a pool cuts its segments
        # for later small blocks, and a refused malloc's flush can then return none of them whole). Sana_1600M_4Kpx
        # triton on a V100-16GB, lot 2026-10-28: 4.5 GB of parked blocks in 12 split segments beside 9.4 GB alive
        # refused the VAE's 2.96 GB convolution; the Mac measured the same rule on HAT-L x4 (10 688 -> 8 913 MB, bit-
        # equal, 77258825).
        if DeviceAllocator._pool_enabled and pooled:
            DeviceAllocator._pool_flush(dev)
        ptr_obj = ctypes.c_void_p()
        dm = _rt.device_memory()
        ret = DeviceAllocator._driver_malloc(ptr_obj, nbytes, dev)

        # No reclaim-and-retry of the hot loops' deferred dead-tensor queues here
        # (ZERO FALLBACK, 2026-10-11). That rescue drained the queue on a refused
        # malloc and retried — `[NBX_OOM_RECLAIM] freed ...` 1 279 times in one
        # 16 GB Allegro run — so a plan that did not leave room for the queue ran
        # anyway. Prism now PRICES the queue (`PrismSolver._deferred_drain_mb`, the
        # vendor profile's `memory.deferred_drain_cliff_bytes`), and a malloc refused
        # with the queue inside its price is a plan defect, raised below and named
        # by the hot loop with the op, the queue it held and the price it had.

        # Phase 2 OOM-retry: if cudaMalloc fails and the pool holds cached
        # blocks, flush them back to the driver and retry. Defragments the
        # driver's internal heap before declaring failure.
        if ret != 0 and DeviceAllocator._pool_enabled:
            DeviceAllocator.clear_last_error()   # a refused call must not poison the next one
            freed = DeviceAllocator._pool_flush(dev)
            if freed > 0:
                ret = DeviceAllocator._driver_malloc(ptr_obj, nbytes, dev)

        # The engine's other cached memory, released the same way: a recorded replay plan's slab between two runs of
        # its sequence (`triton/replay.release_idle_slabs`), freed through the pool, flushed, retried once.
        if ret != 0 and DeviceAllocator._idle_reclaimers:
            DeviceAllocator.clear_last_error()   # a refused call must not poison the next one
            got = sum(int(r(dev) or 0) for r in list(DeviceAllocator._idle_reclaimers))
            if got > 0:
                if DeviceAllocator._pool_enabled:
                    DeviceAllocator._pool_flush(dev)
                ret, _p = dm.malloc(nbytes)          # the runtime's malloc itself, as before (no rung check here)
                ptr_obj.value = _p or None

        # Pre-allocation OOM-recovery (independent of pool): trigger a
        # cycle-collecting gc.collect() then retry. Sana 4Kpx triton
        # showed ~25 GB of NBXTensor instances pinned alive at conv::62
        # OOM by Python frame-args lists from recursive conv2d_wrapper
        # band loops; gc.collect() picks them up if they form a cycle.
        # Gated by NBX_GC_ON_OOM=1 (opt-in until validated cross-model).
        if ret != 0 and os.environ.get("NBX_GC_ON_OOM", "0") == "1":
            DeviceAllocator.clear_last_error()   # a refused call must not poison the next one
            import gc as _gc_oom
            _gc_oom.collect()
            ret = DeviceAllocator._driver_malloc(ptr_obj, nbytes, dev)

        if ret != 0:
            DeviceAllocator.clear_last_error()   # a refused call must not poison the next one
            # Diagnostic: how much live + how big the pool was at OOM time.
            live_now = DeviceAllocator._cuda_live_bytes.get(dev, 0)
            pool_total = 0
            pool_count = 0
            per_dev = DeviceAllocator._pool_free.get(dev, {})
            for sz, ptrs in per_dev.items():
                pool_total += sz * len(ptrs)
                pool_count += len(ptrs)
            try:
                _rc, driver_free, driver_total = dm.mem_get_info()   # a runtime without the symbol raises: 0 below
            except Exception:
                driver_free = driver_total = 0
            # The message is unchanged to the byte: the queue runner's error
            # extractor and every log-reading tool key on this text.
            raise DeviceOOMError(
                f"GPU malloc failed (error {ret}) for {nbytes} bytes "
                f"[device cuda:{dev} live_tracked={live_now/1024/1024:.0f}MB "
                f"pool_cached={pool_total/1024/1024:.0f}MB ({pool_count} blocks) "
                f"driver_free={driver_free/1024/1024:.0f}MB / "
                f"driver_total={driver_total/1024/1024:.0f}MB]"
                + (f" under the imposed rung: {DeviceAllocator._rung_limit_bytes[dev]/1024/1024:.0f}MB this process may "
                   f"hold (NBX_PRISM_BUDGET_MB less the runtime's context)"
                   if DeviceAllocator._rung_limit_bytes.get(dev) is not None else ""),
                requested=nbytes, device_idx=dev, live=live_now,
                pool_cached=pool_total, pool_blocks=pool_count,
                driver_free=driver_free if driver_total else None,
                driver_total=driver_total or None)
        p = ptr_obj.value or 0
        DeviceAllocator._cuda_ptr_size[p] = nbytes
        DeviceAllocator._range_add(p, nbytes)
        DeviceAllocator._cuda_ptr_device[p] = dev
        if DeviceAllocator._pool_enabled and pooled:
            DeviceAllocator._pool_segment_new(dev, p, nbytes)
        DeviceAllocator._cuda_ptr_epoch[p] = DeviceAllocator._alloc_epoch
        live = DeviceAllocator._cuda_live_bytes.get(dev, 0) + nbytes
        DeviceAllocator._cuda_live_bytes[dev] = live
        peak = DeviceAllocator._cuda_peak_bytes.get(dev, 0)
        if live > peak:
            DeviceAllocator._cuda_peak_bytes[dev] = live
        if _MALLOC_TRACE_FILE is not None:
            _record_malloc_site(p, nbytes, dev)
        return p

    @staticmethod
    def free_cuda(ptr: int):
        """Free GPU memory: through the interceptor in force, else directly (`free_direct`)."""
        i = DeviceAllocator._interceptor
        if i is None:
            return DeviceAllocator.free_direct(ptr)
        return i.free(DeviceAllocator.free_direct, ptr)

    @staticmethod
    def free_direct(ptr: int):
        """Free GPU memory.

        Default path: cudaFree. Blocks the calling thread until the
        device is idle for the relevant stream, so TritonSequence
        batches freeable tensors into _deferred and releases them
        after a sync at the end of each forward pass.

        Phase 2 path (NBX_ALLOC_POOL=1): the pointer is returned to the
        per-device free-list pool instead of cudaFree, keeping the
        driver's internal heap intact (avoids fragmenting it with the
        churn of small/medium tensors typical of a diffusion forward
        pass). Live byte counters stay unchanged because the block is
        STILL allocated from the driver's POV — it just sits in the
        pool waiting to satisfy the next compatible malloc. cudaFree
        eventually fires from _pool_flush() on OOM-retry or from
        empty_cache_pool().
        """
        if not ptr:
            return
        DeviceAllocator._maybe_init_pool()

        # A pointer issued by a runtime that no longer exists.
        #
        # Its memory went when that runtime did, so there is nothing to give
        # back — and both of the things this function otherwise does would be
        # wrong: parking it hands a dead address to the next allocation, and
        # freeing it asks the CURRENT runtime to release something it never
        # issued, which it correctly refuses and loudly. Drop the bookkeeping
        # and say nothing. Only a process that replaces its runtime ever
        # reaches this; with one runtime the epoch never moves.
        _epoch = DeviceAllocator._cuda_ptr_epoch.pop(ptr, DeviceAllocator._alloc_epoch)
        if _epoch != DeviceAllocator._alloc_epoch:
            _dev = DeviceAllocator._cuda_ptr_device.pop(ptr, None)
            _sz = DeviceAllocator._cuda_ptr_size.pop(ptr, None)
            DeviceAllocator._pool_alloc_size.pop(ptr, None)
            if _dev is not None and _sz:
                _live = DeviceAllocator._cuda_live_bytes.get(_dev, 0) - _sz
                DeviceAllocator._cuda_live_bytes[_dev] = max(0, _live)
            return

        # The pool: the block goes back to its segment, merged with its free neighbours; a segment wholly free over
        # the parked cap returns to the driver. A block of a segment is never handed to the driver alone.
        if DeviceAllocator._pool_enabled and ptr in DeviceAllocator._blocks:
            if DeviceAllocator._blocks[ptr][3]:
                print(f"[NBX-ALLOC-ERROR] free of a block already free: {ptr:#x}", flush=True)
                return
            dev = DeviceAllocator._blocks[ptr][4]
            nbytes = DeviceAllocator._blocks[ptr][0]
            added = DeviceAllocator._pool_release_block(dev, ptr)
            cached = DeviceAllocator._pool_cached_bytes.get(dev, 0) + added
            DeviceAllocator._pool_cached_bytes[dev] = cached
            if cached > DeviceAllocator._pool_cached_peak.get(dev, 0):
                DeviceAllocator._pool_cached_peak[dev] = cached
            cap = DeviceAllocator._pool_parked_cap(dev)
            if cap and cached > cap:
                DeviceAllocator._pool_evict(dev, cached - cap)
            if _MALLOC_TRACE_FILE is not None:
                _record_free_site(ptr, nbytes)
            return

        alloc_dev = DeviceAllocator._cuda_ptr_device.pop(ptr, None)
        _rc = _rt.device_memory().free(ptr)
        if _rc not in (None, 0):
            # Finalizer context — raising here cascades through GC; a
            # LOUD print preserves the sticky-error surfacing site.
            print(f"[NBX-CUDA-ERROR] cudaFree rc={_rc} ptr={ptr:#x} "
                  f"dev={alloc_dev} — sticky async error surfaced at "
                  f"this free; the fault is at or before it.", flush=True)
        nbytes = DeviceAllocator._cuda_ptr_size.pop(ptr, None)
        DeviceAllocator._range_del(ptr)
        if _MALLOC_TRACE_FILE is not None and nbytes is not None:
            _record_free_site(ptr, nbytes)
        if nbytes is not None:
            if alloc_dev is not None:
                live = DeviceAllocator._cuda_live_bytes.get(alloc_dev, 0) - nbytes
                DeviceAllocator._cuda_live_bytes[alloc_dev] = live
            else:
                dev = DeviceAllocator.get_device()
                live = DeviceAllocator._cuda_live_bytes.get(dev, 0) - nbytes
                if live < 0:
                    for d in DeviceAllocator._cuda_live_bytes:
                        if DeviceAllocator._cuda_live_bytes[d] >= nbytes:
                            DeviceAllocator._cuda_live_bytes[d] -= nbytes
                            break
                else:
                    DeviceAllocator._cuda_live_bytes[dev] = live

    @staticmethod
    def bind_mapped(ptr: int, nbytes: int) -> None:
        """Make `nbytes` of a file mapping at host address `ptr` readable by kernels IN PLACE — on a unified
        device only (one pool: the GPU reads the mapping's pages, nothing is copied). A discrete device has no
        such address space: refused by name. Registered in the range table, so `holds()` knows the range."""
        if _rt._detect_gpu_backend() != "metal":
            raise RuntimeError(f"ZERO FALLBACK: binding a file mapping in place needs unified memory; backend "
                               f"{_rt._detect_gpu_backend()!r} copies host experts into device slots")
        start = _rt.device_memory().map_host(int(ptr), int(nbytes))
        page = int(os.sysconf("SC_PAGESIZE"))
        end = -(-(int(ptr) + int(nbytes)) // page) * page
        DeviceAllocator._range_add(start, max(DeviceAllocator._range_size.get(start, 0), end - start))

    @staticmethod
    def unbind_mapped(ptr: int) -> None:
        """Undo `bind_mapped` for the range starting at `ptr`'s page."""
        page = int(os.sysconf("SC_PAGESIZE"))
        start = int(ptr) // page * page
        _rt.device_memory().unmap_host(start)
        DeviceAllocator._range_del(start)

    @staticmethod
    def malloc_host_pinned(nbytes: int) -> int:
        """Allocate page-locked host memory. Enables non_blocking H2D DMA.

        Uses cudaMallocHost (or hipHostMalloc on ROCm). The returned
        pointer is a HOST address — numpy can wrap it via
        np.frombuffer(ctypes.string_at(ptr, nbytes), ...) or ctypes
        casting, and cudaMemcpy kind=1 (H2D) uses it as src.
        """
        if nbytes == 0:
            return 0
        dm = _rt.device_memory()
        if not dm.supports("host_alloc"):
            raise RuntimeError(
                f"Pinned host allocation unsupported on backend "
                f"{_rt._detect_gpu_backend()!r}")
        ret = DeviceAllocator._past_budget(dm, DeviceAllocator.get_device(), nbytes)
        if ret is None:
            ret, p = dm.host_alloc(nbytes)
        if ret != 0:
            DeviceAllocator.clear_last_error()   # a refused call must not poison the next one
            raise RuntimeError(
                f"Host pinned malloc failed (error {ret}) for {nbytes} bytes")
        DeviceAllocator._host_pinned_ptr_size[p] = nbytes
        DeviceAllocator._range_add(p, nbytes)      # device-mapped (UVA): a kernel may read it, so holds() knows it
        DeviceAllocator._host_pinned_live_bytes += nbytes
        if DeviceAllocator._host_pinned_live_bytes > DeviceAllocator._host_pinned_peak_bytes:
            DeviceAllocator._host_pinned_peak_bytes = DeviceAllocator._host_pinned_live_bytes
        return p

    @staticmethod
    def free_host_pinned(ptr: int):
        if ptr:
            dm = _rt.device_memory()
            if not dm.supports("host_free"):
                return
            dm.host_free(ptr)
            nbytes = DeviceAllocator._host_pinned_ptr_size.pop(ptr, None)
            DeviceAllocator._range_del(ptr)
            if nbytes is not None:
                DeviceAllocator._host_pinned_live_bytes -= nbytes

    # ------------------------------------------------------------------
    # Accounting API — use instead of cudaMemGetInfo for per-NeuroBrix
    # observability (cudaMemGetInfo returns device-wide numbers
    # including other processes, which is useless for debugging a
    # specific zero3 leak).
    # ------------------------------------------------------------------

    @staticmethod
    def memory_allocated(device_idx: Optional[int] = None) -> int:
        """Live bytes allocated via malloc_cuda on the given device.
        With device_idx=None, returns the total across all devices."""
        if device_idx is None:
            return sum(DeviceAllocator._cuda_live_bytes.values())
        return DeviceAllocator._cuda_live_bytes.get(device_idx, 0)

    @staticmethod
    def peak_memory_allocated(device_idx: Optional[int] = None) -> int:
        """Highest live-bytes watermark observed for the device.
        With device_idx=None, returns the total peak across all devices."""
        if device_idx is None:
            return sum(DeviceAllocator._cuda_peak_bytes.values())
        return DeviceAllocator._cuda_peak_bytes.get(device_idx, 0)

    @staticmethod
    def device_free_bytes(device_idx: Optional[int] = None) -> int:
        """Driver-free bytes on `device_idx` (None = current device), via
        cudaMemGetInfo/hipMemGetInfo through ctypes (R33: no torch).

        Returns -1 when the query is unavailable (CPU-only host, runtime
        load failure) so callers can distinguish "unknown" from "0 bytes
        free". Read-only feasibility probe for wrapper-internal memory
        streaming gates (conv3d temporal chunk-streaming) — deliberately
        NOT an accounting API (device-wide, includes other processes) and
        NOT a per-launch kernel-config source (hardware params stay in the
        vendor YAMLs)."""
        prev_dev = None
        try:
            dm = _rt.device_memory()
            if (device_idx is not None
                    and DeviceAllocator.get_device() != device_idx):
                # Probe on another device: switch, query, RESTORE (the
                # allocator's own doctrine — a GC-site free of a cuda:N
                # pointer must never leave cuda:N current; the C3 fast path
                # contracts that the runtime current device is the caller's).
                prev_dev = DeviceAllocator.get_device()
                DeviceAllocator.set_device(device_idx)
            _rc, free_b, _total_b = dm.mem_get_info()
            free = int(free_b)
            # Pool-aware (2026-09-02): bytes parked in the free-list are
            # reclaimable on demand (the malloc OOM path flushes them, the
            # autotune guard flushes them), so every feasibility probe —
            # the SDPA chunk decision, the drain pressure gate, the conv3d
            # chunk-stream gate — reads the same headroom it read with the
            # pool off. Without this the probes under-read free memory by
            # the parked bytes and shrank their chunks on edge rows.
            if DeviceAllocator._pool_enabled:
                d = device_idx if device_idx is not None else DeviceAllocator.get_device()
                free += DeviceAllocator._pool_cached_bytes.get(d, 0)
            return free
        except Exception:
            return -1
        finally:
            if prev_dev is not None:
                try:
                    DeviceAllocator.set_device(prev_dev)
                except Exception:
                    pass

    @staticmethod
    def device_total_bytes(device_idx: Optional[int] = None) -> int:
        """Total device memory on `device_idx` (None = current device) via
        cudaMemGetInfo/hipMemGetInfo through ctypes; -1 when unavailable.
        Same contract and caveats as `device_free_bytes`."""
        prev_dev = None
        try:
            dm = _rt.device_memory()
            if (device_idx is not None
                    and DeviceAllocator.get_device() != device_idx):
                # Probe on another device: switch, query, RESTORE (the
                # allocator's own doctrine — a GC-site free of a cuda:N
                # pointer must never leave cuda:N current; the C3 fast path
                # contracts that the runtime current device is the caller's).
                prev_dev = DeviceAllocator.get_device()
                DeviceAllocator.set_device(device_idx)
            _rc, _free_b, total_b = dm.mem_get_info()
            return int(total_b)
        except Exception:
            return -1
        finally:
            if prev_dev is not None:
                try:
                    DeviceAllocator.set_device(prev_dev)
                except Exception:
                    pass

    @staticmethod
    def reset_peak_memory(device_idx: Optional[int] = None) -> None:
        """Reset the peak watermark to the current live value.
        With device_idx=None, resets all devices."""
        if device_idx is None:
            for d in list(DeviceAllocator._cuda_live_bytes.keys()):
                DeviceAllocator._cuda_peak_bytes[d] = (
                    DeviceAllocator._cuda_live_bytes[d])
        else:
            DeviceAllocator._cuda_peak_bytes[device_idx] = (
                DeviceAllocator._cuda_live_bytes.get(device_idx, 0))

    @staticmethod
    def host_pinned_allocated() -> int:
        """Live bytes allocated via malloc_host_pinned."""
        return DeviceAllocator._host_pinned_live_bytes

    @staticmethod
    def host_pinned_peak() -> int:
        """Peak pinned host bytes observed."""
        return DeviceAllocator._host_pinned_peak_bytes

    @staticmethod
    def memset_cuda(ptr: int, value: int, nbytes: int):
        """Set device bytes: through the interceptor in force, else directly (`memset_direct`)."""
        i = DeviceAllocator._interceptor
        if i is None:
            return DeviceAllocator.memset_direct(ptr, value, nbytes)
        return i.memset(DeviceAllocator.memset_direct, ptr, value, nbytes)

    @staticmethod
    def memset_direct(ptr: int, value: int, nbytes: int):
        if nbytes > 0 and ptr:
            _rt.device_memory().memset(ptr, value, nbytes)

    @staticmethod
    def memcpy(dst: int, src: int, nbytes: int, kind: int = 3):
        """Copy memory (kind: 0=H2H, 1=H2D, 2=D2H, 3=D2D): through the interceptor in force, else directly
        (`memcpy_direct`)."""
        i = DeviceAllocator._interceptor
        if i is None:
            return DeviceAllocator.memcpy_direct(dst, src, nbytes, kind)
        return i.memcpy(DeviceAllocator.memcpy_direct, dst, src, nbytes, kind)

    @staticmethod
    def memcpy_direct(dst: int, src: int, nbytes: int, kind: int = 3):
        """Copy memory. kind: 0=H2H, 1=H2D, 2=D2H, 3=D2D.

        NOTE (NVIDIA API-sync-behavior spec): for D2D the "synchronous"
        cudaMemcpy performs NO host-side synchronization — the copy is
        enqueued on the calling device's legacy stream and the host
        returns immediately. Lifetime of src/dst must be guaranteed by
        stream ordering, never by host program order.
        """
        if nbytes > 0:
            _rc = _rt.device_memory().memcpy(dst, src, nbytes, kind)
            if _rc not in (None, 0):
                raise RuntimeError(
                    f"cudaMemcpy failed rc={_rc} (kind={kind}, "
                    f"nbytes={nbytes}) — CUDA errors are STICKY: this is "
                    f"the first CHECKED call after the poisoning site, "
                    f"not necessarily the fault itself.")

    @staticmethod
    def sync_device():
        """Synchronize the current device: through the interceptor in force, else directly (`sync_direct`)."""
        i = DeviceAllocator._interceptor
        if i is None:
            return DeviceAllocator.sync_direct()
        return i.sync(DeviceAllocator.sync_direct)

    @staticmethod
    def sync_direct():
        """Synchronize current GPU device. Waits for all pending GPU ops.

        RAISES on a nonzero return: cudaDeviceSynchronize returns any
        prior async error (sticky semantics) — discarding it hid the
        true poisoning site of the multi-GPU error-700 class behind the
        next checked malloc (R16 research, 2026-07-24)."""
        rt = _rt._gpu_runtime()
        backend = _rt._active_backend()
        sync_name = backend.get("sync")
        if sync_name:
            _rc = getattr(rt, sync_name)()
            if _rc not in (None, 0):
                raise RuntimeError(
                    f"deviceSynchronize failed rc={_rc} on current device "
                    f"{DeviceAllocator.get_device()} — sticky async error "
                    f"surfaced at this sync; the fault is at or before it.")
        # The portable half of the same contract: a backend that does not
        # honour `tl.device_assert` reports through the fault channel, and
        # this is where it is observed. The sync above has retired the work
        # that would have written it.
        _faults.check_device_faults()

    # ------------------------------------------------------------------
    # Async stream + event primitives (zero torch).
    # Zero3 pipelining opens a dedicated transfer stream so that the H2D
    # copy of block N+1 overlaps with the compute on block N which runs
    # on the default stream. Events synchronize the two streams so we
    # never read a weight that's still being copied.
    #
    # Handles are opaque ctypes.c_void_p values surfaced as Python ints
    # (0 is the default/NULL stream). Keep them in Python state; do not
    # re-wrap them on each use — ctypes creates a fresh opaque object
    # each call.
    # ------------------------------------------------------------------

    @staticmethod
    def create_stream() -> int:
        """Create a new GPU stream (`cudaStreamCreate` / `hipStreamCreate`:
        a BLOCKING stream, i.e. one that implicitly orders against the
        legacy default stream). Returns an opaque handle.

        This ordering is load-bearing: the deferred-free drain skips its
        host sync under the allocator pool on the premise that every
        NeuroBrix triton stream orders against the legacy stream (kernel
        launches and D2D copies on the legacy stream; the zero3 transfer
        stream created here). The only non-blocking stream in the engine
        is the replay-graph capture stream, bracketed by context syncs."""
        rt = _rt._gpu_runtime()
        backend = _rt._active_backend()
        fn_name = backend.get("stream_create")
        if fn_name is None:
            raise RuntimeError(
                f"Stream create unsupported on backend {_rt._detect_gpu_backend()!r}")
        h = ctypes.c_void_p()
        ret = getattr(rt, fn_name)(ctypes.byref(h))
        if ret != 0:
            DeviceAllocator.clear_last_error()   # a refused call must not poison the next one
            raise RuntimeError(f"stream create failed (error {ret})")
        return h.value or 0

    @staticmethod
    def clear_last_error() -> int:
        """Read-and-reset the runtime's sticky "last error". A refused call
        (an ordering event asked for an interval, a peer pair that cannot be
        enabled) leaves its code there, and the next unrelated launch check
        in the process — torch's, in the compiled branch — reports it as its
        own (found 2026-09-05: `invalid device ordinal` on an fp16 op after
        the stopwatch refusal test). Every expected refusal clears it."""
        rt = _rt._gpu_runtime()
        fn_name = _rt._active_backend().get("get_last_error")
        return int(getattr(rt, fn_name)()) if fn_name else 0

    @staticmethod
    def _rt_call(key: str, *args):
        rt = _rt._gpu_runtime()
        fn_name = _rt._active_backend().get(key)
        if fn_name is None:
            raise RuntimeError(f"{key} unsupported on backend {_rt._detect_gpu_backend()!r}")
        ret = getattr(rt, fn_name)(*args)
        if ret != 0:
            DeviceAllocator.clear_last_error()
            raise RuntimeError(f"{fn_name} failed (error {ret})")

    @staticmethod
    def destroy_stream(stream: int):
        """Destroy a stream created by create_stream. No-op for default (0)."""
        if not stream:
            return
        rt = _rt._gpu_runtime()
        backend = _rt._active_backend()
        fn_name = backend.get("stream_destroy")
        if fn_name is None:
            return
        getattr(rt, fn_name)(ctypes.c_void_p(stream))

    @staticmethod
    def holds(ptr: int) -> bool:
        """True when `ptr` lies inside a live device allocation this allocator
        handed out (a block's base, or a view inside it). The launcher asks
        before dispatching a kernel on an address: a foreign pointer is
        refused, never launched (launcher contract, Metal workstream 2026-09-05).
        O(log n): the live ranges are re-sorted only when a registration
        changed since the last call."""
        ptr = int(ptr)
        size = DeviceAllocator._range_size.get(ptr)
        if size is not None:
            return True
        bases = DeviceAllocator._range_bases
        i = bisect.bisect_right(bases, ptr) - 1
        return i >= 0 and bases[i] <= ptr < bases[i] + DeviceAllocator._range_size[bases[i]]

    @staticmethod
    def containing(ptr: int):
        """(base, size) of the live range `ptr` lies in, the NEAREST one: the greatest base at or below `ptr` whose range
        holds it. In-place expert bindings overlap where two projections share a page; the first range in insertion
        order resolved a projection to its neighbour's shorter range and a launch read past it (DSV2 triton printed
        garbage, 2026-10-20). The nearest base is the projection's own binding, which runs to its end. None when no
        range holds `ptr`. O(log n) plus the ranges walked down past."""
        ptr = int(ptr)
        bases = DeviceAllocator._range_bases
        sizes = DeviceAllocator._range_size
        i = bisect.bisect_right(bases, ptr) - 1
        while i >= 0:
            b = bases[i]
            if ptr < b + sizes[b]:
                return b, sizes[b]
            i -= 1
        return None

    @staticmethod
    def _range_add(base: int, nbytes: int) -> None:
        """Register a live range for `holds()` — O(log n) insert, kept sorted
        so a launch pays one bisect and never a re-sort."""
        if base not in DeviceAllocator._range_size:
            bisect.insort(DeviceAllocator._range_bases, base)
        DeviceAllocator._range_size[base] = int(nbytes)

    @staticmethod
    def _range_del(base: int) -> None:
        if DeviceAllocator._range_size.pop(base, None) is not None:
            bases = DeviceAllocator._range_bases
            i = bisect.bisect_left(bases, base)
            if i < len(bases) and bases[i] == base:
                bases.pop(i)

    @staticmethod
    def device_synchronize(device_idx: Optional[int] = None):
        """Block the caller until every stream of the device drains.

        Selects the device first when one is given (the runtime's device
        synchronize is per current device) and restores the previous one."""
        rt = _rt._gpu_runtime()
        backend = _rt._active_backend()
        fn_name = backend.get("sync")
        if fn_name is None:
            return
        prev = None
        if device_idx is not None:
            prev = DeviceAllocator.get_device()
            if prev != device_idx:
                DeviceAllocator.set_device(device_idx)
        try:
            ret = getattr(rt, fn_name)()
            if ret != 0:
                DeviceAllocator.clear_last_error()
                raise RuntimeError(f"{fn_name} failed with error {ret}")
        finally:
            if prev is not None and prev != device_idx:
                DeviceAllocator.set_device(prev)

    @staticmethod
    def mem_get_info(device_idx: Optional[int] = None) -> Tuple[int, int]:
        """(free_bytes, total_bytes) of a device, from the runtime."""
        dm = _rt.device_memory()
        mem_fn = _rt._active_backend().get("mem_get_info", "cudaMemGetInfo")   # named in the refusal, as before
        prev = None
        if device_idx is not None:
            prev = DeviceAllocator.get_device()
            if prev != device_idx:
                DeviceAllocator.set_device(device_idx)
        try:
            ret, free_b, total_b = dm.mem_get_info()
            if ret != 0:
                DeviceAllocator.clear_last_error()
                raise RuntimeError(f"{mem_fn} failed with error {ret}")
            return int(free_b), int(total_b)
        finally:
            if prev is not None and prev != device_idx:
                DeviceAllocator.set_device(prev)

    @staticmethod
    def stream_synchronize(stream: int):
        """Block the caller until the given stream drains."""
        rt = _rt._gpu_runtime()
        backend = _rt._active_backend()
        fn_name = backend.get("stream_sync")
        if fn_name is None:
            return
        getattr(rt, fn_name)(ctypes.c_void_p(stream))

    @staticmethod
    def create_event(timing: bool = True) -> int:
        """Create a new event. Returns opaque handle.

        `timing=False` asks for an event that can order streams but cannot be
        used to measure an interval, and that is what almost every event in
        this engine actually wants. It is not a micro-optimisation: recording
        a timing-enabled event costs **9.5 us** on this rig against ~1 us for
        an ordering-only one, because the runtime has to place a clock read
        in the stream and keep the result addressable. Measured 2026-09-03 —
        it was 28 % of a cross-GPU collective's entire host cost.

        Pass timing=True (the default, kept for compatibility) only when the
        handle is going to reach `event_elapsed_ms`.
        """
        rt = _rt._gpu_runtime()
        backend = _rt._active_backend()
        h = ctypes.c_void_p()
        flags_fn = backend.get("event_create_flags")
        if not timing and flags_fn is not None:
            # 0x02 = cudaEventDisableTiming / hipEventDisableTiming.
            ret = getattr(rt, flags_fn)(ctypes.byref(h), ctypes.c_uint(0x02))
        else:
            fn_name = backend.get("event_create")
            if fn_name is None:
                raise RuntimeError(
                    f"Event create unsupported on backend "
                    f"{_rt._detect_gpu_backend()!r}")
            ret = getattr(rt, fn_name)(ctypes.byref(h))
        if ret != 0:
            DeviceAllocator.clear_last_error()   # a refused call must not poison the next one
            raise RuntimeError(f"event create failed (error {ret})")
        return h.value or 0

    @staticmethod
    def destroy_event(event: int):
        """Destroy an event created by create_event."""
        if not event:
            return
        rt = _rt._gpu_runtime()
        backend = _rt._active_backend()
        fn_name = backend.get("event_destroy")
        if fn_name is None:
            return
        getattr(rt, fn_name)(ctypes.c_void_p(event))

    @staticmethod
    def record_event(event: int, stream: int = 0):
        """Capture the current state of `stream` into `event`."""
        rt = _rt._gpu_runtime()
        backend = _rt._active_backend()
        fn_name = backend.get("event_record")
        if fn_name is None:
            return
        getattr(rt, fn_name)(
            ctypes.c_void_p(event), ctypes.c_void_p(stream))

    @staticmethod
    def stream_wait_event(stream: int, event: int, flags: int = 0):
        """Make `stream` wait for `event` without blocking the host."""
        rt = _rt._gpu_runtime()
        backend = _rt._active_backend()
        fn_name = backend.get("stream_wait_event")
        if fn_name is None:
            return
        getattr(rt, fn_name)(
            ctypes.c_void_p(stream), ctypes.c_void_p(event),
            ctypes.c_uint(flags))

    @staticmethod
    def memcpy_async(dst: int, src: int, nbytes: int,
                     kind: int = 1, stream: int = 0):
        """Asynchronous memcpy on a specific stream.

        kind: 0=H2H, 1=H2D, 2=D2H, 3=D2D (matches cudaMemcpyKind).
        For H2D to actually overlap with compute, `src` must be pinned.
        """
        if nbytes <= 0:
            return
        rt = _rt._gpu_runtime()
        backend = _rt._active_backend()
        fn_name = backend.get("memcpy_async")
        if fn_name is None:
            # Fallback to synchronous memcpy if async is unsupported.
            DeviceAllocator.memcpy(dst, src, nbytes, kind)
            return
        getattr(rt, fn_name)(
            ctypes.c_void_p(dst), ctypes.c_void_p(src),
            ctypes.c_size_t(nbytes), ctypes.c_int(kind),
            ctypes.c_void_p(stream))

    @staticmethod
    def event_synchronize(event: int):
        """Block the host until `event` has been reached."""
        if not event:
            return
        rt = _rt._gpu_runtime()
        fn_name = _rt._active_backend().get("event_sync")
        if fn_name is None:
            DeviceAllocator.sync_device()
            return
        ret = getattr(rt, fn_name)(ctypes.c_void_p(event))
        if ret != 0:
            DeviceAllocator.clear_last_error()   # a refused call must not poison the next one
            raise RuntimeError(f"event synchronize failed (error {ret})")

    @staticmethod
    def event_elapsed_ms(start: int, end: int) -> float:
        """Milliseconds between two recorded events, measured ON THE DEVICE.

        Both events MUST have been recorded on streams of the SAME device:
        the runtime's elapsed-time query is undefined across devices because
        each device has its own clock domain. To time a multi-device
        collective, record both events on one device's stream and make that
        stream `stream_wait_event` on the other devices' completion events —
        then the interval covers the critical path without ever comparing
        two clocks.

        The caller is responsible for having synchronized `end`.
        """
        rt = _rt._gpu_runtime()
        fn_name = _rt._active_backend().get("event_elapsed")
        if fn_name is None:
            raise RuntimeError(
                f"event timing unsupported on backend "
                f"{_rt._detect_gpu_backend()!r}")
        fn = getattr(rt, fn_name)
        fn.restype = ctypes.c_int
        out = ctypes.c_float()
        ret = fn(ctypes.byref(out), ctypes.c_void_p(start), ctypes.c_void_p(end))
        if ret != 0:
            DeviceAllocator.clear_last_error()   # a refusal must not poison the next call
            raise RuntimeError(
                f"event elapsed failed (error {ret}) — the usual cause is "
                f"two events recorded on DIFFERENT devices, which has no "
                f"defined answer.")
        return float(out.value)

    @staticmethod
    def can_access_peer(src_dev: int, dst_dev: int) -> bool:
        """Whether `src_dev` can address `dst_dev`'s memory directly.

        False does not mean a copy will fail — it means the driver will
        stage it through host memory, at host-link speed rather than
        interconnect speed.
        """
        if src_dev == dst_dev:
            return True
        rt = _rt._gpu_runtime()
        fn_name = _rt._active_backend().get("device_can_access_peer")
        if fn_name is None:
            return False
        out = ctypes.c_int()
        ret = getattr(rt, fn_name)(
            ctypes.byref(out), ctypes.c_int(src_dev), ctypes.c_int(dst_dev))
        if ret != 0:
            DeviceAllocator.clear_last_error()
        return ret == 0 and bool(out.value)

    # Pairs already granted, so the hot transfer path pays the query once.
    _peer_enabled: set = set()
    # Pairs the hardware refused, so a rig without peer links does not retry
    # on every single transfer.
    _peer_refused: set = set()

    @staticmethod
    def ensure_peer_access(src_dev: int, dst_dev: int) -> bool:
        """Grant src -> dst once, then remember the answer.

        Called on the cross-device transfer path, which is hot enough that
        asking the driver every time would cost more than it saves. Both
        outcomes are memoised: a rig whose cards have no direct links must
        not re-ask on every hand-off.

        A `False` here is not an error — the copy still works, the driver
        just routes it through host memory. It is 2.4x slower at 4 KB and
        8.7x slower at 34 MB on this rig (measured 2026-09-03).
        """
        if src_dev == dst_dev:
            return True
        key = (src_dev, dst_dev)
        if key in DeviceAllocator._peer_enabled:
            return True
        if key in DeviceAllocator._peer_refused:
            return False
        if os.environ.get("NBX_DISABLE_PEER_ACCESS") == "1":
            # Escape hatch and A/B control. Direct card-to-card access is a
            # driver-level grant; if a rig ever misbehaves under it, this
            # restores the host-staged route without a code change, and it is
            # how the speedup was measured against the old behaviour.
            DeviceAllocator._peer_refused.add(key)
            return False
        try:
            granted = DeviceAllocator.enable_peer_access(src_dev, dst_dev)
        except RuntimeError:
            granted = False
        (DeviceAllocator._peer_enabled if granted
         else DeviceAllocator._peer_refused).add(key)
        return granted

    @staticmethod
    def enable_peer_access(src_dev: int, dst_dev: int) -> bool:
        """Let `src_dev` reach `dst_dev` over the interconnect directly.

        Direction matters: this grants src -> dst only, so a bidirectional
        pair needs both calls. Returns True if access is in force afterwards
        (including when it already was), False if the hardware cannot do it.

        Why this exists: a cross-device `memcpy(kind=3)` with peer access
        OFF is bounced through host memory by the driver. On this rig every
        GPU pair reports NV2 (two bonded NVLinks) and the copy still would
        not use them. Measured cost of the difference:
        `validation_outputs/tp_collective_latency_2026_09_03/`.
        """
        if src_dev == dst_dev:
            return True
        if not DeviceAllocator.can_access_peer(src_dev, dst_dev):
            return False
        rt = _rt._gpu_runtime()
        fn_name = _rt._active_backend().get("device_enable_peer_access")
        if fn_name is None:
            return False
        previous = DeviceAllocator.get_device()
        try:
            DeviceAllocator.set_device(src_dev)
            ret = getattr(rt, fn_name)(ctypes.c_int(dst_dev), ctypes.c_uint(0))
        finally:
            DeviceAllocator.set_device(previous)
        # 704 = cudaErrorPeerAccessAlreadyEnabled. Idempotence is the point:
        # callers enable defensively and must not have to track state.
        if ret in (0, 704):
            return True
        raise RuntimeError(
            f"enable peer access {src_dev}->{dst_dev} failed (error {ret})")

    @staticmethod
    def ensure_triton_device(device_idx: int):
        """Set GPU device for Triton JIT context. Zero torch. Universal.

        Uses cudaSetDevice (runtime) + Triton's own driver API.
        Does NOT call cuCtxSetCurrent — that conflicts with the runtime
        context on multi-GPU systems.

        C2 fast path: called before every kernel launch, so when the
        requested device is already fully current (see the
        per-thread device-cache invariant) the pair of
        driver calls is skipped. Any device change outside this
        function must invalidate the cache (set_device does so
        automatically; raw switches call
        invalidate_current_device_cache()).
        """
        if device_idx == _rt._cached_device_idx():
            return
        # 1. Set runtime device (cudaSetDevice / hipSetDevice)
        _rt.device_memory().set_device(device_idx)
        # 2. Tell the launcher which device to target. The NeuroBrix launcher
        #    binds the runtime's primary context of the current device at each
        #    launch; only the differential arm (NBX_LAUNCHER=triton) still
        #    drives upstream's launcher, whose driver must be told (and whose
        #    backend probes import torch — R33, universal since 2026-09-05).
        if os.environ.get("NBX_LAUNCHER", "nbx").lower() == "triton":
            import triton.runtime.driver
            active = triton.runtime.driver.active
            # A single-device backend has nothing to set and need not
            # implement the setter: triton-msl's MetalDriver provides
            # `get_current_device` and no `set_current_device` (measured
            # 2026-09-05), and this arm is exactly how the Metal A/B is run.
            # Falling through would raise AttributeError at the first launch;
            # assuming it worked would risk launching on a device nobody
            # selected. So the absence is handled by VERIFYING instead.
            set_current = getattr(active, "set_current_device", None)
            if set_current is not None:
                set_current(device_idx)
            else:
                current = active.get_current_device()
                if current != device_idx:
                    raise RuntimeError(
                        f"Triton backend "
                        f"{active.get_current_target().backend!r} exposes no "
                        f"set_current_device and reports device {current}, "
                        f"but device {device_idx} was requested. The kernel "
                        f"would launch on the wrong device, so this refuses "
                        f"instead.")
        # Cache only after BOTH calls succeeded (this thread's state).
        _rt._DEVICE_CACHE.idx = device_idx
