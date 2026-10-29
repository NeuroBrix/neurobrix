"""The device runtime in force: which backend this process detected (CUDA, HIP, Metal), its runtime library, the
current device, and the Triton backend pinned to it. Cut out of tensor.py (M2, 2026-10-29), unchanged."""

from __future__ import annotations
from nbxtensor.memory import allocator as _alloc

import ctypes
import functools
import os
import threading
from typing import Optional

# module level, not in device_memory(): it runs on the finalizer path, where an import at shutdown raises
from nbxtensor.memory.device_memory import AbiMemory as _AbiMemory



# ============================================================================
# GPU RUNTIME — hardware-agnostic via ctypes (CUDA/ROCm)
# ============================================================================

def _environment_runtime_libs(package: str, names) -> list:
    """The runtime libraries the ENVIRONMENT ships, ahead of the system's.
    torch's CUDA wheels carry their own `nvidia/cuda_runtime/lib/libcudart.so.12`
    and bind against it; a `libcudart.so.12` resolved by ldconfig is the
    machine's toolkit (12.2 on this rack), and when the engine's ctypes loader
    opened THAT one first, torch 2.14's `libc10_cuda.so` found an older runtime
    than it was built on — `undefined symbol: cudaGetDriverEntryPointByVersion`
    (2026-09-16, the stack alignment). One process, one runtime: the wheel's
    own when it exists, the system's otherwise. Read from the package's
    location, never a typed path."""
    try:
        import importlib.util
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        return []
    import os
    root = list(spec.submodule_search_locations)[0]
    out = []
    for n in names:
        cand = os.path.join(root, "lib", n)
        if os.path.exists(cand):
            out.append(cand)
    return out


# Runtime API mapping per backend
_GPU_BACKENDS = {
    "cuda": {
        "rt_libs": _environment_runtime_libs("nvidia.cuda_runtime", ["libcudart.so.12"])
                   + ["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"],
        "malloc": "cudaMalloc", "free": "cudaFree",
        "memcpy": "cudaMemcpy", "memset": "cudaMemset",
        "set_device": "cudaSetDevice",
        "get_last_error": "cudaGetLastError", "get_device": "cudaGetDevice",
        "device_count": "cudaGetDeviceCount",
        "mem_get_info": "cudaMemGetInfo",
        "sync": "cudaDeviceSynchronize",
        "malloc_host": "cudaMallocHost", "free_host": "cudaFreeHost",
        # Stream-ordered memory allocator (CUDA 11.2+). The driver
        # handles kernel-lifetime correctness per-stream, so freeing a
        # tensor while an async kernel is still reading from it is
        # safe — the free is ordered against the stream's pending work.
        # Using stream=0 (default stream) keeps us aligned with the
        # rest of NeuroBrix's single-default-stream execution model;
        # when pipelining introduces a transfer stream in a future
        # session we'll route those allocs to it.
        "malloc_async": "cudaMallocAsync", "free_async": "cudaFreeAsync",
        # Stream + event API — used by zero3 pipelining for async H2D
        # overlap of block N+1 with compute on block N.
        "stream_create": "cudaStreamCreate",
        "stream_destroy": "cudaStreamDestroy",
        "stream_sync": "cudaStreamSynchronize",
        "event_create": "cudaEventCreate",
        "event_create_flags": "cudaEventCreateWithFlags",
        "event_destroy": "cudaEventDestroy",
        "event_record": "cudaEventRecord",
        "stream_wait_event": "cudaStreamWaitEvent",
        "memcpy_async": "cudaMemcpyAsync",
        # Event timing — the ONLY way to measure a device-side interval
        # without a host round-trip. `event_elapsed` is undefined across
        # devices, so both events of a pair must be recorded on streams of
        # the SAME device (see DeviceAllocator.event_elapsed_ms).
        "event_sync": "cudaEventSynchronize",
        "event_elapsed": "cudaEventElapsedTime",
        # Peer access. Without it a cross-device D2D copy is STAGED THROUGH
        # HOST MEMORY by the driver — it never touches NVLink. Enabling it
        # is per ORDERED pair and per process.
        "device_can_access_peer": "cudaDeviceCanAccessPeer",
        "device_enable_peer_access": "cudaDeviceEnablePeerAccess",
    },
    "hip": {
        "rt_libs": ["libamdhip64.so", "libamdhip64.so.5"],
        "malloc": "hipMalloc", "free": "hipFree",
        "memcpy": "hipMemcpy", "memset": "hipMemset",
        "set_device": "hipSetDevice",
        "get_last_error": "hipGetLastError", "get_device": "hipGetDevice",
        "device_count": "hipGetDeviceCount",
        "mem_get_info": "hipMemGetInfo",
        "sync": "hipDeviceSynchronize",
        "malloc_host": "hipHostMalloc", "free_host": "hipHostFree",
        # ROCm 4.3+ exposes the same stream-ordered API under hip*
        # names. Not tested in this pass (investigation was V100-only),
        # but the symbol mapping is kept for symmetry.
        "malloc_async": "hipMallocAsync", "free_async": "hipFreeAsync",
        "stream_create": "hipStreamCreate",
        "stream_destroy": "hipStreamDestroy",
        "stream_sync": "hipStreamSynchronize",
        "event_create": "hipEventCreate",
        "event_create_flags": "hipEventCreateWithFlags",
        "event_destroy": "hipEventDestroy",
        "event_record": "hipEventRecord",
        "stream_wait_event": "hipStreamWaitEvent",
        "memcpy_async": "hipMemcpyAsync",
        "event_sync": "hipEventSynchronize",
        "event_elapsed": "hipEventElapsedTime",
        # ROCm exposes peer access under the same semantics; xGMI plays the
        # role NVLink plays here.
        "device_can_access_peer": "hipDeviceCanAccessPeer",
        "device_enable_peer_access": "hipDeviceEnablePeerAccess",
    },
}


# ============================================================================
# CURRENT-DEVICE CACHE (P-KERNEL-LAUNCH-HYGIENE C2)
# ============================================================================
#
# `ensure_triton_device` is called before every Triton kernel launch
# (via `set_device`). The cudaSetDevice + triton set_current_device
# pair is pure overhead when the device is already current, so we cache
# the last device index that was fully synchronised (runtime device AND
# Triton driver device both pointing at it) and skip the pair on a hit.
#
# Cache invariant: `_cached_device_idx() == idx` (per-thread) implies the CUDA/HIP
# runtime current device is `idx` AND Triton's active driver device is
# `idx`. Every NeuroBrix-side device change maintains the invariant:
#   - `ensure_triton_device` refreshes the cache after a full sync;
#   - `DeviceAllocator.set_device` (the single choke point for runtime-
#     only switches) invalidates the cache when it switches to a device
#     that differs from the cached one (it does not touch Triton's
#     driver device, so the pair is no longer known-coherent);
#   - `most_free_device` (raw per-device probe loop) invalidates
#     unconditionally before returning.
# Device changes that bypass NBX entirely (torch.cuda.set_device on the
# compiled-mode path) cannot interleave with a warm cache inside a
# triton run (R33: no torch in triton paths); triton run entry points
# invalidate the cache defensively (see triton/sequence.py run()).
#
# THREAD-LOCAL, not process-global (D9 root cause, 2026-07-27): the
# cudaSetDevice/hipSetDevice state the invariant mirrors IS per-thread.
# A process-global cache warmed by one thread makes every OTHER thread
# skip the sync while its own runtime device is still the default —
# the serving daemon's worker thread then launches kernels holding
# cuda:N pointers from a device-0 context ("CUDA: out of memory" at
# launch with gigabytes free on every device; cmd_run, single-threaded,
# can never reproduce it).
_DEVICE_CACHE = threading.local()


def _cached_device_idx() -> Optional[int]:
    return getattr(_DEVICE_CACHE, "idx", None)


def invalidate_current_device_cache() -> None:
    """Drop the ensure_triton_device fast-path cache (this thread).

    Call after any device switch that does not go through
    `DeviceAllocator.set_device` / `ensure_triton_device`, so the next
    `ensure_triton_device` performs the full runtime + Triton sync.
    """
    _DEVICE_CACHE.idx = None


def set_device(t):
    """Sync Triton/CUDA driver context with the tensor's physical device.

    Called before every Triton kernel launch. Triton's internal driver
    state must match `t._device_idx`, or the kernel will target the
    wrong device and raise "Pointer argument cannot be accessed from
    Triton" when the pointer lives on a different GPU. Single source of
    truth — imported by wrappers.py, dispatch.py, shape_utils.py,
    triton/weight_loader.py so every kernel-launching module honours
    the contract.
    """
    if hasattr(t, '_device_idx'):
        _alloc.DeviceAllocator.ensure_triton_device(t._device_idx)


#: This engine names its backends for the vendor RUNTIME it loads (`cuda`, `hip`,
#: `metal`); Triton names its own for the vendor's COMPILER target, which is the
#: directory under `triton/backends/`. The two vocabularies are not the same and
#: the mapping between them belongs here, once.
_TRITON_BACKEND_BY_RUNTIME = {"cuda": "nvidia", "hip": "amd", "metal": "metal"}


def _pin_triton_backend(name: str) -> str:
    """Tell Triton which backend is here, then return the name.

    R33, and it applies on every backend, not just Apple. Asking
    `triton.runtime.driver.active` for anything makes Triton call
    `is_active()` on EVERY registered backend to find the live one, and
    upstream's AMD probe runs `import torch` inside its own — so a CUDA box
    with no AMD card pays a torch import the moment any module asks the
    driver a question. `kernels/ops/matmul.py` asks one at import time, which
    is how it reached us.

    `TRITON_DEFAULT_BACKEND` makes `_create_driver` select the named backend
    directly and probe nothing else. We already know the answer here — this
    function found it without importing anything — so we say so. `setdefault`,
    because an explicit choice by the user or a test outranks ours.

    **The name has to be TRITON'S, and ours is not.** This engine's backends are
    called `cuda`, `hip` and `metal`; Triton's registry keys are the directory
    names under `triton/backends/` — `nvidia` and `amd` — and `_create_driver`
    raises `Unknown backend device '<name>'` on anything else. Pinning `cuda` was
    therefore always wrong and never showed, because this engine's launcher does
    every launch itself and never asks Triton for a driver. It showed the moment
    a launch went down Triton's own path: torch 2.14 ships in-tree Triton ops and
    the seam now hands those back to Triton, which then read the pin (2026-09-17).

    A name Triton does not have is worse than no pin at all, so the mapped name is
    checked against Triton's own registry when that can be read without probing —
    `triton.backends` is a dict built at import and `is_active()` runs only inside
    `_create_driver` — and the pin is skipped rather than set to something that
    raises.
    """
    triton_name = _TRITON_BACKEND_BY_RUNTIME.get(name, name)
    try:
        from triton.backends import backends as _triton_backends
        if triton_name not in _triton_backends:
            return name                       # do not pin a name Triton would refuse
    except Exception:                         # noqa: BLE001 — no Triton, or a shape we do not know
        pass
    os.environ.setdefault("TRITON_DEFAULT_BACKEND", triton_name)
    return name


@functools.lru_cache(maxsize=1)
def gpu_backend() -> str:
    """Detect GPU backend: 'cuda', 'hip' or 'metal' — from the vendor runtime
    the process can load, or `NBX_GPU_BACKEND`.

    **What this does NOT answer: whether a device is there.** On CUDA and ROCm
    it succeeds when the vendor's runtime LIBRARY loads, which it does on a host
    with the toolkit and no visible card — it answered "cuda" under
    `CUDA_VISIBLE_DEVICES=` and five tests that had gated on it went on to fail
    at their first allocation with `cudaErrorNoDevice` instead of skipping
    (2026-09-16, register 62). On Metal the two questions coincide, because
    there is no library to dlopen and the probe opens the device itself — which
    is why the same call is an executing probe on one backend and a naming one
    on the others. A caller asking "can I run here" wants
    `DeviceAllocator.device_count() > 0`, which asks the driver, or an
    allocation, which answers for the memory too.

    Never through Triton's driver
    probe: `triton.runtime.driver.active` asks every backend `is_active()`,
    and those probes import torch (R33, universal since 2026-09-05 — this
    call was the first torch import of the launch path).

    **Every probe here is torch-free, and the order is R33 rather than
    taste.** Asking Triton which backend is active — the obvious first
    question, and what this function used to do — makes Triton call
    `is_active()` on EVERY registered backend to find the live one, and
    upstream's AMD probe runs `import torch` inside its own. So merely asking
    puts torch in the process, on a CUDA box with no AMD card as much as on a
    Mac. `kernels/ops/matmul.py` asks that question at module import, which is
    how it reached the engine.

    The Metal probe carries no platform strings. It opens the device and asks
    it, because the property that matters is unified memory (which is what
    makes one address valid for both processors) and that is a device answer,
    not an `arm64` answer. It has no `rt_libs` row to try, which is why it is
    not an entry in `_GPU_BACKENDS`: that dict holds symbol tables over one C
    ABI and `test_device_backend_seam.py` pins it to exactly {cuda, hip}.

    Whatever answers, `_pin_triton_backend` records it in
    `TRITON_DEFAULT_BACKEND`, so that the differential arm
    (`NBX_LAUNCHER=triton`) — the only path that still reaches
    `triton.runtime.driver.active` — resolves straight to that backend and
    probes nothing else.
    """
    forced = os.environ.get("NBX_GPU_BACKEND")
    if forced:
        if forced not in _GPU_BACKENDS and forced != "metal":
            raise RuntimeError(f"NBX_GPU_BACKEND={forced!r} is not a known backend "
                               f"({sorted(_GPU_BACKENDS) + ['metal']})")
        return _pin_triton_backend(forced)
    # The backend TABLE decides: the first vendor runtime the process can
    # load names the backend — adding a backend is adding a table entry.
    for name, backend in _GPU_BACKENDS.items():
        for lib in backend["rt_libs"]:
            try:
                ctypes.cdll.LoadLibrary(lib)
                return _pin_triton_backend(name)
            except OSError:
                continue
    # Apple GPUs. Not a "fallback" in the sense the loop above is: there is
    # no library to dlopen, so the probe opens the device itself and answers
    # only when a usable one came back.
    from nbxtensor.backend.metal_device import metal_device_available
    if metal_device_available():
        return _pin_triton_backend("metal")
    raise RuntimeError("No GPU runtime found (tried CUDA, ROCm/HIP and Metal)")


def device_label(tensor) -> str:
    """How a tensor's placement should READ in a message, e.g. "metal:0".

    `NBXTensor._device` is the engine's internal token for "device memory"
    and its value is the string `"cuda"` on every backend — `is_cuda()` and
    the allocator paths are written against that token, so it is not a name
    to change. It is also not a name to PRINT: on an Apple machine a refusal
    that says `arg0=cuda:0` is telling the reader something false about the
    hardware, and the first Metal diagnostics did exactly that.

    So the token stays and the label is resolved, once, from the backend the
    engine actually detected. A tensor on the host still reads "cpu", which
    is true everywhere.
    """
    placement = getattr(tensor, "_device", None)
    index = getattr(tensor, "_device_idx", "?")
    if placement is None:
        return "?"
    if placement != "cuda":
        return f"{placement}:{index}"
    try:
        return f"{gpu_backend()}:{index}"
    except Exception:
        # No backend resolvable: say the token rather than invent a name.
        return f"{placement}:{index}"


def backend_table() -> dict:
    """The entry-point name table for the detected backend.

    Metal resolves to a table that is deliberately NOT a row in
    `_GPU_BACKENDS`. That dict holds symbol tables over one C ABI, and
    `test_device_backend_seam.py` pins it to exactly `{"cuda", "hip"}`
    because a key in one row and missing from the other is a crash on that
    hardware alone. Metal is a different API with a different set of honest
    entry points, so it gets its own table — the second *implementation*
    behind this seam that the seam's own docstring describes.
    """
    name = gpu_backend()
    if name == "metal":
        return _metal_backend_table()
    return _GPU_BACKENDS[name]


@functools.lru_cache(maxsize=1)
def _metal_backend_table() -> dict:
    """The Metal entry-point table, imported ONCE and then cached.

    Cached rather than imported per call because `backend_table` runs
    inside `NBXTensor.__del__`, and a finalizer can fire during interpreter
    shutdown when `sys.meta_path` is already None — an import there raises
    `ImportError: Python is likely shutting down` and every tensor still
    alive prints a traceback on the way out. The cuda/hip path never had the
    problem: `_GPU_BACKENDS` is a module global with no import behind it.
    One `lru_cache` gives Metal the same property.
    """
    from nbxtensor.backend.metal_device import METAL_BACKEND
    return METAL_BACKEND


@functools.lru_cache(maxsize=1)
def gpu_runtime():
    """The runtime of the GPU backend in force: the Metal runtime object on Metal, else the vendor's runtime library
    (CUDA, HIP) loaded through its C ABI, the first of the backend table's libraries that loads."""
    name = gpu_backend()
    if name == "metal":
        from nbxtensor.backend.metal_device import runtime
        return runtime()          # itself cached; see _metal_backend_table
    backend = backend_table()
    for lib in backend["rt_libs"]:
        try:
            return ctypes.cdll.LoadLibrary(lib)
        except OSError:
            continue
    raise RuntimeError(f"GPU runtime not found for {name}")


def device_memory():
    """The memory primitives of the backend in force (`memory.device_memory.DeviceMemory`): the account
    (`memory.allocator.DeviceAllocator`) asks these and nothing else of the device's memory. Built per call from the
    runtime and its table, never cached: a substituted runtime or table is read at once."""
    table = backend_table()
    if gpu_backend() == "metal" and table is _metal_backend_table():
        # Metal's own primitives, no C ABI behind them; a substituted table (a test's C-ABI driver) is read as one
        from nbxtensor.memory.metal import MetalMemory
        return MetalMemory(gpu_runtime())
    return _AbiMemory(gpu_runtime(), table)
