"""The Metal device runtime, behind the seam `DeviceAllocator` already owns.

Step 5 of the Metal adoption plan. `cuda` and `hip` are two symbol tables over
one C ABI, so `_GPU_BACKENDS` can hold both as rows that differ only by name.
**Metal cannot be a third row** — it is an Objective-C API with `MTLDevice`
and `MTLBuffer` and there is no `metalMalloc` to name — and
`test_device_backend_seam.py` pins exactly that, both in its contract
assertion (`set(backends) == {"cuda", "hip"}`) and in its own docstring:
*"it will be a second implementation behind this class rather than a third
row in the dict."*

This module is that second implementation. It presents the **call contract**
the seam already uses — a runtime object whose entry points take `ctypes`
arguments, write results through `ctypes.byref`, and return an integer status
— so every one of the 49 runtime touches inside `DeviceAllocator` works
unchanged. The ctypes shape is the seam's existing convention, not a pretence
that Metal has a C ABI.

## What unified memory deletes

Apple Silicon has one physical memory shared by CPU and GPU, so a buffer made
with `MTLResourceStorageModeShared` has a single address that is valid from
both. `contents()` hands that address back, and it is the same address a
kernel receives. Three consequences, and they are simplifications rather than
work:

* **H2D and D2H do not exist.** All four `cudaMemcpy` kinds are one byte copy
  between two addresses in the same address space.
* **Pinned host memory does not exist either**, because there is no DMA to
  pin for. `malloc_host` returns a shared buffer, which genuinely satisfies
  what `cudaMallocHost` is for: a host address the device can read with no
  copy. That is a correct mapping, not a fallback.
* **Peer access does not apply.** `MTLCopyAllDevices()` returns one device on
  this hardware, and the key is simply absent, which makes
  `DeviceAllocator.enable_peer_access` answer False — true, not silent.

## Zero fallback

Every failure is loud. If Metal is absent, `runtime()` raises. If a buffer
cannot be allocated, the entry point returns a non-zero status and
`DeviceAllocator.malloc_cuda` raises its OOM with full diagnostics. Nothing
here ever hands back host memory pretending it is device memory, and nothing
degrades to CPU.

What Metal has no honest equivalent for is **omitted rather than stubbed**:
events, `memcpy_async`, the stream-ordered allocator. The seam already turns
a missing key into an explicit
`RuntimeError("... unsupported on backend 'metal'")`, so omission is the
loud refusal and a no-op stub would have been the silent wrong.

R33 preserved — no torch, at the boundary included.
R34 preserved — nothing here is keyed on a model name.
"""

from __future__ import annotations

import ctypes
import os
import threading
from typing import Dict, Optional, Tuple

# Status codes. 0 is success, mirroring the C runtimes the seam was written
# against; the non-zero values are ours and appear only in NeuroBrix errors.
_OK = 0
_ERR_ALLOC = 1
_ERR_BAD_DEVICE = 2
_ERR_UNKNOWN_POINTER = 3
_ERR_BAD_ARGUMENT = 4


class MetalUnavailableError(RuntimeError):
    """Metal, or its Python bindings, are not usable on this machine.

    Raised rather than returned so that no caller can mistake absence for an
    empty device list and quietly continue on the CPU.
    """


def _deref(arg):
    """The object behind a `ctypes.byref(...)` out-parameter.

    The seam passes out-parameters exactly as it does to a C runtime. A
    `CArgObject` carries the object it references as `_obj`; a real pointer
    carries `contents`. Both are accepted so this works whichever the caller
    used.
    """
    if hasattr(arg, "_obj"):
        return arg._obj
    if hasattr(arg, "contents"):
        return arg.contents
    raise TypeError(f"not an out-parameter: {type(arg).__name__}")


def _as_int(value) -> int:
    """A Python int from a ctypes scalar or a plain int."""
    return int(getattr(value, "value", value) or 0)


class MetalRuntime:
    """One Metal device, presented as the runtime object the seam expects.

    Instantiated once per process by `runtime()`. Not thread-safe by
    construction beyond the registry lock, which matches the rest of
    `DeviceAllocator` ("single-threaded per component by convention").
    """

    def __init__(self) -> None:
        try:
            import Metal  # pyobjc-framework-Metal
        except ImportError as exc:                      # pragma: no cover
            raise MetalUnavailableError(
                "The Metal Python bindings are not installed. NeuroBrix "
                "reaches Apple GPUs through pyobjc:\n"
                "    pip install pyobjc-framework-Metal\n"
                "It is a dependency of the Metal Triton backend, so this "
                "usually means the backend itself is missing — see "
                "neurobrix.triton.metal_backend."
            ) from exc

        self._Metal = Metal
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            raise MetalUnavailableError(
                "No Metal device. `MTLCreateSystemDefaultDevice()` returned "
                "nil, so this process cannot reach a GPU at all — a headless "
                "session or a virtualised host without GPU passthrough will "
                "do this. NeuroBrix refuses rather than computing on the CPU "
                "under a device name."
            )
        if not device.hasUnifiedMemory():
            raise MetalUnavailableError(
                f"Metal device {device.name()!r} does not report unified "
                f"memory. Every address this allocator hands out is a "
                f"shared-storage buffer's `contents()`, which is only the "
                f"SAME address the GPU sees when host and device share the "
                f"memory. On a discrete Metal GPU that assumption is false "
                f"and the pointers would be silently wrong, so this refuses "
                f"instead. Apple Silicon reports unified memory; an Intel "
                f"Mac with a discrete card does not."
            )
        self._device = device
        # One queue owned by the allocator. See `sync` for exactly what it
        # does and does not order against.
        self._queue = device.newCommandQueue()
        if self._queue is None:                         # pragma: no cover
            raise MetalUnavailableError(
                "Metal device present but `newCommandQueue()` returned nil.")

        # The VM page size, read from the system. Every allocation is rounded
        # up to a multiple of it, and that is not a micro-optimisation:
        # `MTLDevice.newBufferWithBytesNoCopy:length:options:deallocator:` —
        # which is how the Triton Metal backend turns one of our pointers
        # into a buffer it can bind — requires a PAGE-ALIGNED pointer and a
        # page-multiple length. Metal sub-allocates buffers smaller than a
        # page inside one, so `contents()` for them lands mid-page. Measured
        # 2026-09-05: of twelve sizes from 1 byte to 64 KB, the eight below
        # one page all came back unaligned (offsets 1280, 1536, 1792, 2048,
        # 2304, 3328, 7424, 15616) and every one of them was page-aligned
        # once the length was rounded up. Handing a sub-page pointer to
        # NoCopy is undefined behaviour, and it showed as
        # "Failed to create Metal buffer" under memory pressure and, once, a
        # SIGBUS.
        self._page_size = int(os.sysconf("SC_PAGESIZE"))

        self._lock = threading.RLock()
        # address -> (MTLBuffer, memoryview, nbytes). The buffer entry is what
        # keeps the allocation alive: dropping it is the free.
        self._buffers: Dict[int, Tuple[object, memoryview, int]] = {}
        self._host_buffers: Dict[int, Tuple[object, memoryview, int]] = {}
        # Bytes this allocator is currently holding. Ours, exactly, and it
        # goes DOWN on free — which is the property `currentAllocatedSize()`
        # turned out not to have (see `_budget_bytes`).
        self._live_bytes = 0
        # handle -> MTLCommandQueue, for the stream API.
        self._queues: Dict[int, object] = {}
        self._next_queue_handle = 1
        # handle -> event record, for the event API.
        self._events: Dict[int, dict] = {}
        self._next_event_handle = 1

        # `DeviceAllocator.event_elapsed_ms` sets `.restype` on the entry
        # point before calling it — a ctypes idiom that a bound method
        # rejects. A function object accepts attributes, so this one entry
        # point is exposed as a closure rather than a method. The seam's
        # convention, honoured rather than worked around.
        def event_elapsed(out_ms, start, end, _self=self):
            return _self._event_elapsed(out_ms, start, end)

        self.event_elapsed = event_elapsed

        try:
            self._device_count = len(Metal.MTLCopyAllDevices())
        except Exception:                               # pragma: no cover
            # MTLCopyAllDevices is macOS-only; a device exists, so it is one.
            self._device_count = 1

    # -- properties read from the device, never assumed ---------------------

    @property
    def device_name(self) -> str:
        return str(self._device.name())

    @property
    def has_unified_memory(self) -> bool:
        return bool(self._device.hasUnifiedMemory())

    @property
    def max_threadgroup_memory(self) -> int:
        """Threadgroup memory per threadgroup, in bytes, from the device.

        The one number the Apple profile calls its binding constraint, and
        the profile has it `hand_curated_from_docs`. Exposed here so it can
        be measured rather than read.
        """
        return int(self._device.maxThreadgroupMemoryLength())

    # -- allocation ---------------------------------------------------------

    def _budget_bytes(self) -> int:
        """The device's own working-set budget, read from the device."""
        return int(self._device.recommendedMaxWorkingSetSize())

    # Why the budget is checked against OUR live count and not against
    # `MTLDevice.currentAllocatedSize()`, which is the obvious candidate:
    # measured 2026-09-05, that counter does not come back down. Four cycles
    # of allocating 2 GiB and releasing it read 2048 / 4096 / 6144 / 8192 MB
    # while the process RSS stayed flat at ~2.3 GB — so the memory really is
    # released and reused, and the counter is cumulative rather than live.
    # Using it as a budget gauge would have produced an allocator that OOMs
    # permanently after enough churn, on a device with nothing allocated.

    def _new_shared_buffer(self, nbytes: int):
        """A shared-storage MTLBuffer plus the address the CPU and GPU share.

        The device's working-set budget is enforced HERE rather than left to
        Metal, and that is deliberate. `recommendedMaxWorkingSetSize` is a
        recommendation: Metal will keep handing out buffers past it, backed
        by a memory the whole machine is sharing, until the system starts
        paging or the process is killed. An allocator that never says no
        turns "fill the device until it refuses" — which is how the engine's
        own OOM-reclaim path is exercised — into an unbounded claim on the
        user's RAM. Apple's number is the honest ceiling and it is read from
        the device, never guessed.
        """
        Metal = self._Metal
        page = self._page_size
        allocated = (nbytes + page - 1) // page * page
        with self._lock:
            live = self._live_bytes
        if live + allocated > self._budget_bytes():
            return None, None, 0, 0
        buffer = self._device.newBufferWithLength_options_(
            allocated, Metal.MTLResourceStorageModeShared)
        if buffer is None:
            return None, None, 0, 0
        view = buffer.contents().as_buffer(allocated)
        # `from_buffer` holds an export on the view only for as long as the
        # returned object lives; the address belongs to the MTLBuffer, which
        # the registry keeps alive, so it stays valid once the keeper goes.
        keeper = ctypes.c_char.from_buffer(view)
        address = ctypes.addressof(keeper)
        del keeper
        if address % page:                              # pragma: no cover
            # Rounding is supposed to guarantee this. If Metal ever stops
            # honouring it, refuse rather than hand out a pointer the
            # backend will wrap into a buffer with undefined behaviour.
            return None, None, 0, 0
        return buffer, view, address, allocated

    def malloc(self, out_ptr, size) -> int:
        """`cudaMalloc(&ptr, size)`."""
        nbytes = _as_int(size)
        slot = _deref(out_ptr)
        if nbytes <= 0:
            slot.value = 0
            return _ERR_BAD_ARGUMENT if nbytes < 0 else _OK
        buffer, view, address, allocated = self._new_shared_buffer(nbytes)
        if buffer is None or not address:
            slot.value = 0
            return _ERR_ALLOC
        with self._lock:
            self._buffers[address] = (buffer, view, allocated)
            self._live_bytes += allocated
        slot.value = address
        return _OK

    def free(self, ptr) -> int:
        """`cudaFree(ptr)`. Dropping the MTLBuffer reference IS the free."""
        address = _as_int(ptr)
        if not address:
            return _OK
        with self._lock:
            entry = self._buffers.pop(address, None)
            if entry is not None:
                self._live_bytes -= entry[2]
        if entry is None:
            # Freeing a pointer this runtime never handed out is a bug in the
            # caller, and it must not pass silently: on CUDA the same call
            # returns cudaErrorInvalidValue.
            return _ERR_UNKNOWN_POINTER
        entry[1].release()
        return _OK

    def malloc_host(self, out_ptr, size) -> int:
        """`cudaMallocHost(&ptr, size)` — a shared buffer, and that is exact.

        `cudaMallocHost` exists to page-lock host memory so a DMA engine can
        stream it to the device. Apple Silicon has no such transfer: the
        address a shared buffer hands back is already readable by the GPU
        with no copy, which is the whole property the caller wants.
        """
        nbytes = _as_int(size)
        slot = _deref(out_ptr)
        if nbytes <= 0:
            slot.value = 0
            return _ERR_BAD_ARGUMENT if nbytes < 0 else _OK
        buffer, view, address, allocated = self._new_shared_buffer(nbytes)
        if buffer is None or not address:
            slot.value = 0
            return _ERR_ALLOC
        with self._lock:
            self._host_buffers[address] = (buffer, view, allocated)
            self._live_bytes += allocated
        slot.value = address
        return _OK

    def free_host(self, ptr) -> int:
        address = _as_int(ptr)
        if not address:
            return _OK
        with self._lock:
            entry = self._host_buffers.pop(address, None)
            if entry is not None:
                self._live_bytes -= entry[2]
        if entry is None:
            return _ERR_UNKNOWN_POINTER
        entry[1].release()
        return _OK

    # -- copies -------------------------------------------------------------

    def memcpy(self, dst, src, size, kind=None) -> int:
        """`cudaMemcpy(dst, src, n, kind)` — one address space, so one copy.

        `kind` is accepted and ignored on purpose: H2D, D2H, D2D and H2H are
        the same operation when host and device share the memory. `memmove`
        rather than `memcpy` because it is correct for every input the
        caller can produce, including the overlap CUDA leaves undefined.
        """
        nbytes = _as_int(size)
        if nbytes <= 0:
            return _OK if nbytes == 0 else _ERR_BAD_ARGUMENT
        dst_addr, src_addr = _as_int(dst), _as_int(src)
        if not dst_addr or not src_addr:
            return _ERR_BAD_ARGUMENT
        ctypes.memmove(ctypes.c_void_p(dst_addr),
                       ctypes.c_void_p(src_addr), nbytes)
        return _OK

    def memset(self, ptr, value, size) -> int:
        nbytes = _as_int(size)
        if nbytes <= 0:
            return _OK if nbytes == 0 else _ERR_BAD_ARGUMENT
        address = _as_int(ptr)
        if not address:
            return _ERR_BAD_ARGUMENT
        ctypes.memset(ctypes.c_void_p(address),
                      _as_int(value) & 0xFF, nbytes)
        return _OK

    # -- device queries -----------------------------------------------------

    def set_device(self, index) -> int:
        """Metal exposes one GPU here; anything but 0 is an error, not a
        silent clamp to the only device."""
        return _OK if _as_int(index) == 0 else _ERR_BAD_DEVICE

    def get_device(self, out_index) -> int:
        _deref(out_index).value = 0
        return _OK

    def device_count(self, out_count) -> int:
        _deref(out_count).value = self._device_count
        return _OK

    def mem_get_info(self, out_free, out_total) -> int:
        """`cudaMemGetInfo(&free, &total)`, read from the device.

        `total` is `recommendedMaxWorkingSetSize()` — Apple's own budget for
        this device, which on unified memory is the meaningful ceiling rather
        than installed RAM, because that RAM is shared with everything else
        running. `free` is that budget minus `currentAllocatedSize()`, so it
        is a budget headroom rather than a physical free-page count. The
        callers of this in NeuroBrix are feasibility probes and the pool's
        parked-bytes cap, both of which want exactly that.
        """
        total = self._budget_bytes()
        with self._lock:
            used = self._live_bytes
        _deref(out_total).value = total
        _deref(out_free).value = max(0, total - used)
        return _OK

    # -- ordering -----------------------------------------------------------

    def sync(self) -> int:
        """`cudaDeviceSynchronize()` on the allocator's own queue.

        A real device round trip: an empty command buffer is committed and
        waited on, so anything already enqueued on this queue has retired.

        **What it does not cover, named rather than left silent:** the Triton
        Metal backend dispatches kernels on a queue of its own, and this does
        not order against that queue. Today nothing else enqueues work — the
        copies above are CPU `memmove` on shared memory — so there is nothing
        it misses. When the launch path lands it must either share this queue
        or this method must be re-pointed at the backend's, and that is a
        contract for that chantier, not an option.
        """
        command_buffer = self._queue.commandBuffer()
        if command_buffer is None:                      # pragma: no cover
            return _ERR_ALLOC
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        error = command_buffer.error()
        return _OK if error is None else _ERR_ALLOC

    def stream_create(self, out_handle) -> int:
        """A Metal command queue, which is what a CUDA stream is here.

        Handles start at 1: zero is the seam's NULL/default stream and must
        never be produced by a create.
        """
        queue = self._device.newCommandQueue()
        if queue is None:                               # pragma: no cover
            _deref(out_handle).value = 0
            return _ERR_ALLOC
        with self._lock:
            handle = self._next_queue_handle
            self._next_queue_handle += 1
            self._queues[handle] = queue
        _deref(out_handle).value = handle
        return _OK

    def stream_destroy(self, handle) -> int:
        key = _as_int(handle)
        if not key:
            return _OK
        with self._lock:
            return _OK if self._queues.pop(key, None) is not None \
                else _ERR_UNKNOWN_POINTER

    def stream_sync(self, handle) -> int:
        key = _as_int(handle)
        if not key:
            return self.sync()
        with self._lock:
            queue = self._queues.get(key)
        if queue is None:
            return _ERR_UNKNOWN_POINTER
        command_buffer = queue.commandBuffer()
        if command_buffer is None:                      # pragma: no cover
            return _ERR_ALLOC
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        return _OK if command_buffer.error() is None else _ERR_ALLOC

    # -- events -------------------------------------------------------------
    #
    # A CUDA event is a marker placed in a stream. Metal's equivalent is an
    # `MTLSharedEvent` signalled from a command buffer, and the command
    # buffer also carries `GPUEndTime`, which is a GPU-timeline timestamp —
    # so both halves of the CUDA event API map onto real Metal machinery and
    # neither is faked. `timing=False` events skip nothing on Metal (the
    # timestamp comes free with the command buffer) but they still REFUSE to
    # be read as a stopwatch, because the engine's contract is that an
    # ordering handle is not a clock and code relying on that must break
    # here exactly as it breaks on CUDA.

    def _resolve_queue(self, handle: int):
        """The queue a stream handle names; 0 is the allocator's own."""
        if not handle:
            return self._queue
        with self._lock:
            return self._queues.get(handle)

    def event_create(self, out_handle) -> int:
        return self._event_create(out_handle, timing=True)

    def event_create_flags(self, out_handle, flags) -> int:
        # 0x02 == cudaEventDisableTiming.
        return self._event_create(out_handle, timing=not (_as_int(flags) & 0x02))

    def _event_create(self, out_handle, timing: bool) -> int:
        shared = self._device.newSharedEvent()
        if shared is None:                              # pragma: no cover
            _deref(out_handle).value = 0
            return _ERR_ALLOC
        with self._lock:
            handle = self._next_event_handle
            self._next_event_handle += 1
            self._events[handle] = {"shared": shared, "value": 0,
                                    "timing": timing, "buffer": None}
        _deref(out_handle).value = handle
        return _OK

    def event_destroy(self, handle) -> int:
        key = _as_int(handle)
        if not key:
            return _OK
        with self._lock:
            return _OK if self._events.pop(key, None) is not None \
                else _ERR_UNKNOWN_POINTER

    def event_record(self, handle, stream=None) -> int:
        """Signal the event from a command buffer on `stream`'s queue."""
        key = _as_int(handle)
        with self._lock:
            event = self._events.get(key)
        if event is None:
            return _ERR_UNKNOWN_POINTER
        queue = self._resolve_queue(_as_int(stream))
        if queue is None:
            return _ERR_UNKNOWN_POINTER
        command_buffer = queue.commandBuffer()
        if command_buffer is None:                      # pragma: no cover
            return _ERR_ALLOC
        with self._lock:
            event["value"] += 1
            value = event["value"]
        command_buffer.encodeSignalEvent_value_(event["shared"], value)
        command_buffer.commit()
        event["buffer"] = command_buffer
        return _OK

    def event_sync(self, handle) -> int:
        key = _as_int(handle)
        with self._lock:
            event = self._events.get(key)
        if event is None:
            return _ERR_UNKNOWN_POINTER
        command_buffer = event.get("buffer")
        if command_buffer is None:
            # Never recorded: nothing to wait for, exactly as on CUDA.
            return _OK
        command_buffer.waitUntilCompleted()
        return _OK if command_buffer.error() is None else _ERR_ALLOC

    def stream_wait_event(self, stream, handle, flags=None) -> int:
        """Make a queue wait on the event, without blocking the host."""
        key = _as_int(handle)
        with self._lock:
            event = self._events.get(key)
        if event is None:
            return _ERR_UNKNOWN_POINTER
        queue = self._resolve_queue(_as_int(stream))
        if queue is None:
            return _ERR_UNKNOWN_POINTER
        command_buffer = queue.commandBuffer()
        if command_buffer is None:                      # pragma: no cover
            return _ERR_ALLOC
        command_buffer.encodeWaitForEvent_value_(event["shared"],
                                                 event["value"])
        command_buffer.commit()
        return _OK

    def _event_elapsed(self, out_ms, start, end) -> int:
        """Milliseconds between two recorded events, on the GPU timeline."""
        with self._lock:
            a = self._events.get(_as_int(start))
            b = self._events.get(_as_int(end))
        if a is None or b is None:
            return _ERR_UNKNOWN_POINTER
        if not a["timing"] or not b["timing"]:
            # The whole point of an ordering handle: refuse rather than
            # return a plausible wrong number.
            return _ERR_BAD_ARGUMENT
        buf_a, buf_b = a.get("buffer"), b.get("buffer")
        if buf_a is None or buf_b is None:
            return _ERR_BAD_ARGUMENT
        # GPUEndTime is only defined once the buffer has retired.
        buf_a.waitUntilCompleted()
        buf_b.waitUntilCompleted()
        delta = (float(buf_b.GPUEndTime()) - float(buf_a.GPUEndTime())) * 1000.0
        _deref(out_ms).value = max(0.0, delta)
        return _OK

    # -- binding ------------------------------------------------------------

    def buffer_for_pointer(self, address: int):
        """The `MTLBuffer` backing an address this allocator handed out.

        How a launch path SHOULD bind our memory. The alternative — wrapping
        the raw address with
        `newBufferWithBytesNoCopy:length:options:deallocator:`, which is what
        the Triton Metal driver does — asks Metal to take over pages it
        already owns, requires page alignment to be defined at all, and was
        observed returning nil under memory pressure. The buffer already
        exists; the registry can simply hand it back.

        Returns `(MTLBuffer, offset)`, the offset always 0 today because each
        allocation is its own buffer. Returns `(None, 0)` for an address this
        allocator did not produce — the caller decides, and is not handed
        something that merely looks right.
        """
        with self._lock:
            entry = self._buffers.get(address) or self._host_buffers.get(address)
        if entry is None:
            return None, 0
        return entry[0], 0

    # -- diagnostics --------------------------------------------------------

    def live_allocation_count(self) -> int:
        """Buffers this runtime is keeping alive. Tests read it; the engine
        does not."""
        with self._lock:
            return len(self._buffers) + len(self._host_buffers)


# The entry-point name table for this implementation. It is deliberately NOT
# a row in `_GPU_BACKENDS`: that dict holds symbol tables over one C ABI and
# `test_device_backend_seam.py` pins it to exactly `{"cuda", "hip"}`, because
# a key present in one row and missing from the other is a crash on that
# hardware alone. Metal is a different API, so it gets a different table.
#
# Keys ABSENT here are absent on purpose — Metal has no honest equivalent —
# and the seam turns each into an explicit refusal naming the backend:
#   memcpy_async
#       there is no asynchronous copy to make: the copy is a CPU memmove
#       over memory both processors already share.
#   malloc_async / free_async
#       the stream-ordered allocator has no counterpart, and it is disabled
#       on CUDA here anyway.
#   device_can_access_peer / device_enable_peer_access
#       one GPU. `MTLCopyAllDevices()` returns 1, and the seam answers False
#       for peer access when the key is missing, which is the truth.
METAL_BACKEND = {
    "rt_libs": [],              # nothing is dlopen'd: this is not a C library
    "malloc": "malloc", "free": "free",
    "memcpy": "memcpy", "memset": "memset",
    "set_device": "set_device", "get_device": "get_device",
    "device_count": "device_count",
    "mem_get_info": "mem_get_info",
    "sync": "sync",
    "malloc_host": "malloc_host", "free_host": "free_host",
    "stream_create": "stream_create",
    "stream_destroy": "stream_destroy",
    "stream_sync": "stream_sync",
    "event_create": "event_create",
    "event_create_flags": "event_create_flags",
    "event_destroy": "event_destroy",
    "event_record": "event_record",
    "event_sync": "event_sync",
    "event_elapsed": "event_elapsed",
    "stream_wait_event": "stream_wait_event",
}


_RUNTIME: Optional[MetalRuntime] = None
_RUNTIME_LOCK = threading.Lock()


def runtime() -> MetalRuntime:
    """The process-wide Metal runtime. Raises if Metal is unusable."""
    global _RUNTIME
    if _RUNTIME is None:
        with _RUNTIME_LOCK:
            if _RUNTIME is None:
                _RUNTIME = MetalRuntime()
    return _RUNTIME


def metal_device_available() -> bool:
    """True when a Metal device can actually be opened.

    A probe, so it answers False instead of raising — but it opens the real
    device rather than checking for the import, because a machine with the
    bindings and no usable GPU must not be reported as ready.
    """
    try:
        runtime()
        return True
    except Exception:
        return False


def reset_runtime_for_tests() -> None:
    """Drop the cached runtime. Tests only."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        _RUNTIME = None
