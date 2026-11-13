"""Triton's compiled Metal kernels, dispatched on nbxtensor's own Metal stream.

triton-ext compiles a kernel to a `.metallib`; this module is how nbxtensor RUNS it. Each kernel's pipeline is built
here, and every launch is encoded into the stream's one open command buffer and serial compute encoder
(`metal_device.MetalRuntime.encoder_for`), its pointers bound as (buffer, offset) of the allocations this runtime
handed out (`buffer_for_pointer`). PyTorch's MPS stream and MLX run their kernels the same way: one serial encoder,
committed at a synchronisation point, never one command buffer per kernel.

What it replaces (the Mac, 2026-11-12): triton-ext's own runtime, which commits one command buffer per launch on a
queue of its own and wraps each pointer in a fresh no-copy buffer. Metal saw no dependency between two such launches
and ran them overlapped (a kernel reading its predecessor's output was wrong 4 times in 5), so every launch waited for
the whole device: 131 493 waits, 19.58 s of a 30.36 s TinyLlama run. Here a serial encoder orders consecutive kernels,
the host waits only where it reads (`flush`), and no wrap is made.

The argument ABI is triton-ext's, read from its launcher, never re-derived: pointers at indices 0..k-1, then ONE packed
block of every scalar (its layout from `triton_apple_backend.driver._compute_scalar_layout`), then the emitter's
print and assert buffers when the kernel declares them. Threads per threadgroup are `num_warps * 32`; the grid counts
threadgroups. An argument this runtime cannot bind fails by name; there is no other path.
"""
from __future__ import annotations

from nbxtensor.backend import metal_profile as _profile
import hashlib
import os
import tempfile
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple


class Pipeline(NamedTuple):
    """A kernel ready to dispatch: its compute pipeline state and the function name it was compiled under."""
    pso: Any
    name: str

    @property
    def max_total_threads_per_threadgroup(self) -> int:
        return int(self.pso.maxTotalThreadsPerThreadgroup())


_PIPELINES: Dict[Tuple[str, str], Pipeline] = {}


def pipeline(binary: bytes, name: str) -> Pipeline:
    """The pipeline of `name` in the metallib `binary`, built once per (binary, name). Metal loads a library from a
    file here: `newLibraryWithData` wants a dispatch_data object, which PyObjC does not make from bytes (an NSData in
    its place crashes the process, measured 2026-11-12), so the bytes go through a private file removed at once."""
    key = (hashlib.sha256(binary).hexdigest(), name)
    hit = _PIPELINES.get(key)
    if hit is not None:
        return hit
    from nbxtensor.backend.metal_device import runtime
    import Foundation
    device = runtime()._device
    fd, path = tempfile.mkstemp(suffix=".metallib", prefix="nbx-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(binary)
        library, error = device.newLibraryWithURL_error_(Foundation.NSURL.fileURLWithPath_(path), None)
    finally:
        os.unlink(path)
    if library is None:
        raise RuntimeError(f"Metal refused the compiled library of kernel {name!r}: {error}")
    function = library.newFunctionWithName_(name)
    if function is None:
        raise RuntimeError(f"the compiled library holds no function {name!r}: "
                           f"{list(library.functionNames())}")
    pso, error = device.newComputePipelineStateWithFunction_error_(function, None)
    if pso is None:
        raise RuntimeError(f"Metal refused a pipeline for kernel {name!r}: {error}")
    _PIPELINES[key] = Pipeline(pso, name)
    return _PIPELINES[key]


def bind_pointer(rt, encoder, index: int, address: int, kernel: str) -> Any:
    """Bind `address` at argument `index` as (buffer, offset) of the allocation holding it; returns the buffer.
    An address in no allocation of this runtime is refused by name: its length and lifetime are unknown, and binding
    anything else in its place would compute on the wrong bytes without failing."""
    buffer, offset = rt.buffer_for_pointer(int(address))
    if buffer is None:
        raise RuntimeError(
            f"ZERO FALLBACK: kernel {kernel!r}, pointer argument {index}: 0x{int(address):x} lies in no allocation of "
            f"this Metal runtime (device, host or mapped memory), so it cannot be bound")
    encoder.setBuffer_offset_atIndex_(buffer, offset, index)
    return buffer


#: Metal's buffer argument table: indices 0..30 (31 entries) for a compute function.
BUFFER_SLOTS = 31


def check_launch(pipe: Pipeline, n_buffers: int, grid, block, barrier: bool) -> None:
    """The launch-ABI checks triton-ext's own launcher makes, each refused by name before anything is encoded:
    the threads of a threadgroup within what the pipeline accepts, the bound buffers within Metal's argument table
    (scalars travel packed in one block, so a kernel of any number of scalars fits), and a grid-barrier kernel's grid
    no wider than the threadgroups the GPU holds at once (Metal never preempts a spinning threadgroup: a wider grid
    never completes)."""
    threads = int(block[0]) * int(block[1]) * int(block[2])
    limit = pipe.max_total_threads_per_threadgroup
    if threads > limit:
        raise RuntimeError(f"kernel {pipe.name!r} asks {threads} threads a threadgroup; its pipeline accepts at most "
                           f"{limit} (this configuration should have been pruned before launch)")
    if n_buffers > BUFFER_SLOTS:
        raise RuntimeError(f"kernel {pipe.name!r} binds {n_buffers} buffers; Metal's argument table holds "
                           f"{BUFFER_SLOTS}")
    if barrier:
        groups = int(grid[0]) * int(grid[1]) * int(grid[2])
        capacity = gpu_cores() * max(1, limit // threads)
        if groups > capacity:
            raise RuntimeError(f"kernel {pipe.name!r} has a barrier across threadgroups and a grid of {groups}; "
                               f"this GPU holds {capacity} of them at once ({gpu_cores()} cores), and a wider grid "
                               f"never completes")


_CORES: List[int] = []


def gpu_cores() -> int:
    """This GPU's core count, read from the IORegistry (`gpu-core-count`): Metal's API does not state it, and a
    variant of one chip differs (an M4 Pro has 16 or 20). Refused by name when the registry does not say."""
    if not _CORES:
        import re
        import subprocess
        out = subprocess.run(["/usr/sbin/ioreg", "-rc", "AGXAccelerator", "-d1"], capture_output=True, text=True,
                             timeout=10).stdout
        m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', out)
        if m is None:
            raise RuntimeError("the IORegistry states no gpu-core-count for this GPU; a grid-barrier kernel's "
                               "co-residency cannot be checked, and it is not guessed")
        _CORES.append(int(m.group(1)))
    return _CORES[0]


_ASSERT_BLOCKS: Dict[int, Tuple[Any, Any]] = {}     # id(pipeline) -> (layout, its zeroed status block)
_PENDING: Dict[int, Tuple[Any, Any]] = {}           # the blocks launched since the last check


def assert_block(pipe: Pipeline, layout) -> Any:
    """The zeroed status block an asserting kernel writes when an assert fails, one per kernel and kept (its head
    only moves on a failure, and the check that reports it zeroes it again): triton-ext's `planKernelAbi` contract.
    It is checked later, never by a wait after the launch: at the next launch if the host already sees the failure,
    and after every flush (`metal_device.MetalRuntime.flush`)."""
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.tensor import NBXTensor
    held = _ASSERT_BLOCKS.get(id(pipe))
    if held is None:
        block = NBXTensor.zeros((max(int(layout.nbytes) // 4, 1),), dtype=NBXDtype.int32, device="cuda:0")
        held = _ASSERT_BLOCKS[id(pipe)] = (layout, block)
    _PENDING[id(pipe)] = held
    return held[1]


def _words(block) -> Any:
    """The status words, read in place (shared memory: no copy, no wait)."""
    import ctypes
    import numpy as np
    raw = (ctypes.c_uint32 * block.numel()).from_address(block.data_ptr())
    return np.ctypeslib.as_array(raw)


def _raise_failed(only_landed: bool) -> None:
    from triton_apple_backend.device_assert import check
    for key, (layout, block) in list(_PENDING.items()):
        words = _words(block)
        if layout.head(words) == 0:
            if not only_landed:
                _PENDING.pop(key, None)
            continue
        seen = words.copy()
        words[:] = 0
        _PENDING.pop(key, None)
        check(layout, seen)


def raise_landed_asserts() -> None:
    """Before a launch: raise for an earlier launch whose failed assert the host can already see, without waiting."""
    if _PENDING:
        _raise_failed(only_landed=True)


def raise_failed_asserts() -> None:
    """After a flush, every launch retired: raise for the first whose assert failed."""
    if _PENDING:
        _raise_failed(only_landed=False)


def dispatch(pipe: Pipeline, stream: int, pointers: Sequence[int], scalars: Optional[bytes],
             trailing: Sequence[int], grid: Tuple[int, int, int], block: Tuple[int, int, int],
             exposes: bool = False, reads: bool = False, shared: int = 0, barrier: bool = False) -> None:
    """Encode one launch of `pipe` on `stream`: pointers, the packed scalar block, the trailing buffers (addresses
    this runtime allocated), `grid` threadgroups of `block` threads, `shared` bytes of threadgroup memory. `exposes`:
    the kernel stores its pointers' bits, so its buffers join those a kernel `reads` through (named to Metal at such
    a launch, which keeps resident only what an encoder names). `barrier`: its grid must be co-resident."""
    import Metal
    from nbxtensor.backend.metal_device import runtime
    check_launch(pipe, len(pointers) + (1 if scalars else 0) + len(trailing), grid, block, barrier)
    rt = runtime()
    _cb, encoder = rt.encoder_for(stream, kind="compute")
    if encoder is None:                                 # pragma: no cover
        raise RuntimeError(f"no Metal compute encoder for stream {stream} to launch {pipe.name!r}")
    if _profile.SESSION is not None:
        _profile.SESSION.name(_cb, pipe.name)        # the launch's row in the kernel-time table
    encoder.setComputePipelineState_(pipe.pso)
    if shared:
        encoder.setThreadgroupMemoryLength_atIndex_(int(shared), 0)
    bound: List[Any] = [bind_pointer(rt, encoder, i, a, pipe.name) for i, a in enumerate(pointers)]
    index = len(pointers)
    if scalars:
        encoder.setBytes_length_atIndex_(scalars, len(scalars), index)
        index += 1
    for address in trailing:
        bind_pointer(rt, encoder, index, address, pipe.name)
        index += 1
    if exposes:
        rt.expose(bound)
    if reads:
        named = rt.exposed_buffers()
        if named:
            encoder.useResources_count_usage_(named, len(named),
                                              Metal.MTLResourceUsageRead | Metal.MTLResourceUsageWrite)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSizeMake(*map(int, grid)),
                                                        Metal.MTLSizeMake(*map(int, block)))
    rt.note_dispatch(stream)


class BoundLaunch:
    """One launch resolved once — a frozen plan's, whose addresses never move: its pipeline, each pointer's (buffer,
    offset), its packed scalars, its grid. `run(stream)` only encodes (stage A of the Metal replay: the live path's
    lookups, checks and packing cost ~26 us a launch, measured 2026-11-13)."""

    __slots__ = ("pipe", "buffers", "packed", "grid", "block", "shared", "exposes", "reads")

    def __init__(self, pipe: Pipeline, buffers, packed: Optional[bytes], grid, block, shared: int, exposes: bool,
                 reads: bool):
        import Metal
        self.pipe, self.buffers, self.packed, self.shared = pipe, tuple(buffers), packed, int(shared)
        self.grid, self.block = Metal.MTLSizeMake(*map(int, grid)), Metal.MTLSizeMake(*map(int, block))
        self.exposes, self.reads = bool(exposes), bool(reads)

    def run(self, stream: int) -> None:
        import Metal
        from nbxtensor.backend.metal_device import runtime
        rt = runtime()
        cb, encoder = rt.encoder_for(stream, kind="compute")
        if encoder is None:                             # pragma: no cover
            raise RuntimeError(f"no Metal compute encoder for stream {stream} to launch {self.pipe.name!r}")
        if _profile.SESSION is not None:
            _profile.SESSION.name(cb, self.pipe.name)
        encoder.setComputePipelineState_(self.pipe.pso)
        if self.shared:
            encoder.setThreadgroupMemoryLength_atIndex_(self.shared, 0)
        for i, (buffer, offset) in enumerate(self.buffers):
            encoder.setBuffer_offset_atIndex_(buffer, offset, i)
        if self.packed:
            encoder.setBytes_length_atIndex_(self.packed, len(self.packed), len(self.buffers))
        if self.exposes:
            rt.expose([b for b, _o in self.buffers])
        if self.reads:
            named = rt.exposed_buffers()
            if named:
                encoder.useResources_count_usage_(named, len(named),
                                                  Metal.MTLResourceUsageRead | Metal.MTLResourceUsageWrite)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(self.grid, self.block)
        rt.note_dispatch(stream)


def bind(pipe: Pipeline, pointers: Sequence[int], scalars: Optional[bytes], grid, block, exposes: bool = False,
         reads: bool = False, shared: int = 0, barrier: bool = False) -> BoundLaunch:
    """A launch with no trailing buffer resolved once (`BoundLaunch`): the live path's checks made here, each pointer
    found in its allocation here, refused by name as `bind_pointer` refuses."""
    from nbxtensor.backend.metal_device import runtime
    check_launch(pipe, len(pointers) + (1 if scalars else 0), grid, block, barrier)
    rt = runtime()
    resolved = []
    for i, address in enumerate(pointers):
        buffer, offset = rt.buffer_for_pointer(int(address))
        if buffer is None:
            raise RuntimeError(
                f"ZERO FALLBACK: kernel {pipe.name!r}, pointer argument {i}: 0x{int(address):x} lies in no allocation "
                f"of this Metal runtime (device, host or mapped memory), so it cannot be bound")
        resolved.append((buffer, offset))
    return BoundLaunch(pipe, resolved, scalars, grid, block, shared, exposes, reads)
