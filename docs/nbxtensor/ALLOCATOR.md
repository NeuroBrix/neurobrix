# nbxtensor memory — one account, one device-memory object per backend (M2, Dell, 2026-10-29)

## Today
`memory/allocator.DeviceAllocator` is the account: live bytes per device, peaks, the pool and its parked blocks,
the deferred-free queues, the imposed rung, the OOM report. It reaches the device through ~40 call sites of one
shape, `getattr(rt, backend["<name>"])(<ctypes args>)`, where `rt` is `backend.runtime._gpu_runtime()` (a ctypes
CDLL for CUDA/HIP, `MetalRuntime` for Metal) and `backend` a table of symbol names (`_GPU_BACKENDS[...]`, or
`METAL_BACKEND`). Metal emulates the C ABI behind those names, and keeps a SECOND account of its own
(`MetalRuntime._live_bytes`, the working-set budget).

## Now (this commit)
`memory/device_memory.py` defines the contract `DeviceMemory` and `AbiMemory`, its implementation over a C-ABI runtime
and its symbol table (CUDA, HIP, and Metal's emulation today). `backend/runtime.device_memory()` builds it per call (never
cached: a substituted runtime is read at once). Every MEMORY call of the account goes through it: device selection and
count, malloc/free (the pool's segments included), mem_get_info (one device, the most free, every total, the OOM report),
memset, memcpy, pinned host memory, the unified-memory mapping. Return codes are the runtime's, checked by the account as
before; messages unchanged. Peer copies, streams and events stay on the backend table (backend/, not memory).

## The interceptor and the slabs (M2, after the interface)
The account's five public device operations — `malloc_cuda`, `free_cuda`, `memcpy`, `memset_cuda`, `sync_device` — each
go through the `AllocatorInterceptor` in force (`DeviceAllocator.install_interceptor`), handed the DIRECT operation it
stands in front of (`malloc_direct`, `free_direct`, `memcpy_direct`, `memset_direct`, `sync_direct`), or straight to the
direct operation when none is installed. One interceptor at a time; a second is refused by name. Nothing replaces the
allocator's methods any more: the census shadow substitutes the DIRECT operations (so an interceptor still stands in front
of them, as in a real run), and the frozen-plan replayer installs its interceptor.

`memory/slab.py` holds the replay's slab allocators (`SlabAllocator`: first-fit over one device block; `StepSlab`: a
pure-bump slab recycled at every decode step), the registry of slabs with outstanding carvings (`ACTIVE_SLABS`), and
`SlabInterceptor`: while recording, a pooled allocation is carved from the slab of the requested device (a refusal goes to
`slab_refused`, the owner's decision); between recordings, from the recycled step slab when one is in force; else the
allocator serves it and the watermark is measured per device. The engine's replayer (`triton/replay.py`) is a
`SlabInterceptor` that adds what is its own: the copies, sets and synchronisations a plan records.

## Target
```
memory/allocator.py      DeviceAllocator — THE account (unchanged API)
memory/device_memory.py  DeviceMemory   — the primitives of one backend, no account:
    malloc(nbytes) -> ptr            raises DeviceOOM (the account turns it into its report)
    free(ptr)                        Metal: released when the GPU is done (the account sees it freed at once)
    mem_get_info() -> (free, total)  Metal: budget minus our live (currentAllocatedSize is cumulative)
    budget() -> bytes                Metal: recommendedMaxWorkingSetSize; CUDA: total
    memcpy(dst, src, nbytes, kind)   kind in {h2d, d2h, d2d}; memcpy_async(..., stream)
    memset(ptr, value, nbytes)
    host_alloc(nbytes) / host_free(ptr)          pinned (CUDA) / shared (Metal)
    map_host(ptr, nbytes) -> dev_ptr / unmap_host(dev_ptr)   unified memory only
    set_device(i) / get_device() / device_count()
memory/cuda.py           CudaMemory(DeviceMemory) — today's ctypes calls, return codes checked HERE
memory/metal.py          MetalMemory(DeviceMemory) — the Mac moves MetalRuntime's memory part here
```
`backend/runtime.device_memory()` returns the backend's instance (cached, teardown-safe like `_metal_backend_table`).
Streams, events and synchronisation stay in `backend/` (not memory).

## Rules
- One account: a backend object never counts bytes; the account refuses past `budget()`.
- Behaviour unchanged: the same calls in the same order, the same error codes surfaced by name; the 5-model shas,
  the coverage and the suites are the proof, as for every M1/M2 commit.
- The allocation strategies sit on the account: the pool inside it, the replay slab in front of it through the one
  interceptor; the weight arena (a block outside the pool, `pooled=False`) is the next to move.
