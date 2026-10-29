# nbxtensor — adding a backend

nbxtensor runs Triton kernels. Triton compiles them; nbxtensor launches them, allocates their memory and decides their
tiles. A backend is what nbxtensor needs from a device to do that. This page lists every piece a backend provides, with
CUDA and Metal — the two backends that exist — as the worked examples, each taken from the code as it is.

A backend is five things:

1. a **launch driver** — four calls (`backend/launcher.py`, class `Driver`);
2. the **memory primitives** — one interface (`memory/device_memory.py`, class `DeviceMemory`);
3. its **capability rows** — what its kernels can and cannot do (`backend/capabilities.py`);
4. its **device files** — the facts the kernels are tiled by, one file per device (`backend/devices/<vendor>/`);
5. its **certified kernel settings** — the autotune's results proven on the device (`ops/autotune_certified.py`).

Nothing else in the library names a backend. A piece that is not there is refused by name when it is needed: a backend
is never assumed to behave like CUDA.

## 1. The launch driver

`backend/launcher.py` compiles a `@triton.jit` function once (Triton does the compiling) and launches it many times.
What it asks of a backend is a subclass of `Driver`:

| call | what it does | CUDA (`CudaDriver`, `backend/launcher.py`) | Metal (`TritonExtDriver`, `backend/triton_ext_driver.py`) |
|---|---|---|---|
| `load(binary, name, shared)` | turn the compiled artifact into something launchable | `cuModuleLoadData` on the cubin, `cuModuleGetFunction` | the metallib loaded into a pipeline state, as triton-ext's own launcher does |
| `launch(function, grid, block, shared, stream, params, names=, types=, trailing=)` | enqueue one dispatch; `params` are integer device addresses and Python scalars | `cuLaunchKernel` | encode a compute command with the arguments bound as triton-ext binds them |
| `target()` | the Triton `GPUTarget` the kernels are compiled for | `GPUTarget("cuda", <capability>, 32)` from the driver | the selected Metal backend's target (`backend/metal_backend.metal_target`) |
| `max_shared_memory_per_block()` | the device's per-block shared-memory ceiling, in bytes (asked of the driver; a backend that cannot ask raises, and the autotune then prunes nothing) | `cuDeviceGetAttribute` | `maxThreadgroupMemoryLength` |

Three class attributes carry what the launcher must not assume; each default is CUDA's answer:

- `artifact_kind` — the key of the compiled artifact (`"cubin"`; Metal `"msl"`; ROCm `"hsaco"`);
- `wants_scratch_params` — whether the launch ABI has Triton's trailing scratch pointers (Metal: no);
- `wants_extents` — whether a pointer is bound by wrapping memory and needs the byte extent of its tensor (Metal: yes).

Two methods have defaults a backend may override: `trailing_buffers(metadata)` (buffers the compiled kernel declares
after the caller's arguments — Metal's device-assert status area) and `block_for(metadata)` (threads per block;
`32 * num_warps` on CUDA, the emitted kernel's own size on Metal).

The contract a driver must meet is written down as two protocols, `CompiledKernel` and `LauncherDriver`, in
`backend/launcher_contract.py`: integers for device addresses, opaque integers for streams and events, no tensor object
and no torch at the boundary.

## 2. The memory primitives

Every allocation, copy and memory reading goes through one object, `DeviceMemory` (`memory/device_memory.py`); the
allocator, the arenas and the pools are written against it alone.

| primitive | answers |
|---|---|
| `malloc(nbytes)` / `free(ptr)` | `(rc, device pointer)` / `rc` — a backend may release when its queue is done |
| `memset(ptr, value, nbytes)`, `memcpy(dst, src, nbytes, kind)` | `rc`; `kind` 0 host to host, 1 host to device, 2 device to host, 3 device to device |
| `mem_get_info()` | `(rc, free bytes, total bytes)` of the current device |
| `host_alloc(nbytes)` / `host_free(ptr)` | page-locked (or shared) host memory |
| `set_device(i)`, `get_device()`, `device_count()` | the current device |
| `budget()` | `(rc, bytes the device grants this process at most)` |
| `map_host(ptr, nbytes)` / `unmap_host(start)` | unified memory: a device-visible address of host bytes |
| `supports(primitive)` | whether this backend offers an optional primitive (`host_alloc`, `budget`, `map_host`, ...) |

Two implementations exist:

- `AbiMemory` serves every backend that exposes the C ABI of a CUDA-style runtime — CUDA, and HIP through the same
  ABI — from a table of the runtime's function names;
- `MetalMemory` (`memory/metal.py`) implements it over the Metal runtime: `budget()` is the device's
  `recommendedMaxWorkingSetSize`, `mem_get_info` the budget less what this process holds, `map_host` binds a no-copy
  buffer.

`backend/runtime.device_memory()` returns the one in force.

## 3. Capability rows

What a backend's kernels can do is declared once per backend, in `backend/capabilities.py`, never inferred from a
device name; each row is read by a function the kernels call:

| row | CUDA | HIP | Metal | read by |
|---|---|---|---|---|
| a kernel stops on `tl.device_assert` | yes | yes | no | `backend_traps_on_device_assert()` |
| a kernel may dereference an address it loaded from a tensor | yes | yes | no (yes when the selected launcher declares it: triton_ext, through its pinned scope) | `backend_loads_pointers_from_memory()` |
| the flash attention's smallest tile | 16 | 16 | 32 | `fa_min_tile()` |
| the largest reduced extent one program stages | 1 << 20 | 1 << 20 | 1024 | `reduce_tile_max()`, `reduced_extent_max(num_warps)` |
| reductions run per thread | no | no | yes | `reduced_extent_max(num_warps)` |
| threads per warp | 32 | 64 | 32 | `reduced_extent_max` |
| the kernels compute float64 | yes | yes | no | `backend_has_fp64()`, `device_facts.kernels_carry_fp64()` |

A new backend adds its column to these tables. A capability that belongs to the launcher selected rather than to the
silicon is stated by the application that selected it, through `declare_backend_capability(backend, **rows)`, which the
questions read above the tables. A row a backend does not declare is refused by name when a kernel asks it.

## 4. Device files

The facts the kernels are tiled by — shared memory and registers per block, warp size, tiles per op, the matrix unit,
dispatch limits, the autotune's screen tolerances and buckets, and the identification a device is found by
(`compute_capability`, `compute_capability_matches`, `architecture`) — are one YAML file per device,
`backend/devices/<vendor>/<name>.yml`, read by `backend/device_files.py`. The keys a file must hold are the device-key
census, `docs/nbxtensor/device_keys.txt`. Each value carries its source in a comment: a vendor's specification, or a
measurement on the device (never a neighbouring part's value).

What the driver can answer is asked of it at start instead (`max_shared_memory_per_block`, Metal's
`maxThreadgroupMemoryLength`, CUDA's device attributes); the file holds what no driver call answers.

## 5. Certified kernel settings

The autotune measures each kernel's candidate settings on the device and keeps the fastest one that the fp64 oracle
accepts (`ops/autotune_certified.py`, `ops/screen_oracle.py`). Settings proven once are kept in a store
(`autotune_certified.directory()`: the user's cache by default, `set_store(path)` for another) per vendor and device
name, with the code generator they were proven under. A new backend starts with an empty store: its first runs sweep,
and what they prove is kept.

## What exists, what is written, what is not supported

See the README's **Hardware** table: the backends proven on a device (with the device and the date), those written but
not proven for lack of the hardware, and the devices no Triton backend covers.
