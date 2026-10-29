# nbxtensor

An inference-first tensor library, torch-free.

PyTorch is a general framework built for research and training in eager mode: dynamic graphs, autograd, a generic
dispatcher, a caching allocator that discovers memory as the program runs. Inference is a narrower problem, and a
better-informed one: the graph and the shapes are known before the first run. nbxtensor is built for that case.

- **Memory is planned, not discovered.** Allocators per device behind one interface, and a static planner that
  assigns every buffer before the first run.
- **Every kernel declares its cost.** Outputs, workspace and scratch with their lifetimes, bytes moved, flops. A
  planner prices a model from these declarations instead of guessing what the allocator will do.
- **Capture and replay are first-class.** A repeated sequence of launches is recorded, verified byte-equal and
  replayed.
- **One table of facts.** One dtype table, one view rule, one conversion per pair of dtypes, one capability table per
  backend — declared, never guessed.
- **Backends:** NVIDIA CUDA, Apple Metal (through triton-ext), CPU.
- **No torch, ever.** nbxtensor never imports torch, not even optionally. Exchange with any tensor library goes
  through the DLPack protocol (`__dlpack__`, `from_dlpack`), so a torch user writes `torch.from_dlpack(t)` on their
  side.

nbxtensor is extracted from NeuroBrix, the universal inference engine whose Triton engine is its first client. It
lives in `nbxtensor/` of the NeuroBrix repository until the extraction is complete, then in its own repository.

## Status

The library is being moved out of the engine, area by area (`docs/nbxtensor/MIGRATION.md` in the NeuroBrix
repository). This README documents what is in the library TODAY; every example below is executed by
`nbxtensor/tests/test_readme.py`, so an example that stops working fails a test. The target API is
`docs/nbxtensor/API.md`.

| area | in the library | what it holds |
|---|---|---|
| `nbxtensor.dtype` | yes | the dtype table (`dtypes.yml`) and its reader, the enum's tables derived from it, host bfloat16 conversions, the promotion rules |
| `nbxtensor.shape` | yes | numel, strides, contiguity, the view rule, broadcasting, slice bounds |
| `nbxtensor.random` | yes | seeded random streams and draws |
| `nbxtensor.tensor`, `memory`, `backend`, `ops`, `quant` | moved in, public names being fixed (M2) | the tensor class, the allocators, the device backends, the Triton kernels and their cost declarations |
| `nbxtensor.host` | yes | the services still passed by an application while the boundary is cleaned (below; API.md section 11) |

## Installation

nbxtensor never needs torch: it is not a dependency, not even an optional one, and the library and its kernels run in
an environment where torch is not installed (proven on Metal, 2026-10-29: `tools/torch_free_proof.py` records every
attempt to import torch, caught ones included, and recorded none). Python 3.10 or later; its dependencies are numpy,
pyyaml and Triton.

```
pip install -e nbxtensor          # from the NeuroBrix repository
```

**Linux, NVIDIA (CUDA).** Triton comes from PyPI with the install above.

**macOS, Apple silicon (Metal).** PyPI publishes no Triton for macOS, so Triton and its Metal backend are built from
source, then nbxtensor is installed into that environment:

1. Triton (github.com/triton-lang/triton) built with `TRITON_EXT_ENABLED=1`, which lets out-of-tree backends plug in;
2. the AppleGPU backend of triton-ext (github.com/triton-lang/triton-ext, `backend/AppleGPU`), installed into the same
   environment;
3. `pyobjc-framework-Metal` (the library opens the Metal device through it);
4. Xcode's Metal toolchain (`xcrun metal` must find the real compiler, not Xcode's placeholder: `TOOLCHAINS=Metal`).

`nbxtensor.dtype` and `nbxtensor.shape` need neither a device nor Triton.

## Concepts

**Plain data at the edges.** A dtype is named by its table name (`"float16"`, `"bfloat16"` …), a shape is a tuple of
ints, strides are a tuple of ints counted in ELEMENTS (never bytes), a device is a kind and an index. Nothing in the
API takes a duck-typed object or keeps state in a module global: the same API is meant to be implemented later by a
Rust core.

**One table.** Every width, safetensors code, launch-key spelling and kind of a dtype is one entry of `dtypes.yml`.
A name the table does not carry is refused by name; no width is ever guessed.

**Zero fallback.** A missing fact is an error that names it, never a default.

## Examples

### Dtype widths and kinds

```python
from nbxtensor.dtype import itemsize as I

assert I.itemsize("bfloat16") == 2
assert I.itemsize("complex64") == 8
assert I.kind("bfloat16") == "float" and I.kind("uint8") == "uint"
assert I.from_safetensors("BF16") == "bfloat16"
assert I.from_autotune_key("fp16") == "float16"
# a store without float64 holds float32 in its place
assert I.storage_dtype("float64", stores_fp64=False) == "float32"
# a packed int4 weight with float16 scales and minima per group of 128
assert I.representation_bytes("int4-g128-asym", 256) == 128 + 2 * (2 + 2)
try:
    I.itemsize("fp8_unknown")
except ValueError as e:
    assert "unknown dtype" in str(e)
```

### bfloat16 on the host

numpy has no bfloat16: a bfloat16 array on the host is its 16-bit patterns (`uint16`).

```python
import numpy as np
from nbxtensor.dtype import convert

x = np.array([1.0, 3.14159, -2.5e-3, np.nan], dtype=np.float32)
bits = convert.f32_to_bf16_bits(x)              # round to nearest, ties to even
assert bits.dtype == np.uint16 and bits[3] == 0x7FC0     # every NaN becomes the quiet NaN, as torch does
back = convert.bf16_bits_to_f32(bits)           # exact widening
assert back[0] == 1.0 and abs(back[1] - 3.14159) < 0.02
half = convert.bf16_bits_to_f16(bits)           # bfloat16 -> float16, through float32
assert half.dtype == np.float16

# a caller splitting a large array across workers writes each span into one output
out = np.empty(x.size, dtype=np.float16)
convert.bf16_bits_to_f16(bits[:2], out=out[:2])
convert.bf16_bits_to_f16(bits[2:], out=out[2:])
assert np.array_equal(out[:3], half[:3])
```

### Type promotion

```python
from nbxtensor.dtype.promotion import promoted, float_math_input

assert promoted(["float16", "float32"]) == "float32"      # the widest float, past two bytes
assert promoted(["float16", "bfloat16"]) is None          # two half dtypes: the caller's precision policy decides
assert float_math_input("int64") == "float32"             # rsqrt of an integer reads it as float32

from nbxtensor.dtype.promotion import binary_result
assert binary_result("int8", "uint8") == "int16"          # the Array API standard: wide enough for both
assert binary_result("complex64", "float64") == "complex128"   # a complex keeps the wider float's precision
assert binary_result("int64", "float16") == "float16"     # across kinds (left open by the standard): PyTorch's rule
assert binary_result("bfloat16", "float16") == "float32"  # two half formats meet in float32, whatever the order
```

### Precision by policy

The dtype an op computes in is decided by a `Policy`: plain, immutable data handed to the calls that use it (no
context manager, no global switch). Its op classes are PyTorch's autocast tables (`nbxtensor/dtype/policy.yml`):
some ops compute in float32 whatever the policy, some in its compute dtype, some in the widest of their inputs; an op
in no class keeps its inputs' dtypes.

```python
from nbxtensor.dtype.policy import get_policy

p = get_policy("params=float32,compute=float16,output=float32")      # the spelling of JAX's jmp
assert p.compute_dtype("mm", ["float32", "float32"]) == "float16"     # a matrix product: the compute dtype
assert p.compute_dtype("softmax", ["float16"]) == "float32"           # a softmax: float32, whatever the policy
assert p.compute_dtype("addcmul", ["float16", "float32", "float16"]) == "float32"   # the widest input
assert p.fill_value("masked_fill", -3.39e38, "float16") == -65504.0   # a bfloat16 mask sentinel fits float16
assert p.replace(compute="bfloat16").compute == "bfloat16"            # a policy never changes: a new one is made
```

### Array API: type promotion and Python scalars

Type promotion follows the [Array API standard](https://data-apis.org/array-api/draft/API_specification/type_promotion.html),
and a Python scalar is weak (it takes the array's dtype), as the standard and NumPy's NEP 50 say. Measured by
`nbxtensor/tests/test_array_api_promotion.py`: the standard's promotion tables, 120 of 120 pairs of the dtypes the
library holds, in both operand orders, and the order-independence of the result over three operands; its scalar rules.

What the standard leaves open, decided once:

| case | nbxtensor | note |
|---|---|---|
| arrays of two kinds (bool, unsigned, signed, float, complex) | the higher kind, its own width | as PyTorch |
| float16 with bfloat16 (the standard defines neither) | float32, whatever the order | as PyTorch |
| a signed integer with uint64 | refused by name | no signed integer holds it |
| a Python float with an integer or bool array | float64 | NEP 50; PyTorch gives float32 |
| a Python complex with an integer array; with a half float array | complex128; complex64 | NEP 50; the table has no complex32 |
| a Python int outside an integer array's range | refused by name | never a silent wrap |

The gap: nbxtensor exposes no array namespace (`__array_namespace__`) yet, so the standard's own test suite
(`array-api-tests`) cannot run against it; the conformance above is measured on the standard's tables.

```python
from nbxtensor.dtype.promotion import binary_result, scalar_result

assert binary_result("uint8", "int8") == "int16"                # wide enough for both
assert scalar_result("float16", 0.5) == "float16"               # a Python scalar is weak
assert scalar_result("int32", 1.5) == "float64"                 # left open by the standard: NEP 50
```

### Shapes, strides and views

```python
from nbxtensor import shape as S

assert S.numel((2, 3, 4)) == 24
assert S.contiguous_strides((2, 3, 4)) == (12, 4, 1)

# the view rule (ATen's computeStride): merging contiguous dims is a view ...
assert S.view_strides((2, 3, 4), (12, 4, 1), (6, 4)) == [4, 1]
# ... flattening a transposed layout is not: None means "copy first"
assert S.view_strides((4, 3), (1, 4), (12,)) is None

# a dim of extent 1 may carry any stride: this decode-step head block is contiguous
assert S.is_contiguous((1, 32, 64, 1), (2048, 64, 1, 64))

assert S.broadcast_shapes((5, 3, 1), (4,)) == (5, 3, 4)
assert S.broadcast_strides((3, 1), (1, 1), (2, 3, 4)) == ((2, 3, 4), (0, 1, 0))   # expand without a copy
assert S.slice_bounds(50, 80, 60) == (50, 60)                                       # clamped, as numpy does
```

### The dtype enum and its tables

```python
from nbxtensor.dtype.nbx_dtype import NBXDtype, dtype_size, parse_dtype

assert NBXDtype.bfloat16.value == 1                        # values are frozen: new members are appended
assert dtype_size(NBXDtype.complex64) == 8                 # read from the one table, never written twice
assert parse_dtype("torch.float16") is NBXDtype.float16    # a torch spelling, read as text
assert parse_dtype("half") is NBXDtype.float16             # an alias the table carries
```

## What the library reads by itself, and what you may set

The library measures the facts of the machine it runs on and has a default for every setting, so `import nbxtensor`
works with no application around it:

- **device descriptions**: what the kernels are tiled by — shared memory, registers, warp size, tiles, the matrix
  unit, dispatch limits — and how a device is recognised, one file per device in the package
  (`nbxtensor/backend/devices/<vendor>/<name>.yml`, read by `nbxtensor.backend.device_files`);
- **measured facts**: the host's free memory (`nbxtensor.backend.host_memory.memory_state`), whether the device
  shares the host's memory, whether its kernels compute float64 (`nbxtensor.backend.device_facts`);
- **settings with a default**, set with plain data: the tiling table the memory splits are cut by
  (`nbxtensor.ops.tiling_policy.set_tiling`, default: the library's `ops/tiling.yml`), the directory of proven kernel
  settings (`nbxtensor.ops.autotune_settings.set_store`, a path or a callable returning one, default: your cache, `~/.cache/nbxtensor/certified/`,
  filled by the autotune at first use), the memory limit a process holds to (`nbxtensor.memory.set_limit`, default:
  the physical card). `None` restores a default.

```python
from pathlib import Path
from nbxtensor.backend.device_names import arch_string
from nbxtensor.backend.host_memory import memory_state
from nbxtensor.ops import autotune_settings, tiling_policy
from nbxtensor.backend import device_files

m4 = device_files.read("apple", "apple_m4_pro")             # one device's description, from the package itself
assert m4["compute_capability"] == "apple-m4-pro" and m4["memory"]["max_shared_memory_per_block"] == 32768

assert arch_string("Apple M4 Pro") == "apple-m4-pro"        # a device's key: the library's own rule
assert memory_state().source                                # the host reading names where it came from

autotune_settings.set_store("/tmp/my-proven-settings")      # your own store of proven kernel settings
assert autotune_settings.directory() == Path("/tmp/my-proven-settings")
autotune_settings.set_store(None)                           # back to the default in your cache

table = tiling_policy.tiling_table()                        # the split sizes in force (a copy)
tiling_policy.set_tiling(table)                             # a whole table, as tiling.yml writes it
tiling_policy.set_tiling(None)                              # back to the library's file
```

A memory limit is a setting, not a service: `nbxtensor.memory.set_limit` holds the process to a card of that many
bytes (the allocator refuses what such a card refuses, and its pool parks a fraction of the limit); absent, the
physical card is the limit.

```python
from nbxtensor import memory

memory.set_limit(8 << 30)        # behave as a card of 8 GiB on every device
memory.set_limit(None)           # the physical card again
```

Some services are still passed through `nbxtensor.host` while the boundary is cleaned before 0.0.1 (the device
description, the operand decisions of the matrix unit and attention, and where a backend build may live).
`docs/nbxtensor/API.md` section 11 lists each one, its fate (generic in the library, or back in the application) and
its state.

An application may also watch every kernel launch without running it — a dry run, to list the kernel keys a model
forms without a device: `nbxtensor.ops.launch_observer.set_launch_observer` installs an observer (`active()`,
`record(tuned, key)`, `shadow_run(...)`); with none installed, every launch runs.

```python
from nbxtensor.ops import launch_observer

assert launch_observer.launch_observer().active() is False     # no observer: nothing runs dry
```

## The CPU: every core

Host work over a large array runs on a thread pool, one contiguous piece per thread, each piece a numpy call (numpy
releases the GIL in its loops). The pool's size is measured on the machine (Apple silicon: its performance cores) and
can be set; under a threshold measured for each kind of work, the work runs inline.

```python
import numpy as np
from nbxtensor.backend.cpu import pool

assert pool.threads() == pool.measured_threads() >= 1        # this machine's compute cores, measured
x = np.linspace(-3, 3, 1 << 20, dtype=np.float32)
out = np.empty_like(x)
probe = lambda n: (lambda a, b: np.exp(x[a:b], out=out[a:b]))   # how the threshold for "exp" is measured
pool.parallel_for(x.size, lambda a, b: np.exp(x[a:b], out=out[a:b]), "exp", probe)
assert out.tobytes() == np.exp(x).tobytes()                   # the same bytes as one numpy call
pool.set_threads(2)                                           # an application's own size
assert pool.threads() == 2
pool.set_threads(None)                                        # back to the measurement
```

## Hardware

| hardware | backend | state |
|---|---|---|
| NVIDIA Tesla V100 (Volta, sm_70) | CUDA (`CudaDriver`, `AbiMemory`) | **proven**: the library's tests and the reference models' outputs identical to their judged sha, on a 4 x V100 rack, 2026-10-29 |
| Apple M4 Pro (Apple silicon, 24 GB unified) | Metal through triton-ext (`TritonExtDriver`, `MetalMemory`) | **proven**: the library's tests, the reference models' outputs identical to their judged sha, in an environment where torch is not installed, 2026-10-29 |
| NVIDIA Ampere, Hopper | CUDA | **written, not proven here**: device files exist (`backend/devices/nvidia/`), no such card on our machines |
| Apple M1, M2, M3, M5 families and the other M4 chips | Metal | **written, not proven here**: device files exist (`backend/devices/apple/`); only the M4 Pro is measured |
| AMD Instinct (CDNA, CDNA2, CDNA3) | HIP through the CUDA-style C ABI (`AbiMemory`) | **interface ready, not tested for lack of hardware**: memory primitives and device files exist; the HIP launch driver is still to write (`docs/nbxtensor/BACKENDS.md` section 6) |
| Intel GPUs (XPU) | Level Zero (not the CUDA ABI) | **interface ready, not tested for lack of hardware**: the backend contract is documented; its `DeviceMemory` over Level Zero and its launch driver are to write (BACKENDS.md section 6) |
| a CPU | none yet | **not supported** for the kernels: host work (loading, conversions) runs with numpy on every core (`backend.cpu.pool`); a CPU kernel backend is not part of 0.0.1 |
| hardware with no Triton backend (NPUs, phones) | none | **not supported**: covering it needs a Triton backend for it and the five pieces of `docs/nbxtensor/BACKENDS.md` |

## Backends

CUDA (NVIDIA) and Apple Metal through triton-ext; what a new backend provides is `docs/nbxtensor/BACKENDS.md`, and
`tests/nbxtensor_backend_conformance` is what it must pass. The backend in force is public API:
`nbxtensor.backend.runtime.gpu_backend()` names it ("cuda", "hip", "metal"), `gpu_runtime()` is its runtime,
`backend_table()` its entry points, `device_memory()` its memory primitives, and `set_device(tensor)` binds the
driver to a tensor's device before a launch.

## Exchange with other libraries: DLPack

nbxtensor speaks DLPack and nothing else: `Tensor.__dlpack__` / `__dlpack_device__` and `nbxtensor.from_dlpack(x)`
for any producer. The other library imports itself (`torch.from_dlpack(t)`); nbxtensor never imports torch. Planned
with the tensor class (API.md §4).

## Limits

- Inference only: no autograd.
- Slicing has step 1 (`narrow`); a strided slice is refused by name.
- A dtype absent from the table is refused; add it to `dtypes.yml` with its source.

## Versions

The first publication is **0.0.1** (this version); each fix published after it increments the number (0.0.2, 0.0.3,
…), with no alpha or beta label. The rewrite of the core in Rust will be **2.0.0**. `nbxtensor.__version__` is the
installed version.

## License

Apache 2.0.
