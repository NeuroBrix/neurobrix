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
| `nbxtensor.dtype` | yes | the dtype table (`dtypes.yml`) and its reader, host bfloat16 conversions, the promotion rule |
| `nbxtensor.shape` | yes | numel, strides, contiguity, the view rule, broadcasting, slice bounds |
| `nbxtensor.random` | yes | seeded random streams and draws |
| `Tensor`, `memory`, `backend`, `ops` | moving | the tensor class, the allocators, the device backends, the Triton kernels |

## Installation

```
pip install -e nbxtensor          # from the NeuroBrix repository; numpy, pyyaml and triton are its dependencies
```

Python 3.10 or later. The device areas need Triton for the target backend (CUDA, or triton-ext on Apple Metal).
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

## Backends

CUDA (NVIDIA), Apple Metal through triton-ext, and the CPU. The backend modules are moving into the library; until
then the device examples live in NeuroBrix's tests.

## Exchange with other libraries: DLPack

nbxtensor speaks DLPack and nothing else: `Tensor.__dlpack__` / `__dlpack_device__` and `nbxtensor.from_dlpack(x)`
for any producer. The other library imports itself (`torch.from_dlpack(t)`); nbxtensor never imports torch. Planned
with the tensor class (API.md §4).

## Limits

- Inference only: no autograd.
- Slicing has step 1 (`narrow`); a strided slice is refused by name.
- A dtype absent from the table is refused; add it to `dtypes.yml` with its source.

## License

Apache 2.0.
