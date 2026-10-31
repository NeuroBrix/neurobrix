# nbxtensor — public API, version 1 (proposal)

Written by the Mac agent on 28/10/2026 for the supervisor's review (order of 21:00, point 3a), on branch
`nbxtensor` 76a18217. It fixes the names DESIGN.md §4 leaves open. Every entry is drawn from code that
exists today, cited `file:line` on that commit, and judged against two rules:

- **Rule 10, a documented public API.** An exported name is API. Anything not listed here stays `_private`.
  Each export ships with its docstring and its test, and the README's examples run as tests.
- **Rule 11, an API a Rust core can implement.** Only plain data crosses the boundary: dtypes, shapes,
  strides, device ids, byte counts, and op descriptions given as data. No duck-typed inputs, no
  overloaded positional arguments, no dynamic attributes, and no state hidden in module globals. A
  v2 Rust core (PyO3) must be able to sit behind the same Python names without a behaviour change.

Section 9 lists what today's code does that this API does not keep, and why.

## 1. Plain data at the edges

| concept | Python v1 | Rust v2 | today |
|---|---|---|---|
| dtype | `nbxtensor.DType`, an `IntEnum` whose values never change | `#[repr(u8)] enum DType` | `NBXDtype` (kernels/nbx_tensor.py:174) |
| shape | `tuple[int, ...]` | `&[i64]` | tuples everywhere |
| strides | `tuple[int, ...]`, in **elements** | `&[i64]` | `_strides` slot (nbx_tensor.py:3227) |
| device | `nbxtensor.Device(kind: str, index: int)`, frozen; kind is `"cuda"`, `"hip"`, `"metal"` or `"cpu"` | `struct Device { kind: DeviceKind, index: u32 }` | strings `"cuda:0"`, plus `.device` returning the tensor itself (nbx_tensor.py:3338) |
| bytes, offsets | `int` | `u64` / `i64` | ints |
| pointer | `int` (device address) | `u64` | `data_ptr()` |
| stream, event | `int` handles | `u64` | `DeviceAllocator.create_stream / create_event` |
| op description | `dict` of plain values (op name, attribute names to ints, floats, bools, tuples, DType) | `struct OpDesc` | graph.json `attributes`, resolved by the engine |

The library never reads a NeuroBrix profile, a family file or a model name (rule 3). Device facts reach
it as plain values: a `Capabilities` record, a matrix-unit description `dict`.

## 2. `nbxtensor.dtype` — the one table (M2, Mac)

The dense entries of `config/dtypes.yml` move into the library as its data file. Its two readers,
`triton/itemsize.py` and the `_DTYPE_SIZES` dict in kernels/nbx_tensor.py:192, become one module.
Both engine branches read this module; the engine keeps no reader of its own (its PyTorch-branch mirror left 2026-10-29).

```python
class DType(IntEnum):        # values frozen; new members are appended, never renumbered
    float16 = 0; bfloat16 = 1; float32 = 2; float64 = 3
    int8 = 4; int16 = 5; int32 = 6; int64 = 7; uint8 = 8; bool = 9
    complex64 = 10; complex128 = 11
    # appended in M2 from the table: uint16 = 12, uint32 = 13, uint64 = 14,
    # float8_e4m3fn = 15, float8_e5m2 = 16

def dtype(name: str) -> DType            # canonical name or a table alias ("fp16", "bf16", "int1"); else ValueError naming the table
def name(dt: DType) -> str               # "float16" … ; `bool`, never `bool_`
def itemsize(dt: DType) -> int           # bytes per element (triton/itemsize.py:80 itemsize)
def is_floating(dt: DType) -> bool       # nbx_tensor.py:240 _FLOATING_DTYPES
def is_complex(dt: DType) -> bool        # nbx_tensor.py:303
def key_name(dt: DType) -> str           # the launch-key spelling ("fp16"); launch_keys.py:49 tag
def from_safetensors(code: str) -> DType # itemsize.py from_safetensors
def storage_dtype(dt: DType, stores_fp64: bool) -> DType   # itemsize.py storage_dtype (the table's without_fp64)
def representation_bytes(enc: str, numel: int, split: bool = False) -> int  # dense, hi+lo split, packed (itemsize.py)
def finite_max(dt: DType) -> float       # triton/dtype.py:136 _HALF_MAX, extended to every floating dtype from the table
def saturate(value: float, dt: DType) -> float    # triton/dtype.py:139 saturate_scalar
def promote(dts: Sequence[DType]) -> DType        # the ONE promotion rule (below)
def accumulator(dt: DType) -> DType      # triton/dtype.py:280 contraction_accumulator_dtype
```

**Promotion.** Today there are two rules. `TritonDtypeEngine._wrap_promote` (triton/dtype.py:1069) takes
the widest floating input, but only when that input is wider than 2 bytes. `wrappers._promote_int_unary`
(kernels/wrappers.py:944) turns an integer or bool input of a float-math unary into float32. The v1
`promote(dts)` keeps both and nothing else:

- floats present and the widest is wider than 2 bytes: return that widest float;
- floats present, all 2 bytes wide: no change (the AMP policy chooses);
- no float present and the op is float-math: float32.

Whether an op is float-math stays policy, held by the engine's AMP op sets. The library takes it as an
argument: `promote(dts, float_math: bool)`.

**Host conversions: `nbxtensor.dtype.convert`, one module in place of three copies.** The three copies
today are nbx_tensor.py:245,276, kernels/autotune_certify.py:269,302 and triton/weight_loader.py:110.

```python
def f32_to_bf16_bits(a: np.ndarray, *, chunk_bytes: int) -> np.ndarray   # round to nearest, ties to even; NaN stays a quiet NaN (nbx_tensor.py:276 = autotune_certify.py:269)
def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray                     # exact widening (autotune_certify.py:302; the void-carrier decode of nbx_tensor.py:245 becomes a view to uint16 at the call)
def bf16_bits_to_f16(bits: np.ndarray, *, workers: int) -> np.ndarray     # weight_loader.py:110, the parallelism an argument instead of the loader's global pool
```

`chunk_bytes` is the certifier's streaming chunk, which kept the 14 GB guard from firing on CogVideoX
(autotune_certify.py:278). `workers` is the loader's split. Both become arguments, so neither leaves a
module-global pool behind. The device-side bf16 to fp16 kernel (weight_loader.py:456
`_bf16_to_fp16_inplace`) becomes `astype` on a tensor viewed over the same buffer (section 4).

### `nbxtensor.dtype.promotion` — the Array API's promotion (DESIGN rule 13)

```python
def binary_result(a: str, b: str) -> str            # two arrays' dtypes -> the dtype they compute in
def scalar_result(dtype: str, scalar) -> str        # an array's dtype and a Python scalar (weak, NEP 50)
def promoted(dtypes: Sequence[str]) -> Optional[str]  # floating inputs brought to the widest, when over two bytes
def float_math_input(dtype: str) -> str             # the dtype a floating-math op reads an input of `dtype` at
```

### `nbxtensor.dtype.operands` — the kernels' operand decisions (DESIGN rule 12)

```python
def contraction_accumulator_dtype(dtype: str) -> str        # a contraction's partials: float32 for a half dtype
def contraction_accumulator_bytes(store_bytes: int) -> int  # the same rule read on a width
def matrix_unit_representation(dtype: str, unit: dict) -> Optional[str]                 # "native" / "split" / None
def matrix_unit_flash_representation(q: str, k: str, v: str, unit: dict) -> Optional[str]
ATTENTION_OPERAND_ALIGNMENTS = ("narrowest", "widest")
def attention_operand_alignment(profile: dict) -> str       # a device's precision.attention_operands, refused if absent
def attention_operand_dtypes(q, k, v, alignment, q_round=None) -> tuple   # (q, k, v, q_round) the attention runs at
def matrix_unit_split_scale(unit: dict) -> tuple            # (hi_exp, lo_shift) of the split representation
def constant_load_dtype(traced: str, compute: str, stores_fp64: bool) -> str   # the dtype a graph constant is bound in
```

### `nbxtensor.dtype.outputs` — an op's output dtype (DESIGN rule 12)

```python
COMPLEX_DTYPE_NAMES = frozenset({"complex32", "complex64", "complex128", "chalf", "cfloat", "cdouble"})
def traced_output_dtype_name(record: dict) -> Optional[str]   # output_dtypes[0] at the record's top level, plain name
def traced_output_is_complex(record: dict) -> bool            # any traced output complex (never the op's name or inputs)
def amp_fp32_output_dtype(compute_dtype, graph_dtype, safe, narrowed, traced=None) -> str
    # the output of an op computed float32 inside a half forward: bfloat16 follows the graph dtype,
    # float16 narrows only under `safe` / `narrowed`; a traced float32 island only under `narrowed`
```

`nbxtensor.dtype.convert.saturate_scalar(value, dtype)`: a Python number clamped to a half float's finite range (the table's `max`); anything else unchanged.

### `nbxtensor.memory` — the names left: allocator, arenas, pools, transfers, layouts, OOM advice

`nbxtensor.memory.allocator`

```python
class ImpossibleExtentError(RuntimeError)
    # An op would produce a tensor with an extent that cannot exist (<= 0 where the op defines a length, or a rank the op does not produce).
class AllocatorInterceptor
    # What stands in front of the allocator's five device operations (allocate, free, copy, set, synchronise) while installed.
```

`nbxtensor.memory.arena`

```python
class Arena
    # O(1) tensor storage via integer-indexed list.
```

`nbxtensor.memory.device_memory`

```python
class AbiMemory(DeviceMemory)
    # The memory primitives of a runtime that exposes the C ABI of a CUDA-style runtime (CUDA, HIP), through its symbol table.
```

`nbxtensor.memory.device_transfer`

```python
def parse_device_idxs(device)
    # All CUDA ordinals named in a device string, in order of appearance.
def parse_device_idx(device)
    # The first CUDA ordinal named in a device string.
def needs_move(t: NBXTensor, target_dev: int)
    # True iff `t` must be transferred to land on cuda:target_dev.
def transfer_tensor(tensor: NBXTensor, target_dev: int)
    # Copy an NBXTensor to cuda:target_dev, preserving shape/strides/dtype.
```

`nbxtensor.memory.layout`

```python
def tensor_nbytes(shape: Sequence[int], dtype)
    # The bytes a dense tensor of `shape` holds at `dtype` (a name, a torch spelling or an NBXDtype, as `itemsize`).
class ArenaLayout
    # The placement of tensors in one arena: each tensor's offset and the arena's total bytes.
def arena_layout(sizes: Iterable[int], alignment: int=ARENA_ALIGNMENT)
    # The layout of tensors of `sizes` bytes (placement order) in one arena
```

`nbxtensor.memory.memory_pool`

```python
def release_flow_memory(device=None)
    # Release what a finished stage held: synchronise the devices named, collect garbage (finalizers free), give the pool's cached blocks back to the driver.
class ComponentArena
    # Single contiguous GPU memory block for one component's weights.
class HostPinnedArena
    # One page-locked host block for weights held on the host, sub-allocated like `ComponentArena`.
    # On a backend that registers host memory (`DeviceMemory.host_register`), ordinary memory written by the load,
    # then page-locked once by `pin()` before any transfer reads it; elsewhere the backend's page-locked allocation.
```

`nbxtensor.memory.metal`

```python
class MetalMemory(DeviceMemory)
    # The memory primitives of the one Metal device of this process, over its runtime.
```

`nbxtensor.memory.oom_advice`

```python
def what_would_have_fit(oom, shape: Optional[Sequence]=None)
    # One sentence naming what would have made the refused allocation fit.
def annotate(message: str, cause, shape: Optional[Sequence]=None)
    # Append the advice to an op-failure message when the cause is an OOM.
```

`nbxtensor.memory.pointer_tables`

```python
def release_pinned_tables()
    # Drop every cached table that PINS (Metal), releasing its pins
class PtrTables
    # Per-projection absolute pointer tables for zero-copy expert access.
```

`nbxtensor.memory.pool`

```python
class SegmentPool
    # The free blocks of every device's segments, and the blocks in use inside them.
class Simulation
    # What the allocator holds for a sequence of allocations and frees
def simulate(events: Iterable[Tuple[str, object, int]], parked_cap: int=0, align: int=POOL_ALIGN, split_min: int=POOL_SPLIT_MIN, pooled: bool=True)
    # Replay `events` — ("alloc", key, nbytes), ("arena", key, nbytes) and ("free", key, 0), `key` any hashable naming one allocation — through the allocator's own pool (`SegmentPool`) and its own order
```

`nbxtensor.memory.shadow`

```python
def malloc(nbytes: int, dev_idx: Optional[int]=None, pooled: bool=True)
    # `DeviceAllocator.malloc_direct` in a shadow, its signature kept whole (`pooled`
def shadow_range(addr: int)
    # (base, nbytes) of the RECORDED shadow allocation containing `addr`, or None.
```

`nbxtensor.memory.slab`

```python
class StepSlab(SlabAllocator)
    # Per-sequence slab RECYCLED at every decode step (P-REPLAY-KV- DECODE B1).
class SlabInterceptor(AllocatorInterceptor)
    # The allocator's interceptor for slab-served runs.
```

### `nbxtensor.ops.tiling_sizes` — the sizes the tiled and chunked paths cut by (the tiling table in force: `ops.tiling_policy`)

Every figure is read from the tiling table in force (`set_tiling` / `tiling_table`), never written in the code.

```python
def conv3d_chunk_bytes()
    # The per-transient bound of the chunked conv3d path (`conv3d.chunk_bytes`).
def conv2d_band_bytes()
    # The conv2d band-streaming threshold (`conv2d.band_bytes`).
def conv3d_need(x_shape: Sequence[int], w_shape: Sequence[int], stride, padding, dilation, in_bytes: int, out_bytes: int)
    # (bytes the ONE-SHOT path holds at its peak, folded bytes per output frame).
def conv3d_one_shot_peak(x_shape: Sequence[int], w_shape: Sequence[int], stride, padding, dilation, in_bytes: int, out_bytes: int)
    # The bytes the one-shot conv3d path (`_conv3d_via_conv2d`) holds at its peak BESIDE its input
def residual_chain_bands(H: int, tile_factor: int, halo: int)
    # The bands a band-streamed residual chain runs over its spatial axis
def chunk_frames(frame_bytes: int)
    # Output frames per chunk
def conv2d_band_rows(N: int, out_c: int, out_h: int, out_w: int, out_bytes: int, band_bytes: int)
    # (output rows of one band, bytes of one output row) of conv2d band streaming
def tiled_conv2d_bands(IH: int, out_h: int, kh: int, sh: int, dh: int, pad_h: int, tile_factor: int)
    # The bands of the op-level tiled conv (real halo)
def tiled_conv2d_band_bytes(x_shape: Sequence[int], w_shape: Sequence[int], stride, padding, dilation, tile_factor: int, in_bytes: int, out_bytes: int)
    # What a tiled 2-D conv holds BESIDE its input and its whole output while its largest band runs
def sdpa_scores_bytes(batch: int, nheads: int, Tq: int, Tk: int)
    # The bytes of an attention's fp32 scores.
def sdpa_scores_bound(budget_bytes: int, head_dim_pow2: bool)
    # The fp32 scores bound an attention routes on
def sdpa_chunk_rows(bound: int, batch: int, nheads: int, Tq: int, Tk: int, min_chunk_rows: int, max_chunks: int)
    # Query rows per chunk that keep each chunk's fp32 scores within `bound`, aligned to the arch's row block (`memory.sdpa_math_min_chunk_rows`); 0 when the shape cannot chunk within the arch's chunk ceiling (`memory.sdpa_math_max_chunks`).
def sdpa_chunk_plan(bound: int, batch: int, nheads: int, Tq: int, Tk: int, min_chunk_rows: int)
    # (slices per chunk, query rows per chunk) that keep each chunk's fp32 scores within `bound`
def sdpa_device_scores_budget(base_bytes: int, device_fraction: float, device_memory_mb)
    # The scores budget on one device
def sdpa_math_scores_budget(base_bytes: int, device_fraction: float, device_memory_mb)
    # The math route's scores budget on the device that executes an attention
def op_budget_fraction()
    # The fraction of a card an op's transient is measured against (`op_level.op_budget_card_fraction`).
def op_budget_bytes(card_bytes: int, resident_bytes: int)
    # An op's budget on its card
def conv_band_factor(total_bytes: int, kept_bytes: int, card_bytes: int)
    # Bands an overflowing convolution (or upsample->conv pair) runs in
def upsample_overflows(out_bytes: int, card_bytes: int)
    # An upsample output worth fusing into its conv even when the conv does not overflow.
def rms_norm_overflows(out_bytes: int, card_bytes: int)
    # An rms_norm whose output runs in H bands.
def rms_norm_band_factor(out_bytes: int, card_bytes: int)
    # Bands an overflowing rms_norm runs in
def residual_chain_band_factor(bytes_fp32: int)
    # Bands a residual chain streams in
def residual_chain_min_base_bytes_fp32()
    # The base-tensor size (fp32) from which a residual chain is detected.
def spatial_halo(extent: int)
    # The proportional spatial overlap of a tile of `extent`
def spatial_overlap(extent: int)
    # A spatial tile's overlap
def temporal_halo(t_tile: int)
    # The proportional temporal overlap of a tile of `t_tile` frames
def inplace_min_bytes()
    # The output size from which an add or an element-wise unary writes into its dead input.
```

### `nbxtensor.ops.fused_upsample_conv` — a convolution streamed in bands of output rows

```python
def tiled_conv2d(x, weight, bias, sh, sw, pad_h, pad_w, dh, dw, groups, tile_factor) -> NBXTensor
    # conv2d computed in `tile_factor` bands of output rows, each band reading its halo rows (zero padding only at
    # the image's edges): the bytes of one conv2d, with the peak of one band
def tiled_upsample_conv2d(x_or_proxy, weight, bias, stride, padding, dilation, transposed, output_padding,
                          groups, tile_factor) -> NBXTensor
    # the same over a nearest upsample fused in (`FusionUpsampleProxy`): the upsampled input is never materialised
def pair(v) -> tuple           # (h, w) of a convolution argument given as one int or a pair
```

### `nbxtensor.ops.elementwise_wrappers` — element-wise ops on NBXTensor (activations, unary and binary arithmetic, comparisons, bitwise and logical, interpolation)

`nbxtensor.ops.elementwise_wrappers`

```python
def begin_run()
    # A Triton sequence's run begins
def relu(x, out=None)
def silu(x, out=None)
def gelu(x, approximate: str='none', out=None)
def sigmoid_wrapper(x)
def tanh_wrapper(x)
def hardsigmoid(x)
def hardswish(x, out=None)
def leaky_relu(x, negative_slope: float=0.01, out=None)
def elu(x, alpha: float=1.0, out=None)
def mish(x, out=None)
def selu_wrapper(x)
def neg(x)
def exp(x)
def sin(x)
def cos(x)
def rsqrt(x)
def sqrt_wrapper(x)
def abs_wrapper(x)
def log_wrapper(x)
def reciprocal(x)
def pow_wrapper(x, exponent)
def clamp(x, min_val=None, max_val=None)
def erf(x)
def floor_wrapper(x)
def ceil_wrapper(x)
def floor_divide_wrapper(a, b)
    # aten::floor_divide — floor(a / b), result in a's dtype.
def round_wrapper(x)
def trunc_wrapper(x)
def add_inplace_nbx(target, other, alpha: float=1.0)
    # In-place element-wise add
def mul(a, b)
def div(a, b, rounding_mode=None)
def rsub(a, b)
    # Reverse subtraction
def maximum_wrapper(a, b)
def minimum_wrapper(a, b)
def gt(a, b)
def ge(a, b)
def lt(a, b)
def eq(a, b)
def ne(a, b)
def isin_wrapper(elements, test_elements, *, invert: bool=False, assume_unique: bool=False)
    # aten::isin — elementwise membership test.
def invoke_silu_and_mul(input_tensor, M, N)
    # Launch fused SwiGLU kernel
def swiglu_fused_wrapper(gate, up)
    # Fused SwiGLU with split gate/up tensors
def rope_fused_wrapper(q_raw, k_raw, cos, sin)
    # Fused RoPE (Liger-style rotate_half) — applies to Q and K in one launch.
def bf16_to_fp16_gpu(src: NBXTensor, out: 'NBXTensor | None'=None)
    # bfloat16 patterns (a 2-byte integer tensor holding the raw bits) to float16 on the device
def glu_wrapper(x, dim: int=-1, act_func: str='sigmoid')
    # Gated Linear Unit
def le(a, b)
def softplus_wrapper(x, beta: float=1.0, threshold: float=20.0)
def remainder_wrapper(a, b)
    # Remainder (modulo)
def exp2_wrapper(x)
def tan_wrapper(x)
def celu_wrapper(x, alpha: float=1.0)
def log_sigmoid_wrapper(x)
def isfinite_wrapper(x)
def isinf_wrapper(x)
def isnan_wrapper(x)
def nan_to_num_wrapper(x, nan: float=0.0, posinf: float=3.4028235e+38, neginf: float=-3.4028235e+38, inplace: bool=False)
    # `inplace`
def threshold_wrapper(x, threshold: float, value: float)
def addcdiv_wrapper(input, tensor1, tensor2, value: float=1.0)
    # result = input + value * (tensor1 / tensor2)
def addcmul_wrapper(input, tensor1, tensor2, value: float=1.0)
    # result = input + value * tensor1 * tensor2
def bitwise_and_wrapper(a, b)
def bitwise_or_wrapper(a, b)
def bitwise_not_wrapper(x)
def logical_and_wrapper(a, b)
def logical_or_wrapper(a, b)
def logical_not_wrapper(x)
def lerp_wrapper(input, end, weight)
    # Linear interpolation
def min_wrapper(x, dim=None, keepdim=False)
    # Min reduction.
def max_wrapper(x, dim=None, keepdim=False)
    # Max reduction.
def bincount_wrapper(x, minlength: int=0)
    # Bincount via CPU numpy (input is small 1D int tensor from MoE routing).
def sort_wrapper(x, dim: int=-1, descending: bool=False, stable: bool=False)
    # Sort via Triton radix sort.
def logical_xor_wrapper(a, b)
def bitwise_left_shift_wrapper(a, b)
def bitwise_right_shift_wrapper(a, b)
def addr_wrapper(input, vec1, vec2, beta: float=1.0, alpha: float=1.0)
    # Outer product add
def clamp_min_wrapper(x, min_val)
    # Clamp minimum
def topk_wrapper(x, k: int, dim: int=-1, largest: bool=True, sorted: bool=True)
    # Top-k via 2-stage Triton pipeline.
def trace_wrapper(x)
    # Sum of diagonal elements.
def bernoulli_wrapper(x, p: float=0.5, **kwargs)
    # bernoulli(tensor) → 0/1 values based on probabilities.
def fold_wrapper(*args, **kwargs)
    # fold — NOT YET IMPLEMENTED in Triton.
def clear_bias_caches()
    # Evict the module-global attention bias caches.
def angle_wrapper(x)
    # angle(complex_tensor) → atan2(imag, real).
def complex_wrapper(real, imag)
    # complex(real, imag) → complex64 tensor (interleaved [real, imag] pairs).
def lstm_wrapper(input_, hx, params, has_biases=True, num_layers=1, dropout=0.0, train=False, bidirectional=False, batch_first=False)
    # Triton-pure aten::lstm.
```

### `nbxtensor.ops.shape_wrappers`

`nbxtensor.ops.shape_wrappers`

```python
def grid_sampler_2d_wrapper(inp, grid, interpolation_mode=0, padding_mode=0, align_corners=False)
    # aten::grid_sampler_2d — pure Triton 2D grid sampler.
def copy_to(x, dtype: object)
def upsample_nearest1d_wrapper(x, output_size, scales=None)
    # Upsample nearest 1D via 2D
def upsample_nearest2d_wrapper(x, output_size, scales_h=None, scales_w=None)
    # Upsample nearest 2D.
def triu_wrapper(x, diagonal: int=0)
    # Upper triangular.
def tril_wrapper(x, diagonal: int=0)
    # Lower triangular.
def roll_wrapper(x, shifts, dims=None)
    # Circular shift along given dim(s) — `torch.roll` semantics.
def flip_wrapper(x, dims)
    # Reverse elements along the given dimension(s).
def broadcast_tensors_wrapper(*tensors)
    # aten::broadcast_tensors(Tensor[] tensors) -> Tensor[].
def pixel_shuffle_wrapper(x, upscale_factor: int)
    # Pixel shuffle
def pixel_unshuffle_wrapper(x, downscale_factor: int)
    # Pixel unshuffle
def upsample_nearest3d_wrapper(x, output_size, scales_d=None, scales_h=None, scales_w=None)
    # Upsample nearest 3D
def repeat_interleave_self_int_wrapper(inp, repeats: int, dim: int=0, output_size: int=None)
    # repeat_interleave with scalar repeats
def repeat_interleave_tensor_wrapper(repeats, output_size: int=None)
    # repeat_interleave with tensor repeats
def repeat_interleave_self_tensor_wrapper(inp, repeats, dim: int=0, output_size: int=None)
    # repeat_interleave with tensor repeats applied to input.
def repeat_interleave_wrapper(*args, **kwargs)
    # aten::repeat_interleave dispatcher — routes by args, since the bare op_type cannot distinguish the overloads
def constant_pad_nd_wrapper(x, pad_list, value: float=0.0)
    # constant_pad_nd via Triton kernel.
def pad_wrapper(x, pad_list, mode: str='constant', value: float=0.0)
    # F.pad via Triton — routes by mode + padded-dim count (len(pad_list)//2).
def reflection_pad1d_wrapper(x, pad_list)
    # Reflection pad 1D — `F.pad(mode='reflect')` on the last dim.
def reflection_pad2d_wrapper(x, pad_list)
    # Reflection pad 2D — `F.pad(mode='reflect')` semantics.
def reflection_pad3d_wrapper(x, pad_list)
    # Reflection pad 3D (last three dims) — F.pad(mode='reflect').
def replication_pad1d_wrapper(x, pad_list)
    # Replication pad 1D (last dim) — F.pad(mode='replicate'), one pass (`_replication_pad_last`).
def replication_pad2d_wrapper(x, pad_list)
    # Replication pad 2D (last two dims) — F.pad(mode='replicate'), one pass (`_replication_pad_last`).
def replication_pad3d_wrapper(x, pad_list)
    # Replication pad 3D (last three dims) — F.pad(mode='replicate'), one pass (`_replication_pad_last`).
def interpolate_wrapper(x, size=None, scale_factor=None, mode: str='nearest', align_corners=None, **kwargs)
    # F.interpolate via Triton upsample kernels.
```

### `nbxtensor.ops.linalg_wrappers`

`nbxtensor.ops.linalg_wrappers`

```python
def mm(a, b, _epilogue: int=0)
    # Matrix multiplication
def bmm(a, b, allow_strided_b: bool=False)
    # Batched matrix multiplication
def matmul_wrapper(a, b)
    # General matmul dispatcher
def linear_wrapper(input, weight, bias=None)
    # aten::linear — y = input @ weight.T + bias.
def addmm(bias, a, b, beta: float=1.0, alpha: float=1.0, _epilogue: int=0)
    # C = beta * bias + alpha * (A @ B).
def mm_epilogue_wrapper(a, b, epilogue: str='silu', approximate: str='none')
    # custom::mm_epilogue — aten::mm fused with a unary elementwise epilogue (fusion_vertical pass, EXACT gate).
def addmm_epilogue_wrapper(bias, a, b, beta: float=1.0, alpha: float=1.0, epilogue: str='silu', approximate: str='none')
    # custom::addmm_epilogue — aten::addmm fused with a unary elementwise epilogue (fusion_vertical pass, EXACT gate).
def dot_wrapper(x, y)
    # Dot product
def mv_wrapper(mat, vec)
    # Matrix-vector multiply
def baddbmm_wrapper(input, batch1, batch2, beta: float=1.0, alpha: float=1.0)
    # Batched addmm
def addmv_wrapper(input, mat, vec, beta: float=1.0, alpha: float=1.0)
    # Matrix-vector with bias
def upsample_bilinear2d_wrapper(x, output_size, align_corners=False, scales_h=None, scales_w=None)
    # Bilinear upsampling.
def upsample_linear1d_wrapper(x, output_size, align_corners: bool=False, scales=None)
    # upsample_linear1d via bilinear2d
def mm_cost(a, b, epilogue: int=0, native_bf16=None, force_accum=None, contract_half=None)
    # `mm` on `a` [M, K] and `b` [K, N] (`_cost.Spec`s), in the wrapper's order
def addmm_cost(bias, a, b, epilogue: int=0, native_bf16=None, force_accum=None, contract_half=None)
    # `addmm` on `bias` ([N], [1, N] or [M, N]), `a` [..., K] and `b` [K, N] (`_cost.Spec`s), in the wrapper's order
def bmm_cost(a, b, allow_strided_b: bool=False, native_bf16=None, force_accum=None, contract_half=None, unit=None)
    # `bmm` on `a` [B, M, K] and `b` [B, K, N] (`_cost.Spec`s)
```

### `nbxtensor.ops.conv_wrappers`

`nbxtensor.ops.conv_wrappers`

```python
def conv_transpose_wrapper(x, weight, bias=None, stride=1, padding=0, dilation=1, output_padding=0, groups=1)
    # Transposed convolution — aten::convolution(transposed=True).
def conv3d_chunked_wrapper(x, weight, bias=None, stride=1, padding=0, dilation=1, transposed=False, output_padding=0, groups=1, *args, **kwargs)
    # A 3-D convolution the caller's plan chunks (op-level tiling, `conv3d_chunks`)
def conv2d_wrapper(x, weight, bias=None, stride=1, padding=0, dilation=1, transposed=False, output_padding=0, groups=1, out=None)
    # Convolution forward.
def conv1d_wrapper(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
    # Conv1D via conv2d with unsqueeze.
def conv_transpose1d_wrapper(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)
    # aten::conv_transpose1d — 1D transposed convolution.
def avg_pool2d_wrapper(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
    # Average pooling 2D forward.
def max_pool2d_wrapper(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
    # Max pooling 2D forward.
def max_pool3d_with_indices_wrapper(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False)
    # Max pooling 3D forward — aten::max_pool3d_with_indices.
def max_pool3d_wrapper(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False)
    # aten::max_pool3d — values-only variant.
def adaptive_avg_pool2d_wrapper(x, output_size)
    # Adaptive average pooling 2D.
def conv_depthwise2d_wrapper(x, weight, bias=None, stride=1, padding=0, dilation=1)
    # Depthwise conv2d.
def unfold_backward_wrapper(grad_in, input_sizes, dim, size, step)
    # unfold_backward (overlap-add)
def im2col_wrapper(x, kernel_size, dilation, padding, stride)
    # aten::im2col / nn.Unfold — sliding local blocks of an NCHW tensor.
def conv2d_cost(x, w, stride=1, padding=0, dilation=1, groups=1, bias=False, out_dtype=None, out_given=False, krsc=False, unit=None, depthwise_enabled=True)
    # `conv2d_wrapper` on an input `x` and a weight `w` (`_cost.Spec`s)
def conv3d_cost(x, w, stride=1, padding=0, dilation=1, groups=1, bias=False, out_dtype=None, unit=None)
    # `_conv3d_via_conv2d` on an input `x` [B, Cin, T, H, W] and a weight `w` [Cout, Cin_g, kt, kh, kw]
```

### `nbxtensor.ops.index_wrappers`

`nbxtensor.ops.index_wrappers`

```python
def where_wrapper(cond, x, y)
def masked_fill(x, mask, value: float)
def is_nonzero_wrapper(x)
    # aten::is_nonzero — returns the scalar truthiness of a 1-element tensor.
def embedding(weight, indices, padding_idx=-1, **kwargs)
    # Embedding lookup
def index_select_wrapper(x, dim: int, index)
    # Index select along dimension.
def gather_wrapper(input, dim: int, index)
    # Gather along dimension.
def masked_scatter_wrapper(x, mask, source)
    # aten::masked_scatter — fill mask-True elements of `x` from the flat `source` buffer in row-major order (torch semantics).
def scatter_wrapper(input, dim: int, index, src)
    # Scatter
def scatter_add_wrapper(input, dim: int, index, src)
    # Scatter add
def index_add_wrapper(x, dim: int, index, source, alpha: float=1.0)
    # Index add
def nonzero_1d_wrapper(mask, expected=None)
    # Ascending int64 indices of the True elements of a 1-D mask.
def bucketize_wrapper(x, boundaries, out_int32: bool=False, right: bool=False)
    # aten::bucketize.Tensor — per-element boundary index (linear scan kernel, kernels/ops/bucketize.py).
def index_put_wrapper(x, indices, values, accumulate: bool=False)
    # aten::index_put / index_put_ — scatter-write `values` into `x`.
```

### `nbxtensor.ops.reduction_wrappers`

`nbxtensor.ops.reduction_wrappers`

```python
def mean_wrapper(x, dim=None, keepdim=False, dtype=None)
    # Mean reduction.
def sum_wrapper(x, dim=None, keepdim=False, dtype=None)
    # Sum reduction.
def amax_wrapper(x, dim=None, keepdim=False)
    # Max reduction (one dim, a 1-elem list, or an arbitrary list of dims).
def argmax_wrapper(x, dim=None, keepdim=False)
    # Argmax reduction.
def argmin_wrapper(x, dim=None, keepdim=False)
    # Argmin reduction.
def cumsum_wrapper(x, dim: int=0, dtype=None)
    # Cumulative sum along dimension.
def all_wrapper(x, dim=None, keepdim=False)
    # Test if all elements are non-zero.
def any_wrapper(x, dim=None, keepdim=False)
    # Test if any element is non-zero.
def prod_wrapper(x, dim=None, keepdim=False)
    # Product reduction.
def std_wrapper(x, dim=None, correction=1, keepdim=False)
    # Standard deviation.
def var_wrapper(x, dim=None, correction=1, keepdim=False)
    # Variance.
def scatter_reduce_wrapper(input, dim: int, index, src, reduce: str, include_self: bool=True)
    # scatter_reduce
```

### `nbxtensor.ops.norm_wrappers`

`nbxtensor.ops.norm_wrappers`

```python
def native_layer_norm(x, normalized_shape, weight, bias, eps=1e-05)
    # LayerNorm matching ATen native_layer_norm signature.
def layer_norm_wrapper(x, normalized_shape, weight=None, bias=None, eps=1e-05, cudnn_enable=True)
    # aten::layer_norm — the high-level op returns ONLY the normalized output (native_layer_norm also returns mean/rstd).
def norm_wrapper(x, p=2, dim=None, keepdim=False)
    # aten::norm(self, p, dim, keepdim) — Lp norm reduction over ``dim``.
def group_norm_wrapper(x, num_groups: int, weight, bias, eps=1e-05)
    # GroupNorm.
def native_group_norm_wrapper(input, weight, bias, N, C, HxW, num_groups, eps)
    # Adapter for ATen's low-level `aten::native_group_norm`.
def batch_norm_wrapper(x, weight, bias, running_mean, running_var, training: bool=False, momentum: float=0.1, eps: float=1e-05)
    # BatchNorm / InstanceNorm forward (kernel from FlagGems).
def vector_norm_wrapper(x, ord: float=2.0, dim=None, keepdim=False)
    # Vector norm — dispatches by ord to L0/L1/L2/Linf/Lp kernels.
def weight_norm_interface_wrapper(v, g, dim: int=0)
    # Weight normalization
def normal_wrapper(*args, **kwargs)
    # normal(mean, std, size) → N(mean, std) via Triton randn + transform.
def group_norm_statistics(x, num_groups: int, eps: float)
    # (mean, rstd) [N, G] fp32 of `x` [N, C, ...] over each group — the first pass of an exact tiled group norm (`core/strategies/tiled_norm.py`), over the WHOLE input, without its normalised tensor.
def group_norm_apply(x, weight, bias, num_groups: int, mean, rstd)
    # `aten::native_group_norm`'s three outputs for a slice `x` given the whole input's statistics — the second pass
```

### `nbxtensor.ops.launch_keys`

`nbxtensor.ops.launch_keys`

```python
def matmul_out_dtype(dt: NBXDtype, M: int, force_fp32: bool, native_bf16: bool, force_accum: bool=False, contract_half: bool=False)
    # The store dtype of mm/bmm/addmm/mv/addmv (see `wrappers._matmul_out_dtype` for the why)
def mm_dtypes(a: NBXDtype, b: NBXDtype, native_bf16: bool, force_accum: bool=False)
    # `mm`'s operand decisions
def addmm_dtypes(a: NBXDtype, b: NBXDtype, bias: NBXDtype, native_bf16: bool, force_accum: bool=False)
    # `addmm`'s operand decisions
def bmm_dtypes(a: NBXDtype, b: NBXDtype, native_bf16: bool, force_accum: bool=False)
    # `bmm`'s operand decisions
def matrix_unit_operands(a: NBXDtype, b: NBXDtype, M: int, unit: Optional[dict]=None)
    # (rep_a, rep_b)
def matrix_unit_takes(a: NBXDtype, b: NBXDtype, M: int)
    # The profile's matrix unit runs this GEMM (`matrix_unit_operands`).
def gemm_row_representative(m_bucket: int, a: NBXDtype, b: NBXDtype)
    # The largest row count M whose GEMM launch forms the autotune key of M bucket `m_bucket`
def matrix_unit_native(a: NBXDtype, b: NBXDtype, M: int, unit: Optional[dict]=None)
    # Both operands are the unit's operand dtype in memory
def mv_launches(N: int, K: int, mat: NBXDtype, vec: NBXDtype, native_bf16: bool, force_accum: bool=False, contract_half: bool=False)
    # `mv_wrapper` on a [N, K] matrix and a [K] vector
def addmv_launches(N: int, K: int, mat: NBXDtype, vec: NBXDtype, bias: NBXDtype, native_bf16: bool, force_accum: bool=False, contract_half: bool=False)
    # `addmv_wrapper` on a [N, K] matrix, a [K] vector and a [N] bias
def mm_launches(M: int, K: int, N: int, a: NBXDtype, b: NBXDtype, native_bf16: bool, force_accum: bool=False, contract_half: bool=False)
    # `mm` (and `mm_epilogue`)
def addmm_launches(M: int, K: int, N: int, a: NBXDtype, b: NBXDtype, bias: NBXDtype, native_bf16: bool, force_accum: bool=False, contract_half: bool=False, bias_row: bool=True)
    # `addmm` on a 2-D activation (an N-D one is flattened to (numel/K, K) first)
def bmm_launches(M: int, K: int, N: int, a: NBXDtype, b: NBXDtype, native_bf16: bool, force_accum: bool=False, contract_half: bool=False)
    # `bmm`
def baddbmm_launches(M: int, K: int, N: int, a: NBXDtype, b: NBXDtype, bias: NBXDtype)
    # `baddbmm_wrapper`
def mm_out(M: int, a: NBXDtype, b: NBXDtype, native_bf16: bool, force_accum: bool=False, contract_half: bool=False)
    # The dtype `mm` stores for these operands (its launch or its per-row GEMV alike).
def bmm_out(M: int, a: NBXDtype, b: NBXDtype, native_bf16: bool, force_accum: bool=False, contract_half: bool=False)
    # The dtype `bmm` stores for these operands
def row_breakpoints(lo: int, hi: int)
    # The row counts M in (lo, hi] at which a GEMM's launch CLASS changes on the bound profile
def matmul_route(a_rank: int, b_rank: int)
    # `matmul_wrapper`'s route by operand ranks (torch.matmul semantics)
def matmul_launches(a_shape, b_shape, a: NBXDtype, b: NBXDtype, native_bf16: bool, force_accum: bool=False, contract_half: bool=False)
    # `matmul_wrapper` (and `linear_wrapper`, which calls it with the transposed weight)
def lstm_launches(B: int, T: int, I: int, H: int, num_layers: int, bidirectional: bool, x: NBXDtype, native_bf16: bool)
    # `lstm_wrapper` -> `_lstm_run_direction`, per layer and direction
def dft_r2c_launches(M: int, N: int, onesided: bool, native_bf16: bool)
    # `fft_r2c_wrapper` on a length that is not a power of two
def dft_c2r_launches(M: int, bins: int, N: int, native_bf16: bool)
    # `fft_c2r_wrapper` -> `_dft_c2r`, any length
def stft_launches(lead: int, length: int, n_fft: int, hop_length: Optional[int], win_length: Optional[int], onesided: Optional[bool], native_bf16: bool)
    # `stft_wrapper` on a signal of `lead` rows x `length` samples
def istft_launches(batch: int, bins: int, frames: int, n_fft: int, native_bf16: bool)
    # `istft_wrapper` on a [batch, bins, frames] spectrum
def sdpa_operand_dtypes(q: NBXDtype, k: NBXDtype, v: NBXDtype, q_round: Optional[NBXDtype]=None, *, alignment: Optional[str]=None)
    # The (q, k, v, q_round) dtypes `scaled_dot_product_attention_wrapper` computes attention with
def decode_vec_takes(headdim: int, headdim_v: int, mask_numel: Optional[int], seqlen_k: int)
    # Whether a single-query attention on the math route runs the vector decode kernel (no autotuned key) — `_try_decode_vec`'s contract
def flash_any_head_dim()
    # The arch's profile MEASURED the tl.dot flash kernel correct at every head dim (`flash.any_head_dim_correct`)
def flash_tiles_certified()
    # The tl.dot flash's tile is certified per key (workstream 1) where the profile declares no hand table of tiles (`sdpa_thresholds`)
def flash_launches(Tq: int, Tk: int, D: int, q: NBXDtype, k: NBXDtype, v: NBXDtype, bias: Optional[NBXDtype]=None)
    # The tl.dot flash where its tile is certified (`flash_tiles_certified`)
def flash_headdim_detour(D: int, enabled: bool=True)
    # The head dim the flash route runs at
def conv_weight_layout()
    # The order a convolution weight is stored in on this arch
def conv_weight_krsc(w_shape, groups: int, transposed: bool, layout: str)
    # Whether a convolution weight of this shape is stored KRSC under `layout`
def krsc_conv_weights(dag: dict, layout: str)
    # The tensor ids of the graph weights stored KRSC
def is_krsc_weight(w)
    # A logical (K, C, R, S) weight whose memory is (K, R, S, C), contiguous.
def unit_flash(D: int, q: NBXDtype, k: NBXDtype, v: NBXDtype, unit: Optional[dict]=None)
    # (representation, tile) of this attention's flash on the profile's matrix unit, or None
def unit_flash_takes(D: int, q: NBXDtype, k: NBXDtype, v: NBXDtype, unit: Optional[dict]=None)
    # Whether the device's matrix unit runs this attention's flash route (`unit_flash`)
def unit_flash_launches(Tq: int, Tk: int, D: int, q: NBXDtype, k: NBXDtype, v: NBXDtype, bias: Optional[NBXDtype]=None)
    # The flash on the profile's matrix unit (`unit_flash`, ops/flash_attention_m8n8k4.py)
def sdpa_route(batch: int, nheads: int, Tq: int, Tk: int, D: int, Dv: int, budget_bytes: int, min_chunk_rows: int, max_chunks: int, force_math: bool=False, unit_flash: bool=False)
    # The attention route
def math_attention_launches(B: int, H: int, Hk: int, Tq: int, Tk: int, D: int, Dv: int, q: NBXDtype, k: NBXDtype, v: NBXDtype, native_bf16: bool, force_accum: bool=False, contract_half: bool=False)
    # `_math_attention`
def chunked_math_attention_launches(B: int, H: int, Hk: int, Tq: int, Tk: int, D: int, Dv: int, q: NBXDtype, k: NBXDtype, v: NBXDtype, native_bf16: bool, rows: int, force_accum: bool=False, contract_half: bool=False)
    # `_math_attention_chunked`
def conv_width_key(in_h: int, out_h: int, in_w: int, out_w: int, kw: int, stride_w: int, pad_w: int, dil_w: int)
    # The width a convolution's key carries
def conv_dtypes(x: NBXDtype, w: NBXDtype)
    # A convolution's operands, narrowed to the narrowest of the pair when they differ.
def conv_out_hw(in_h, in_w, kh, kw, sh, sw, ph, pw, dh, dw)
def conv2d_route(N: int, in_c: int, out_c: int, out_h: int, out_w: int, dh: int, dw: int, groups: int, x: NBXDtype, band_bytes: int, depthwise_enabled: bool=True)
    # "depthwise" (groups == in_c == out_c at dilation 1), "band" (the output, sized at the input's dtype, over `band_bytes`) or "plain" — the order the wrapper tests them in.
class ConvRowOverBand(ValueError)
    # A convolution whose single output row is over the band budget
def conv2d_launches(N: int, in_c: int, in_h: int, in_w: int, out_c: int, kh: int, kw: int, sh: int, sw: int, ph: int, pw: int, dh: int, dw: int, groups: int, x: NBXDtype, w: NBXDtype, compute: Optional[NBXDtype], band_bytes: int, depthwise_enabled: bool=True)
    # `conv2d_wrapper` on a 4-D weight
def tiled_conv2d_launches(N: int, in_c: int, IH: int, IW: int, out_c: int, kh: int, kw: int, sh: int, sw: int, ph: int, pw: int, dh: int, dw: int, groups: int, x: NBXDtype, w: NBXDtype, compute: Optional[NBXDtype], band_bytes: int, tile_factor: int)
    # The launches of an op-level tiled conv
def conv_launches(x_shape, w_shape, stride, padding, dilation, transposed: bool, groups: int, x: NBXDtype, w: NBXDtype, compute: Optional[NBXDtype], band_bytes: int, depthwise_enabled: bool=True)
    # `conv2d_wrapper` by the weight's rank
def softmax_feat_bucket(feat: int)
    # The row length's class in a softmax key
def softmax_launches(rows: int, feat: int, log: bool, x: NBXDtype, out: NBXDtype)
    # `softmax` / `log_softmax` over a row of `feat` (rows = every other element)
def layer_norm_launches(rows: int, feat: int, has_weight: bool, has_bias: bool, x: NBXDtype, w: Optional[NBXDtype], b: Optional[NBXDtype], out: NBXDtype)
    # `wrappers.native_layer_norm` over rows of `feat` (the normalized extent
def rms_norm_launches(rows: int, feat: int, has_weight: bool, x: NBXDtype, w: Optional[NBXDtype], out: NBXDtype)
    # `wrappers.rms_norm` over rows of `feat` (the last dim)
def elementwise_classes(largest: int)
    # Every element-wise class (`ops/elementwise_launch.ew_bucket`
def group_norm_launches(N: int, C: int, HxW: int, groups: int, has_weight: bool, has_bias: bool, x: NBXDtype, w: Optional[NBXDtype], b: Optional[NBXDtype])
    # `wrappers.group_norm_wrapper` (one program per batch x group, the group's channel x spatial extent walked in chunks)
def row_reduce_launches(kind: str, rows: int, feat: int, x: NBXDtype)
    # `wrappers.mean_wrapper` / `sum_wrapper` / `amax_wrapper` over rows of `feat` (the reduced elements
```

### `nbxtensor.ops` — the kernel modules and the remaining wrappers (each module's public kernels, launchers and helpers)

`nbxtensor.ops.abs`

```python
def abs_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.adaptive_avg_pool2d`

```python
def adaptive_avg_pool2d_kernel(input_ptr, output_ptr, n_elements, IH, IW, OH, OW, stride_n, stride_c, stride_h, stride_w, BLOCK_SIZE: tl.constexpr)
    # Adaptive average pool 2D.
def adaptive_avg_pool2d_tiled_kernel(input_ptr, output_ptr, N, C, IH, IW, OH, OW, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr)
    # Tiled variant for spatial parallelism.
```

`nbxtensor.ops.add`

```python
def add_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr)
    # out = x + alpha * y (tensor + tensor)
def add_scalar_kernel(x_ptr, output_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr)
    # out = x + scalar
def add_scalar_dev_kernel(x_ptr, s_ptr, output_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr)
    # out = x + (s * alpha) where s is a 0-d DEVICE tensor.
def add_bias_broadcast_kernel(x_ptr, bias_ptr, output_ptr, n_elements, feat_dim, alpha, BLOCK_SIZE: tl.constexpr)
    # out[i] = x[i] + alpha * bias[i % feat_dim].
```

`nbxtensor.ops.addcdiv`

```python
def addcdiv_forward_kernel(input_ptr, tensor1_ptr, tensor2_ptr, output_ptr, n_elements, value, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.addcmul`

```python
def addcmul_forward_kernel(input_ptr, tensor1_ptr, tensor2_ptr, output_ptr, n_elements, value, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.addmv_op`

```python
def addmv_kernel(A_ptr, B_ptr, inp_ptr, out_ptr, N, M, N_BUCKET, M_BUCKET, alpha, beta, stride_an, stride_am, stride_bm, stride_in, stride_outn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # Matrix-vector multiply with bias
```

`nbxtensor.ops.addr`

```python
def addr_kernel(input_ptr, vec1_ptr, vec2_ptr, output_ptr, beta, alpha, M, N, stride_input_m, stride_input_n, stride_vec1, stride_vec2, stride_output_m, stride_output_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # Outer product add
```

`nbxtensor.ops.all_reduce`

```python
def reduce_all(a, b)
def all_kernel_mid(inp, mid, n_elements, mid_size, BLOCK_SIZE: tl.constexpr)
    # Pass 1
def all_kernel_result(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr)
    # Pass 2
def all_kernel_dim(inp, out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # Check all non-zero along compressed inner dim N.
```

`nbxtensor.ops.amax`

```python
def amax_forward_kernel(input_ptr, output_ptr, batch_dim, feat_dim, BATCH_BUCKET, FEAT_BUCKET, input_batch_stride, input_feat_stride, BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr)
    # Max reduction over the last dimension
```

`nbxtensor.ops.any_reduce`

```python
def reduce_any(a, b)
def any_kernel_mid(inp, mid, n_elements, mid_size, BLOCK_SIZE: tl.constexpr)
    # Pass 1
def any_kernel_result(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr)
    # Pass 2
def any_kernel_dim(inp, out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # Check any non-zero along compressed inner dim N.
```

`nbxtensor.ops.arange_op`

```python
def arange_kernel(output_ptr, start, step, n_elements, BLOCK_SIZE: tl.constexpr)
    # Generate sequential values
```

`nbxtensor.ops.argmax`

```python
def argmax_kernel_1(inp, mid_value, mid_index, M, BLOCK_SIZE: tl.constexpr)
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr)
def argmax_kernel_inner(inp, out_index, M, N, TILE_N: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr)
    # Argmax along the innermost (last) dimension.
```

`nbxtensor.ops.argmin`

```python
def argmin_kernel_1(inp, mid_value, mid_index, M, BLOCK_SIZE: tl.constexpr)
def argmin_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr)
def argmin_kernel(inp, out_index, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # Argmin along an arbitrary dimension.
```

`nbxtensor.ops.atan2`

```python
def atan2_kernel(y_ptr, x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.attention_wrappers`

```python
def paged_kv_append(new, pool, slot_mapping)
    # Write new positions into a layer's paged pool (state engine step 2, ops/paged_decode_attn.py).
def paged_decode_attention(q, k_pool, v_pool, block_table, seq_lens, softmax_scale, max_seq_len, block_n, q_round=None)
    # Decode attention (one query position) read through the page table — the vector split-KV kernel of ops/decode_attn_vec.py on paged K / V, merged by the fixed-order `flash_decode_reduce_kernel`.
def math_attention_cost(q, k, v, mask=None, is_causal=False, native_bf16=None, force_accum=None, contract_half=None, unit=None)
    # `_math_attention` on q [B, H, Tq, D], k [B, Hk, Tk, D], v [B, Hk, Tk, Dv] and an optional additive `mask` (`_cost.Spec`s
def chunked_math_attention_cost(q, k, v, chunk_rows: int, mask=None, is_causal=False, native_bf16=None, force_accum=None, contract_half=None, unit=None)
    # `_math_attention_chunked`
```

`nbxtensor.ops.autotune_settings`

```python
def miss_explainer()
    # The explainer in force, or None.
```

`nbxtensor.ops.avg_pool2d`

```python
def avg_pool2d_forward_kernel(input_ptr, output_ptr, in_stride_n, in_stride_c, in_stride_h, in_stride_w, in_c, in_h, in_w, out_h, out_w, kernel_h: tl.constexpr, kernel_w: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr, padding_h: tl.constexpr, padding_w: tl.constexpr, dilation_h: tl.constexpr, dilation_w: tl.constexpr, COUNT_INCLUDE_PAD: tl.constexpr, divisor_override, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr)
```

`nbxtensor.ops.baddbmm_op`

```python
def baddbmm_kernel(A_ptr, B_ptr, out_ptr, bias_ptr, alpha, beta, M, N, K, M_BUCKET, N_BUCKET, K_BUCKET, stride_ab, stride_am, stride_ak, stride_bb, stride_bk, stride_bn, stride_ob, stride_om, stride_on, bias_batch_stride, bias_m_stride, bias_n_stride, IEEE_PRECISION: tl.constexpr=False, PROMOTE_B: tl.constexpr=False, HAS_BIAS: tl.constexpr=True, BLOCK_M: tl.constexpr=64, BLOCK_N: tl.constexpr=64, BLOCK_K: tl.constexpr=32, GROUP_M: tl.constexpr=8)
    # out[b] = beta*bias[b] + alpha*(A[b] @ B[b]) (HAS_BIAS=True) out[b] = A[b] @ B[b] (HAS_BIAS=False)
```

`nbxtensor.ops.batch_norm`

```python
def batch_norm_forward_kernel(input_pointer, weight_pointer, bias_pointer, mean_pointer, inv_std_pointer, output_pointer, running_mean_pointer, running_var_pointer, batch_dim, spatial_dim, input_batch_stride, input_feat_stride, input_spatial_stride, output_batch_stride, output_feat_stride, output_spatial_stride, momentum, eps, is_train: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
```

`nbxtensor.ops.binary_strided`

```python
def add_strided_nd_kernel(x_ptr, y_ptr, output_ptr, n_elements, alpha, shape_ptr, x_stride_ptr, y_stride_ptr, BLOCK_SIZE: tl.constexpr, NDIM: tl.constexpr)
    # out = x + alpha * y, x and y read by their strides, out contiguous.
def mul_strided_nd_kernel(x_ptr, y_ptr, output_ptr, n_elements, shape_ptr, x_stride_ptr, y_stride_ptr, BLOCK_SIZE: tl.constexpr, NDIM: tl.constexpr)
    # out = x * y, x and y read by their strides, out contiguous.
def sub_strided_nd_kernel(x_ptr, y_ptr, output_ptr, n_elements, alpha, shape_ptr, x_stride_ptr, y_stride_ptr, BLOCK_SIZE: tl.constexpr, NDIM: tl.constexpr)
    # out = x - alpha * y, x and y read by their strides, out contiguous.
def div_strided_nd_kernel(x_ptr, y_ptr, output_ptr, n_elements, shape_ptr, x_stride_ptr, y_stride_ptr, BLOCK_SIZE: tl.constexpr, NDIM: tl.constexpr)
    # out = x / y, x and y read by their strides, out contiguous.
def add_scalar_strided_nd_kernel(x_ptr, output_ptr, n_elements, scalar, shape_ptr, x_stride_ptr, BLOCK_SIZE: tl.constexpr, NDIM: tl.constexpr)
    # out = x + scalar, x read by its strides, out contiguous (add_scalar_kernel's expression).
def mul_scalar_strided_nd_kernel(x_ptr, output_ptr, n_elements, scalar, shape_ptr, x_stride_ptr, BLOCK_SIZE: tl.constexpr, NDIM: tl.constexpr)
    # out = x * scalar, x read by its strides, out contiguous (mul_scalar_kernel's expression).
```

`nbxtensor.ops.bitwise_and`

```python
def bitwise_and_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.bitwise_left_shift`

```python
def bitwise_left_shift_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
    # Bitwise left shift
def bitwise_left_shift_scalar_kernel(x_ptr, output_ptr, n_elements, shift_amount, BLOCK_SIZE: tl.constexpr)
    # Bitwise left shift by scalar
```

`nbxtensor.ops.bitwise_not`

```python
def bitwise_not_forward_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.bitwise_or`

```python
def bitwise_or_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.bitwise_right_shift`

```python
def bitwise_right_shift_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
    # Bitwise right shift
def bitwise_right_shift_scalar_kernel(x_ptr, output_ptr, n_elements, shift_amount, BLOCK_SIZE: tl.constexpr)
    # Bitwise right shift by scalar
```

`nbxtensor.ops.bucketize`

```python
def bucketize_kernel(x_ptr, boundaries_ptr, out_ptr, n_elements, n_boundaries, RIGHT: tl.constexpr, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.cat_op`

```python
def cat_copy_kernel_4(out_ptr, in_ptr_a, in_ptr_b, in_ptr_c, in_ptr_d, dim_size_in_a, dim_size_in_b, dim_size_in_c, dim_size_in_d, dim_size_out, dim_prod_post, dim_offset_a, dim_offset_b, dim_offset_c, dim_offset_d, total_elements_a, total_elements_b, total_elements_c, total_elements_d, BLOCK_X: tl.constexpr)
    # Copy up to 4 input tensors into their correct positions in the output.
```

`nbxtensor.ops.celu`

```python
def celu_forward_kernel(input_ptr, output_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.clamp`

```python
def clamp_forward_kernel(input_ptr, output_ptr, n_elements, min_val, max_val, has_min: tl.constexpr, has_max: tl.constexpr, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.clamp_min`

```python
def clamp_min_forward_kernel(input_ptr, output_ptr, n_elements, min_val, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.conv1d`

```python
def conv1d_forward_kernel(input_pointer, weight_pointer, output_pointer, bias_pointer, in_n, input_length, out_c, out_length, input_n_stride, input_c_stride, input_l_stride, weight_n_stride, weight_c_stride, weight_l_stride, output_n_stride, output_c_stride, output_l_stride, weight_c: tl.constexpr, weight_k: tl.constexpr, stride_l: tl.constexpr, padding_l: tl.constexpr, dilation_l: tl.constexpr, groups: tl.constexpr, BLOCK_NI_LO: tl.constexpr, BLOCK_CI: tl.constexpr, BLOCK_CO: tl.constexpr)
    # Conv1D forward
```

`nbxtensor.ops.conv2d`

```python
def conv2d_forward_kernel(input_pointer, weight_pointer, output_pointer, batch_dim, in_feat_dim, in_height, in_width, out_feat_dim, out_height, out_width, in_width_key, out_width_key, input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride, weight_out_feat_stride, weight_in_feat_stride, weight_height_stride, weight_width_stride, output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride, kernel_height: tl.constexpr, kernel_width: tl.constexpr, stride_height: tl.constexpr, stride_width: tl.constexpr, padding_height: tl.constexpr, padding_width: tl.constexpr, groups: tl.constexpr, fp16: tl.constexpr, dilation_height: tl.constexpr=1, dilation_width: tl.constexpr=1, BLOCK_SIZE_BHW: tl.constexpr=64, BLOCK_SIZE_INF: tl.constexpr=32, BLOCK_SIZE_OUTF: tl.constexpr=64, WIDE_OFFSETS: tl.constexpr=True, SPATIAL_FAST: tl.constexpr=False)
def conv2d_bias_inplace_kernel(out_ptr, bias_ptr, n, HW, C, BLOCK: tl.constexpr)
    # out[i] += bias[(i // HW) % C], in place, flat over a CONTIGUOUS NCHW output.
```

`nbxtensor.ops.conv2d_m8n8k4`

```python
def conv2d_m8n8k4_kernel(X, Wt, Bias, C, M, NG, K, IH, IW, OH, OW, stride_xn, stride_xc, stride_xh, stride_xw, stride_wo, stride_cn, stride_cc, stride_ch, stride_cw, CG, OG, KH: gl.constexpr, KW: gl.constexpr, SH_: gl.constexpr, SW_: gl.constexpr, PH: gl.constexpr, PW: gl.constexpr, DH: gl.constexpr, DW: gl.constexpr, HAS_BIAS: gl.constexpr, MB: gl.constexpr, NB: gl.constexpr, BK: gl.constexpr, GROUP_M: gl.constexpr, L: gl.constexpr, LA: gl.constexpr, LB: gl.constexpr, GA: gl.constexpr, GB: gl.constexpr)
def conv_layouts(MB: int, NB: int, WM: int, WN: int, BK: int)
    # The GEMM's layouts (ops/matmul_m8n8k4.py, B k-contiguous) with the gathered A walked row-fastest
```

`nbxtensor.ops.conv_depthwise2d`

```python
def conv_depthwise2d_kernel(input_ptr, weight_ptr, output_ptr, N, C_in, IH, IW, C_out, KH, KW, OH, OW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ch_mult, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr)
    # Depthwise conv2d kernel.
def conv_depthwise2d_bias_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr, N, C_in, IH, IW, C_out, KH, KW, OH, OW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ch_mult, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr)
    # Depthwise conv2d with bias, same structure as above.
```

`nbxtensor.ops.conv_transpose2d`

```python
def conv_transpose2d_kernel(input_ptr, weight_ptr, output_ptr, N, C_in, IH, IW, C_out, KH, KW, OH, OW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, C_in_per_g, C_out_per_g, BLOCK_SIZE: tl.constexpr)
    # Scatter-based transposed convolution kernel (groups + dilation aware).
```

`nbxtensor.ops.copy`

```python
def copy_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
    # Copy tensor (with implicit dtype conversion via pointer types).
```

`nbxtensor.ops.copy_op`

```python
def copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr, SATURATE_F16: tl.constexpr=False)
    # Copy src to dst.
```

`nbxtensor.ops.cos`

```python
def cos_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.count_nonzero`

```python
def count_nonzero_kernel(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr)
    # Count non-zero elements via block-local sum + atomic global add.
def count_nonzero_dim_kernel(x_ptr, out_ptr, N, numel, BLOCK_SIZE: tl.constexpr)
    # Count non-zero elements along a dimension (after dim_compress).
```

`nbxtensor.ops.cross_entropy`

```python
def cross_entropy_loss_forward_kernel(input_pointer, target_pointer, weight_pointer, sum_weights_pointer, output_pointer, batch_dim, feat_dim, input_batch_stride, input_feat_stride, weighted: tl.constexpr, BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr)
    # Cross entropy loss
```

`nbxtensor.ops.cummax`

```python
def scan_part_max_kernel(inp, out, out_indices, partial_max, partial_max_indices, n_elements, BLOCK_SIZE: tl.constexpr, NEED_PARTIAL: tl.constexpr)
    # Block-local cumulative maximum with optional partial output.
def add_base_max_kernel(out, out_indices, partial_max, partial_max_indices, n_elements, BLOCK_SIZE: tl.constexpr)
    # Fan-out phase
```

`nbxtensor.ops.cummin`

```python
def scan_part_min_kernel(inp, out, out_indices, partial_min, partial_min_indices, n_elements, BLOCK_SIZE: tl.constexpr, NEED_PARTIAL: tl.constexpr)
    # Block-local cumulative minimum with optional partial output.
def add_base_min_kernel(out, out_indices, partial_min, partial_min_indices, n_elements, BLOCK_SIZE: tl.constexpr)
    # Fan-out phase
```

`nbxtensor.ops.cumsum`

```python
def get_scan_accum_type(inp_dtype: tl.dtype)
def scan_part_sum_kernel(inp, out, partial_sum, n_elements, part_num, BLOCK_SIZE: tl.constexpr)
def add_base_sum_kernel(out, partial_sum, n_elements, part_num, BLOCK_SIZE: tl.constexpr)
def scan_part_sum_abc_kernel(inp, out, partial_sum, B, C, part_num, BLOCK_SIZE: tl.constexpr)
def add_base_sum_abc_kernel(out, partial_sum, B, C, part_num, BLOCK_SIZE: tl.constexpr)
def reduce_then_scan_block_sum_kernel_row(in_ptr, block_sum_ptr, N, tiles_per_cta, TILE_SIZE: tl.constexpr)
    # Block sum
def reduce_then_scan_root_scan_kernel_row(in_ptr, out_ptr, N, TILE_SIZE: tl.constexpr)
    # Persistent scan of block sums (or small row).
def reduce_then_scan_block_scan_kernel_row(in_ptr, previous_sum_ptr, out_ptr, N, num_tiles_n, tiles_per_cta, TILE_SIZE: tl.constexpr)
    # Final block scan
```

`nbxtensor.ops.decode_attn_vec`

```python
def decode_attn_vec_split_kernel(q_ptr, k_ptr, v_ptr, bias_ptr, opart_ptr, mpart_ptr, lpart_ptr, T_k, seg_len, sm_scale, stride_qh, stride_kh, stride_kn, stride_vh, stride_vn, stride_oh, stride_os, stride_mh, GQA_GROUPS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr, D_V: tl.constexpr, HAS_BIAS: tl.constexpr, Q_TO: tl.constexpr=None, Q_SATURATE: tl.constexpr=False)
def decode_attn_vec_grouped_kernel(q_ptr, k_ptr, v_ptr, bias_ptr, opart_ptr, mpart_ptr, lpart_ptr, T_k, seg_len, sm_scale, stride_qh, stride_kh, stride_kn, stride_vh, stride_vn, stride_oh, stride_os, stride_mh, GQA_GROUPS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr, D_V: tl.constexpr, HAS_BIAS: tl.constexpr, Q_TO: tl.constexpr=None, Q_SATURATE: tl.constexpr=False)
    # One KV HEAD per program (grid = (B*H_kv, split))
```

`nbxtensor.ops.depthwise_conv2d`

```python
def depthwise_conv2d_kernel(x_ptr, w_ptr, out_ptr, N, C, H_in, W_in, H_out, W_out, W_in_key, W_out_key, x_n_stride, x_c_stride, x_h_stride, x_w_stride, w_c_stride, w_kh_stride, w_kw_stride, out_n_stride, out_c_stride, out_h_stride, out_w_stride, kh: tl.constexpr, kw: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr, pad_h: tl.constexpr, pad_w: tl.constexpr, fp16: tl.constexpr, BLOCK_HW: tl.constexpr=64, BLOCK_C: tl.constexpr=32, WIDE_OFFSETS: tl.constexpr=True)
```

`nbxtensor.ops.dequant_gemv`

```python
def dequant_int4_kernel(wq_ptr, scales_ptr, mins_ptr, out_ptr, K, N, stride_wk, stride_wn, stride_sg, stride_sn, stride_ok, stride_on, BLOCK_K_C: tl.constexpr, BLOCK_N_C: tl.constexpr, GROUP_C: tl.constexpr, PACK_C: tl.constexpr)
    # Standalone dequant
def gemv_ref_kernel(x_ptr, w_ptr, out_ptr, K, N, stride_wk, stride_wn, BLOCK_K_C: tl.constexpr, BLOCK_N_C: tl.constexpr)
    # Reference dense-weight GEMV
def dequant_gemv_int4_kernel(x_ptr, wq_ptr, scales_ptr, mins_ptr, out_ptr, K, N, stride_wk, stride_wn, stride_sg, stride_sn, BLOCK_K_C: tl.constexpr, BLOCK_N_C: tl.constexpr, GROUP_C: tl.constexpr, PACK_C: tl.constexpr)
    # Fused
```

`nbxtensor.ops.dft`

```python
def dft_rfft_matrix_kernel(cos_ptr, nsin_ptr, N_bins, N, two_pi_over_N, BLOCK_SIZE: tl.constexpr)
    # Forward rfft basis
def idft_c2r_matrix_kernel(cos_ptr, sin_ptr, N_bins, N, two_pi_over_N, inv_N, n_even, BLOCK_SIZE: tl.constexpr)
    # Inverse rfft (irfft) basis, exploiting Hermitian symmetry.
```

`nbxtensor.ops.dispatch`

```python
def has_kernel(op_type: str)
def list_kernels()
```

`nbxtensor.ops.div`

```python
def div_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
    # out = x / y (tensor / tensor)
def div_scalar_kernel(x_ptr, output_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr)
    # out = x / scalar
def div_scalar_dev_kernel(x_ptr, s_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
    # out = x / s where s is a 0-d DEVICE tensor.
def int_div_kernel(x_ptr, y_ptr, output_ptr, n_elements, FLOOR: tl.constexpr, BLOCK_SIZE: tl.constexpr)
    # out = x // y on integers, computed in the integers
```

`nbxtensor.ops.dot_op`

```python
def dot_kernel_small(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr)
    # Single-block dot product for small vectors (N < 4096).
def dot_kernel_partial(x_ptr, y_ptr, mid_ptr, N, BLOCK_SIZE: tl.constexpr)
    # Phase 1
def dot_kernel_reduce(mid_ptr, out_ptr, M, BLOCK_MID: tl.constexpr)
    # Phase 2
```

`nbxtensor.ops.dropout`

```python
def dropout_inference_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
    # Dropout at inference = identity copy.
```

`nbxtensor.ops.dtype_convert`

```python
def bf16_to_fp16_kernel(src_ptr, dst_ptr, N, BLOCK: tl.constexpr)
    # Convert bf16 → fp16 on GPU via fp32 intermediate.
```

`nbxtensor.ops.elementwise_launch`

```python
def ew_launch_kernel(x_ptr, y_ptr, n_elements, N_BUCKET, BLOCK_SIZE: tl.constexpr)
    # y = x over n_elements
def ew_meta(n_elements: int)
    # The tile and warps of an element-wise launch of `n_elements`
def ew_grid(n_elements: int, meta: dict)
    # The 1-D grid of an element-wise launch at its configuration.
```

`nbxtensor.ops.elu`

```python
def elu_forward_kernel(input_ptr, output_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.embedding`

```python
def embedding_kernel(output_ptr, indices_ptr, weight_ptr, V, fault_ptr, FAULT_CODE: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr)
    # Embedding forward
```

`nbxtensor.ops.eq`

```python
def eq_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
def eq_scalar_kernel(x_ptr, output_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.erf`

```python
def erf_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.exp`

```python
def exp_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.exp2`

```python
def exp2_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.fft_op`

```python
def bit_reverse_rows_kernel(real_in, imag_in, real_out, imag_out, n, total, log2n, BLOCK_SIZE: tl.constexpr)
    # Bit-reversal permutation of EVERY row of a [rows, n] pair in one launch
def fft_stage_rows_kernel(real_ptr, imag_ptr, n, pairs, stage, INVERSE: tl.constexpr, BLOCK_SIZE: tl.constexpr)
    # One butterfly stage of the Cooley-Tukey FFT on EVERY row of a [rows, n] pair, in place, one butterfly pair per lane (`pairs` = rows * n // 2 lanes).
def scale_kernel(real_ptr, imag_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr)
    # Scale all elements by 1/N after inverse FFT.
```

`nbxtensor.ops.fill_op`

```python
def fill_kernel(output_ptr, value, n_elements, BLOCK_SIZE: tl.constexpr)
    # Fill output with a scalar value.
```

`nbxtensor.ops.flash_attention`

```python
def flash_attention_forward_kernel(Q, K, V, Bias, Out, Lse, TMP, softmax_scale, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bm, stride_ob, stride_oh, stride_om, nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim, CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K, BIAS_TYPE: tl.constexpr, BLOCK_HEADDIM: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GQA_GROUPS: tl.constexpr, QK_CHUNK: tl.constexpr=0)
def flash_configs()
    # The tiles the certifier proves per key (workstream 1) where the profile declares no hand table of tiles (`launch_keys.flash_tiles_certified`)
```

`nbxtensor.ops.flash_attention_m8n8k4`

```python
def unit_flash_configs()
    # The tiles the certifier proves per key (workstream 1)
def flash_attention_m8n8k4_kernel(Q, K, V, Bias, Out, Lse, softmax_scale, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bm, stride_ob, stride_oh, stride_om, nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim, Q_BUCKET, K_BUCKET, MB: gl.constexpr, NS: gl.constexpr, DA: gl.constexpr, DB: gl.constexpr, GQA_GROUPS: gl.constexpr, HAS_BIAS: gl.constexpr, SPLIT: gl.constexpr, HI_EXP: gl.constexpr, LO_SHIFT: gl.constexpr, KSA: gl.constexpr, KSB: gl.constexpr)
def p_layout(MB: int, NS: int, NBO: int, WM: int)
    # P as [MB, 2, 8, 4, BN/4] = (mb, mi3, mi, k, j)
def staged_shared_bytes(BM: int, BN: int, DA: int, DB: int, split: bool)
    # The shared memory the kernel above allocates by name for one program
def head_parts(headdim: int)
    # The head dim as DA + DB, each a power of two >= 16 (DB may be 0)
def flash_layouts(MB: int, NS: int, DA: int, DB: int, WM: int)
    # Every layout the kernel takes, for MB 16-row bands over WM warps (WN = 1).
def flash_layout(name, MB, NS, DA, DB, WM)
    # One of the kernel's layouts (`flash_layouts`), evaluated at compile time from its constexprs.
```

`nbxtensor.ops.flash_decode`

```python
def flash_decode_split_kernel(q_ptr, k_ptr, v_ptr, bias_ptr, opart_ptr, mpart_ptr, lpart_ptr, T_k, seg_len, sm_scale, stride_qh, stride_qg, stride_kh, stride_kn, stride_oh, stride_os, stride_og, stride_mh, stride_ms, GROUPS: tl.constexpr, BLOCK_G: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr, HAS_BIAS: tl.constexpr)
def flash_decode_reduce_kernel(opart_ptr, mpart_ptr, lpart_ptr, out_ptr, stride_oh, stride_os, stride_og, stride_mh, stride_ms, stride_outh, stride_outg, SPLIT: tl.constexpr, GROUPS: tl.constexpr, BLOCK_G: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr)
    # Combine the split partials via log-sum-exp correction.
```

`nbxtensor.ops.flip_op`

```python
def flip_1d_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
    # Reverse a 1D contiguous tensor
def flip_strided_kernel(src_ptr, dst_ptr, n_elements, dim_size, stride_dim, stride_post, BLOCK_SIZE: tl.constexpr)
    # Flip along a single dimension of an N-D contiguous tensor.
```

`nbxtensor.ops.floor`

```python
def floor_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
def ceil_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
def round_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
def trunc_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.full_op`

```python
def full_kernel(output_ptr, fill_value, n_elements, BLOCK_SIZE: tl.constexpr)
    # Fill output tensor with a constant scalar value.
```

`nbxtensor.ops.fused_moe`

```python
def fused_moe_kernel(a_ptr, expert_ptrs_ptr, c_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr, N, K, EM, num_valid_tokens, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr, compute_type: tl.constexpr, TOPK_DIVIDE: tl.constexpr=True)
    # Grouped GEMM with absolute expert pointer table.
def fused_moe_wna16_kernel(a_ptr, qw_ptrs_ptr, sc_ptrs_ptr, mn_ptrs_ptr, c_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr, N, K, EM, num_valid_tokens, stride_am, stride_ak, stride_qk, stride_qn, stride_sg, stride_sn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr, compute_type: tl.constexpr, QGROUP: tl.constexpr, QPACK: tl.constexpr, TOPK_DIVIDE: tl.constexpr=True)
    # Grouped GEMM over int4-g128-asym experts — in-register dequant.
def fused_moe_fp32b_kernel(a_ptr, expert_ptrs_ptr, c_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr, N, K, EM, num_valid_tokens, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr, compute_type: tl.constexpr, TOPK_DIVIDE: tl.constexpr=True)
    # The wna16 kernel's BYTE ORACLE
def silu_and_mul_kernel(input_ptr, output_ptr, M, N, stride_im, stride_in, stride_om, stride_on, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # Fused SwiGLU
def silu_mul_split_kernel(gate_ptr, up_ptr, output_ptr, M, N, stride_gm, stride_gn, stride_um, stride_un, stride_om, stride_on, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # Fused SwiGLU with split gate/up tensors
```

`nbxtensor.ops.fused_upsample_conv`

```python
class BroadcastClonePyroxy
    # Sentinel returned by an intercepted aten.clone op whose input is a broadcast-view (stride=0 on some dim from a preceding aten.expand) and whose consumer chain is `aten.view -> aten.pixel_shuffle`.
def make_upsample_proxy_interceptor()
    # Return an interceptor that turns an upsample into a FusionUpsampleProxy (no compute, no allocation of the full upsampled tensor).
```

`nbxtensor.ops.gather_op`

```python
def gather_kernel(inp, index, out, N, inp_dim_stride, idx_stride_outer, idx_stride_dim, idx_stride_inner, inp_stride_outer, inp_stride_inner, out_stride_outer, out_stride_dim, out_stride_inner, outer_size, dim_size, inner_size, BLOCK_SIZE: tl.constexpr)
    # Gather kernel — works for any tensor rank via pre-computed strides.
```

`nbxtensor.ops.ge`

```python
def ge_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
def ge_scalar_kernel(x_ptr, output_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.gelu`

```python
def gelu_forward_kernel(input_ptr, output_ptr, n_elements, approximate: tl.constexpr, BLOCK_SIZE: tl.constexpr)
    # GELU forward.
```

`nbxtensor.ops.gemv_vec`

```python
def gemv_vec_kernel(A_ptr, B_ptr, C_ptr, N, K, N_BUCKET, K_BUCKET, stride_an, stride_am, stride_bm, stride_cn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr)
```

`nbxtensor.ops.glu`

```python
def glu_forward_kernel(input1_ptr, input2_ptr, output_ptr, size, act_func: tl.constexpr, BLOCK_SIZE: tl.constexpr)
    # GLU forward
```

`nbxtensor.ops.grid_sampler`

```python
def grid_sampler_2d_kernel(inp_ptr, grid_ptr, out_ptr, C, H, W, OUT_SPATIAL, stride_in, stride_ic, stride_ih, stride_iw, stride_gn, stride_gs, stride_gc, stride_on, stride_oc, stride_os, MODE: tl.constexpr, PAD_BORDER: tl.constexpr, ALIGN: tl.constexpr, BLOCK_C: tl.constexpr)
```

`nbxtensor.ops.group_norm_tiled`

```python
def group_norm_stats_kernel(X, Mean, Rstd, group_size, C, HW, num_groups, eps, BLOCK_SIZE: tl.constexpr)
def group_norm_apply_kernel(X, Y, W, B, Mean, Rstd, group_size, C, HW, num_groups, scale_by_weight: tl.constexpr, add_bias: tl.constexpr, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.groupnorm`

```python
def group_norm_forward_kernel(X, Y, W, B, Mean, Rstd, group_size, C, HW, num_groups, eps, PROG_BUCKET, HIDDEN_BUCKET, scale_by_weight: tl.constexpr, add_bias: tl.constexpr, BLOCK_SIZE: tl.constexpr)
    # One program handles one (batch, group) pair.
```

`nbxtensor.ops.gt`

```python
def gt_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
def gt_scalar_kernel(x_ptr, output_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.hardsigmoid`

```python
def hardsigmoid_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.hardswish`

```python
def hardswish_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.im2col`

```python
def im2col_kernel(inp_ptr, out_ptr, C, IH, IW, OW, L, KH: tl.constexpr, KW: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr, pad_h: tl.constexpr, pad_w: tl.constexpr, dil_h: tl.constexpr, dil_w: tl.constexpr, in_sn, in_sc, in_sh, in_sw, out_sn, out_sc, out_sl, BLOCK_L: tl.constexpr)
```

`nbxtensor.ops.index_add`

```python
def index_add_gather_kernel(x, src, index, out, dim_size, inp_shape_dim, inner_size, alpha, src_stride_outer, src_stride_dim, src_stride_inner, INNER_BLOCK: tl.constexpr)
    # One program == one (outer, dest, inner-tile) output cell.
```

`nbxtensor.ops.index_put_op`

```python
def index_put_kernel(out_ptr, idx_ptr, val_ptr, T, N, R, fault_ptr, FAULT_CODE: tl.constexpr, VAL_SCALAR: tl.constexpr, ACCUMULATE: tl.constexpr, BLOCK_SIZE: tl.constexpr)
    # out[idx[off // T], off % T] {=|+=} values[off] for off < N.
```

`nbxtensor.ops.index_select`

```python
def index_select_kernel(inp, out, M, N, index, index_len, fault_ptr, FAULT_CODE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # `debug=True` keeps the `tl.device_assert` below in the compiled kernel regardless of TRITON_DEBUG
def index_select_mid_kernel(inp, out, outer, N, inner, index, index_len, fault_ptr, FAULT_CODE: tl.constexpr, BLOCK: tl.constexpr)
    # Gather along a MIDDLE axis of a contiguous input read as (outer, N, inner)
```

`nbxtensor.ops.isfinite`

```python
def isfinite_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.isinf`

```python
def isinf_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.isnan`

```python
def isnan_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.layernorm`

```python
def layer_norm_forward_kernel(input_ptr, weight_ptr, bias_ptr, mean_ptr, inv_std_ptr, output_ptr, batch_dim, feat_dim, BATCH_BUCKET, FEAT_BUCKET, input_batch_stride, input_feat_stride, output_batch_stride, output_feat_stride, eps, scale_by_weight: tl.constexpr, add_bias: tl.constexpr, save_stats: tl.constexpr, BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr)
    # Layer normalization forward.
```

`nbxtensor.ops.le`

```python
def le_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
def le_scalar_kernel(x_ptr, output_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.leaky_relu`

```python
def leaky_relu_forward_kernel(input_ptr, output_ptr, n_elements, negative_slope, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.lerp`

```python
def lerp_tensor_forward_kernel(input_ptr, end_ptr, weight_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
    # Lerp with tensor weight
def lerp_scalar_head_kernel(input_ptr, end_ptr, output_ptr, n_elements, weight, BLOCK_SIZE: tl.constexpr)
    # Lerp with scalar weight < 0.5
def lerp_scalar_tail_kernel(input_ptr, end_ptr, output_ptr, n_elements, weight, BLOCK_SIZE: tl.constexpr)
    # Lerp with scalar weight >= 0.5
```

`nbxtensor.ops.linspace_op`

```python
def linspace_kernel(out_ptr, out_stride0, start, mid, end, step_size, steps, BLOCK_SIZE: tl.constexpr)
    # Generate evenly spaced values from start to end.
```

`nbxtensor.ops.log`

```python
def log_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.log_sigmoid`

```python
def log_sigmoid_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.logical_and`

```python
def logical_and_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.logical_not`

```python
def logical_not_forward_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.logical_or`

```python
def logical_or_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.logical_xor`

```python
def logical_xor_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
    # Logical XOR
```

`nbxtensor.ops.loss_wrappers`

```python
def mse_loss_wrapper(input, target, reduction: int=1)
    # MSE Loss.
def nll_loss_wrapper(input, target, weight=None, ignore_index: int=-100, reduction: int=1)
    # NLL Loss.
def cross_entropy_wrapper(input, target, weight=None, reduction: int=1)
    # Cross entropy loss
```

`nbxtensor.ops.lt`

```python
def lt_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
def lt_scalar_kernel(x_ptr, output_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.masked_fill`

```python
def masked_fill_forward_kernel(input_ptr, mask_ptr, output_ptr, n_elements, fill_value, BLOCK_SIZE: tl.constexpr)
    # out = fill_value where mask is True, else input.
```

`nbxtensor.ops.matmul`

```python
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, M_BUCKET, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, IEEE_PRECISION: tl.constexpr=False, PROMOTE_B: tl.constexpr=False, PROMOTE_A: tl.constexpr=False, EPILOGUE: tl.constexpr=0, BLOCK_M: tl.constexpr=64, BLOCK_N: tl.constexpr=64, BLOCK_K: tl.constexpr=32, GROUP_M: tl.constexpr=8)
    # C = A @ B where A is [M, K], B is [K, N], C is [M, N].
def addmm_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K, M_BUCKET, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, alpha, beta, IEEE_PRECISION: tl.constexpr=False, PROMOTE_B: tl.constexpr=False, PROMOTE_A: tl.constexpr=False, PROMOTE_BIAS: tl.constexpr=False, EPILOGUE: tl.constexpr=0, BLOCK_M: tl.constexpr=64, BLOCK_N: tl.constexpr=64, BLOCK_K: tl.constexpr=32, GROUP_M: tl.constexpr=8)
    # C = beta * bias + alpha * (A @ B) where bias is [N].
```

`nbxtensor.ops.matmul_m8n8k4`

```python
def matmul_m8n8k4_kernel(A, B, Bias, C, M, N, K, stride_az, stride_am, stride_ak, stride_bz, stride_bk, stride_bn, stride_cz, stride_cm, stride_cn, stride_xz, stride_xm, stride_xn, alpha, beta, HAS_BIAS: gl.constexpr, PROMOTE_BIAS: gl.constexpr, EPILOGUE: gl.constexpr, B_K_CONTIG: gl.constexpr, SPLIT_A: gl.constexpr, SPLIT_B: gl.constexpr, HI_EXP: gl.constexpr, LO_SHIFT: gl.constexpr, MB: gl.constexpr, NB: gl.constexpr, BK: gl.constexpr, GROUP_M: gl.constexpr, L: gl.constexpr, LA: gl.constexpr, LB: gl.constexpr, GA: gl.constexpr, GB: gl.constexpr)
def split_exponent(m, HI_EXP: gl.constexpr)
    # The exponent s of the power-of-two scale 2^s that brings a largest magnitude m (fp32) into [2^HI_EXP, 2^(HI_EXP+1))
def hi_lo(x, LO_SHIFT: gl.constexpr)
    # fp32 x (already scaled) as hi = fp16(x) and lo = fp16((x - hi) 2^LO_SHIFT)
def mm_layouts(MB: int, NB: int, WM: int, WN: int, BK: int, b_k_contig: bool)
    # Every layout the kernel takes for a tile of 16 MB x 16 NB x BK over WM x WN warps.
```

`nbxtensor.ops.max_pool2d`

```python
def max_pool2d_forward_kernel(input_ptr, output_ptr, indices_ptr, in_stride_n, in_stride_c, in_stride_h, in_stride_w, in_c, in_h, in_w, out_h, out_w, kernel_h: tl.constexpr, kernel_w: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr, padding_h: tl.constexpr, padding_w: tl.constexpr, dilation_h: tl.constexpr, dilation_w: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr)
```

`nbxtensor.ops.max_pool3d`

```python
def max_pool3d_forward_kernel(input_ptr, output_ptr, indices_ptr, in_stride_n, in_stride_c, in_stride_t, in_stride_h, in_stride_w, in_c, in_t, in_h, in_w, out_t, out_h, out_w, kernel_t: tl.constexpr, kernel_h: tl.constexpr, kernel_w: tl.constexpr, stride_t: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr, padding_t: tl.constexpr, padding_h: tl.constexpr, padding_w: tl.constexpr, dilation_t: tl.constexpr, dilation_h: tl.constexpr, dilation_w: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr)
```

`nbxtensor.ops.max_reduce`

```python
def max_kernel_mid(inp, mid, M, BLOCK_SIZE: tl.constexpr)
    # Pass 1
def max_kernel_result(mid, out, mid_size, BLOCK_MID: tl.constexpr)
    # Pass 2
def max_kernel(inp, out_value, out_index, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # Max reduction along compressed inner dim N.
```

`nbxtensor.ops.maximum`

```python
def maximum_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.mean`

```python
def mean_forward_kernel(input_ptr, output_ptr, batch_dim, feat_dim, BATCH_BUCKET, FEAT_BUCKET, input_batch_stride, input_feat_stride, BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr)
    # Mean reduction over the last dimension
```

`nbxtensor.ops.min_reduce`

```python
def min_kernel_mid(inp, mid, M, BLOCK_SIZE: tl.constexpr)
    # Pass 1
def min_kernel_result(mid, out, mid_size, BLOCK_MID: tl.constexpr)
    # Pass 2
def min_kernel(inp, out_value, out_index, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # Min reduction along compressed inner dim N.
```

`nbxtensor.ops.minimum`

```python
def minimum_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.mish`

```python
def mish_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.mma_m8n8k4`

```python
def acc_layout(MB: int, NB: int, WM: int, WN: int)
    # The [MB, 2, 8, NB, 2, 2, 4] accumulator over WM x WN warps (WM | MB, WN | NB, all powers of two).
def frag_a_layout(MB: int, NB: int, WM: int, WN: int)
    # One k-step of the A operand, [16 MB, 4] = (row, k), in the registers acc_layout(MB, NB, WM, WN) needs.
def frag_b_layout(MB: int, NB: int, WM: int, WN: int)
    # One k-step of the B operand, [16 NB, 4] = (column, k), in the registers acc_layout(MB, NB, WM, WN) needs.
def acc_rows_cols(L: gl.constexpr, MB: gl.constexpr, NB: gl.constexpr)
    # Row and column index of every accumulator element, broadcastable over L ([.., 1, 1, 1, 1] / [1, 1, 1, ..]).
def row_layout_rows(L: gl.constexpr, MB: gl.constexpr)
    # Row index [MB, 2, 8] in the layout of a row statistic of L (L with its four column dims reduced).
def row_max(x)
def row_sum(x)
def bcast_rows(r, L: gl.constexpr)
    # A [MB, 2, 8] row statistic -> broadcastable over accumulator L (any NB
def stage_k_inner(smem, tile, R: gl.constexpr, KK: gl.constexpr)
    # tile [R, KK] whose k runs along its columns -> smem [(KK/4) R, 4] laid out [k-step, R, 4].
def stage_k_outer(smem, tile, KK: gl.constexpr, C: gl.constexpr)
    # tile [KK, C] whose k runs along its rows -> smem [(KK/4) C, 4] laid out [k-step, C, 4].
def mma_k4(sa, a_row0, sb, b_row0, acc, MB: gl.constexpr, NB: gl.constexpr, LA: gl.constexpr, LB: gl.constexpr, L: gl.constexpr)
    # acc += A[:, 4j:4j+4] @ B[4j:4j+4, :] for the k-step whose rows start at a_row0 / b_row0 of the staged operands (16 MB rows of sa, 16 NB rows of sb).
```

`nbxtensor.ops.moe_align`

```python
def moe_align_stage1_kernel(topk_ids_ptr, offsets_ptr, padded_ptr, num_post_pad_ptr, n, BS: tl.constexpr, BE: tl.constexpr, E: tl.constexpr, BT: tl.constexpr, BF: tl.constexpr)
    # Single program
def moe_align_stage2_kernel(offsets_ptr, padded_ptr, sorted_ids_ptr, expert_ids_ptr, n, max_total, BS: tl.constexpr, BE: tl.constexpr, E: tl.constexpr, BLK: tl.constexpr, BF: tl.constexpr)
    # Grid over output positions
def moe_align_stage3_kernel(topk_ids_ptr, offsets_ptr, sorted_ids_ptr, n, E: tl.constexpr, BLKT: tl.constexpr, BN: tl.constexpr)
    # Grid over tokens
```

`nbxtensor.ops.moe_decode_vec`

```python
def moe_gateup_vec_kernel(x_ptr, topk_ids_ptr, gate_qw_tab, gate_sc_tab, gate_mn_tab, up_qw_tab, up_sc_tab, up_mn_tab, h_ptr, K, N, stride_wk, stride_wn, stride_sg, stride_sn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr, PACK: tl.constexpr)
def moe_down_combine_vec_kernel(h_ptr, topk_ids_ptr, topk_w_ptr, down_qw_tab, down_sc_tab, down_mn_tab, out_ptr, Kd, N, stride_wk, stride_wn, stride_sg, stride_sn, TOP_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr, PACK: tl.constexpr)
    # The combine IS the kernel
def moe_down_split_vec_kernel(h_ptr, topk_ids_ptr, topk_w_ptr, down_qw_tab, down_sc_tab, down_mn_tab, part_ptr, Kd, N, stride_wk, stride_wn, stride_sg, stride_sn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr, PACK: tl.constexpr)
    # ADOPTED second pass (2026-08-25)
def moe_part_reduce_kernel(part_ptr, out_ptr, N, TOP_K: tl.constexpr, BLOCK_N: tl.constexpr)
    # Fixed-order sum of the TOP_K weighted partials — deterministic (the same contract as flash_decode_reduce
def moe_gateup_vec_half_kernel(x_ptr, topk_ids_ptr, gate_tab, up_tab, h_ptr, K, N, stride_wk, stride_wn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, W_BF16: tl.constexpr)
def moe_down_split_vec_half_kernel(h_ptr, topk_ids_ptr, topk_w_ptr, down_tab, part_ptr, Kd, N, stride_wk, stride_wn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, W_BF16: tl.constexpr)
```

`nbxtensor.ops.moe_device_residency`

```python
def expert_slot_assign_kernel(ids_ptr, slot_of_ptr, owner_ptr, stamp_ptr, clock_ptr, slot_out_ptr, miss_out_ptr, layer_base, S, TOP_K: tl.constexpr, BLOCK_S: tl.constexpr)
def expert_miss_copy_kernel(miss_ptr, slot_ptr, host_g, host_u, host_d, slot_g, slot_u, slot_d, layer_base, n_g, n_u, n_d, BLOCK: tl.constexpr, W_BF16: tl.constexpr)
```

`nbxtensor.ops.moe_wrappers`

```python
def invoke_fused_moe(hidden_states, expert_ptrs, output, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, N, K, stride_bk, stride_bn, top_k, mul_routed_weight=False, topk_divide=True)
    # Launch fused MoE grouped GEMM kernel (absolute pointer table).
def invoke_fused_moe_wna16(hidden_states, qw_ptrs, sc_ptrs, mn_ptrs, output, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, N, K, stride_qk, stride_qn, stride_sg, stride_sn, top_k, mul_routed_weight=False, topk_divide=True)
    # Launch the W4 fused MoE grouped GEMM (int4-g128-asym experts, in-register dequant — byte-gated vs fused_moe_fp32b_kernel).
```

`nbxtensor.ops.mse_loss`

```python
def mse_loss_partial_kernel(inp, target, mid, M, BLOCK_SIZE: tl.constexpr, reduction: tl.constexpr)
    # Pass 1
def mse_loss_reduce_kernel(mid, out, mid_size, BLOCK_MID: tl.constexpr)
    # Pass 2
def mse_loss_none_kernel(inp_ptr, target_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
    # Pointwise (no reduction)
```

`nbxtensor.ops.mul`

```python
def mul_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
    # out = x * y (tensor * tensor)
def mul_scalar_kernel(x_ptr, output_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr)
    # out = x * scalar
def mul_scalar_dev_kernel(x_ptr, s_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
    # out = x * s where s is a 0-d DEVICE tensor.
```

`nbxtensor.ops.nan_to_num`

```python
def nan_to_num_forward_kernel(input_ptr, output_ptr, n_elements, nan_val, posinf_val, neginf_val, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.ne`

```python
def ne_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
def ne_scalar_kernel(x_ptr, output_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.neg`

```python
def neg_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.nllloss`

```python
def nll_loss_forward_kernel(inp_ptr, tgt_ptr, wgt_ptr, out_ptr, ignore_index, N, C, reduction: tl.constexpr, BLOCK_N: tl.constexpr)
    # NLL Loss forward for 1D/2D input [N, C].
def nll_loss_backward_kernel(out_grad_ptr, tgt_ptr, wgt_ptr, inp_grad_ptr, ignore_index, total_weight, N, C, reduction: tl.constexpr, BLOCK_N: tl.constexpr)
    # NLL Loss backward for 1D/2D input [N, C].
```

`nbxtensor.ops.nonzero`

```python
def nonzero_1d_kernel(mask_ptr, out_ptr, count_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.nonzero_op`

```python
def nonzero_kernel(inp, prefix_sum, out, n_elements, shape, ndim: tl.constexpr, BLOCK_SIZE: tl.constexpr)
    # Write indices of non-zero elements into output tensor.
```

`nbxtensor.ops.ones_op`

```python
def ones_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
    # Fill output with ones.
```

`nbxtensor.ops.oracles.conv2d_fp64`

```python
def conv2d_reference(x: np.ndarray, w: np.ndarray, *, stride: Sequence[int], padding: Sequence[int], dilation: Sequence[int], groups: int)
    # `[N, Cin, H, W]` ⊛ `[Cout, Cin//groups, KH, KW]` → `[N, Cout, Ho, Wo]`, fp64.
def receptive_slab(h: int, wd: int, k_h: int, k_w: int, *, stride, padding, dilation, window)
    # The input slab an output window `(r0, r1, c0, c1)` reads, and the padding it still needs.
def conv2d_reference_window(x, w: np.ndarray, *, stride: Sequence[int], padding: Sequence[int], dilation: Sequence[int], groups: int, window=None, shape: Optional[Sequence[int]]=None, slab_of=None)
    # `conv2d_reference` on one output block only
def conv2d_rows(named: dict, slab_of, weight_f64, rows)
    # The reference for the flat output rows `rows = (r0, r1)` of `conv2d_forward_kernel`, the output read as `[N*Cout*Ho, Wo]` — the form the runtime screen windows every kernel in.
def depthwise_rows(named: dict, slab_of, weight_f64, rows)
    # The same for `depthwise_conv2d_kernel` (`groups == C`, weight `[C, KH, KW]`).
class RowOracle(NamedTuple)
    # One kernel's row-windowed reference
def oracle_row_bytes(kernel: str, named: dict)
    # What one flat output row costs `kernel`'s reference, or None for a kernel this table does not window.
def depthwise_reference(x: np.ndarray, w: np.ndarray, *, stride: Sequence[int], padding: Sequence[int])
    # `[N, C, H, W]` ⊛ `[C, KH, KW]` → `[N, C, Ho, Wo]`, fp64.
def conv2d_from_args(named: dict, to_f64)
    # The reference output for `conv2d_forward_kernel`, or None.
def depthwise_from_args(named: dict, to_f64)
    # The reference output for `depthwise_conv2d_kernel`, or None.
```

`nbxtensor.ops.pad_op`

```python
def constant_pad_2d_kernel(x_ptr, output_ptr, total, in_h, in_w, out_h, out_w, channels_times_batch, x_stride_n, x_stride_c, x_stride_h, x_stride_w, out_stride_n, out_stride_c, out_stride_h, out_stride_w, pad_left, pad_top, pad_value, BLOCK_SIZE: tl.constexpr, WIDE: tl.constexpr=True)
    # Constant pad for 4D tensors (N,C,H,W).
def constant_pad_1d_kernel(x_ptr, output_ptr, total, in_size, out_size, batch_size, pad_left, pad_value, BLOCK_SIZE: tl.constexpr, WIDE: tl.constexpr=True)
    # Constant pad for 2D/3D tensors along last dim.
def constant_pad_nd_kernel(x_ptr, output_ptr, total, in0, in1, in2, in3, in4, out0, out1, out2, out3, out4, xs0, xs1, xs2, xs3, xs4, pb0, pb1, pb2, pb3, pb4, pad_value, BLOCK_SIZE: tl.constexpr, WIDE: tl.constexpr=True)
    # Constant pad of up to 5 dims in ONE pass
def replication_pad_nd_kernel(x_ptr, output_ptr, total, in0, in1, in2, in3, in4, out0, out1, out2, out3, out4, xs0, xs1, xs2, xs3, xs4, pb0, pb1, pb2, pb3, pb4, BLOCK_SIZE: tl.constexpr, WIDE: tl.constexpr=True)
    # Replication pad of up to 5 dims in ONE pass (PyTorch `replicate`)
```

`nbxtensor.ops.paged_decode_attn`

```python
def paged_kv_append_kernel(new_ptr, pool_ptr, slot_ptr, page_tokens, stride_nt, stride_nh, stride_pp, stride_ph, stride_ps, BLOCK_D: tl.constexpr, D: tl.constexpr)
    # One program per (new token, KV head)
def paged_decode_attn_split_kernel(q_ptr, k_pool_ptr, v_pool_ptr, block_table_ptr, seq_lens_ptr, opart_ptr, mpart_ptr, lpart_ptr, n_seqs, page_tokens, sm_scale, H_Q, stride_qh, stride_kp, stride_kh, stride_ks, stride_vp, stride_vh, stride_vs, stride_bt, stride_oh, stride_os, stride_mh, GQA_GROUPS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr, D_V: tl.constexpr, Q_TO: tl.constexpr=None, Q_SATURATE: tl.constexpr=False)
def paged_kv_append_step_kernel(k_new_ptr, v_new_ptr, k_pool_ptr, v_pool_ptr, block_table_ptr, lens_ptr, page_tokens, stride_kb, stride_kh, stride_vb, stride_vh, stride_kp, stride_kph, stride_ks, stride_vp, stride_vph, stride_vs, stride_bt, BLOCK_D: tl.constexpr, D_K: tl.constexpr, D_V: tl.constexpr)
    # A decode step's append with no host input
def paged_lens_advance_kernel(lens_ptr, n)
    # lens[b] += n for every sequence
```

`nbxtensor.ops.pixel_shuffle`

```python
def pixel_shuffle_kernel(input_ptr, output_ptr, n_elements, C_out, H, W, r, OH, OW, stride_in_n, stride_in_c, stride_in_h, stride_in_w, stride_out_n, stride_out_c, stride_out_h, stride_out_w, BLOCK_SIZE: tl.constexpr)
    # Pixel shuffle
def pixel_shuffle_broadcast_aware_kernel(bcast_view_ptr, output_ptr, n_elements, C_out, OH, OW, r, bcast, stride_v_n, stride_v_c, stride_v_b, stride_v_h, stride_v_w, stride_out_n, stride_out_c, stride_out_h, stride_out_w, BLOCK_SIZE: tl.constexpr)
    # Pixel shuffle that reads through an unmaterialized NBX broadcast view.
```

`nbxtensor.ops.pixel_unshuffle`

```python
def pixel_unshuffle_kernel(input_ptr, output_ptr, n_elements, C, IH, IW, r, C_out, H, W, stride_in_n, stride_in_c, stride_in_h, stride_in_w, stride_out_n, stride_out_c, stride_out_h, stride_out_w, BLOCK_SIZE: tl.constexpr)
    # Pixel unshuffle
```

`nbxtensor.ops.pow`

```python
def pow_forward_kernel(input_ptr, output_ptr, n_elements, exponent, BLOCK_SIZE: tl.constexpr, INT_EXP: tl.constexpr=0)
```

`nbxtensor.ops.prod`

```python
def reduce_mul(a, b)
def prod_kernel_mid(inp, mid, M, BLOCK_SIZE: tl.constexpr)
    # Pass 1
def prod_kernel_result(mid, out, mid_size, BLOCK_MID: tl.constexpr)
    # Pass 2
def prod_kernel(inp, out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # Reduce product along the compressed inner dimension N.
```

`nbxtensor.ops.rand_op`

```python
def rand_kernel(output_ptr, n_elements, seed, BLOCK_SIZE: tl.constexpr)
    # Fill output with uniform random values in [0, 1).
def randn_kernel(output_ptr, n_elements, seed, BLOCK_SIZE: tl.constexpr)
    # Fill output with standard normal random values N(0, 1).
```

`nbxtensor.ops.random_wrappers`

```python
def dropout_wrapper(x, p: float=0.5, train: bool=False)
    # Dropout at inference = identity.
def rand_wrapper(*args, **kwargs)
    # rand(size, ...) → uniform [0, 1) via Triton kernel.
def randn_wrapper(*args, **kwargs)
    # randn(size, ...) → standard normal N(0,1) via Triton kernel.
def rand_like_wrapper(x, **kwargs)
    # rand_like(tensor) → uniform [0, 1) same shape/dtype/device.
def randn_like_wrapper(x, **kwargs)
    # randn_like(tensor) → standard normal N(0,1) same shape/dtype/device.
def uniform_wrapper(x, low: float=0.0, high: float=1.0, **kwargs)
    # uniform_(tensor, low, high) → fill with uniform [low, high).
def multinomial_wrapper(x, num_samples: int, replacement: bool=False, **kwargs)
    # multinomial(probs, num_samples) → sampled indices.
```

`nbxtensor.ops.reciprocal`

```python
def reciprocal_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.relu`

```python
def relu_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.remainder`

```python
def remainder_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
def remainder_scalar_kernel(x_ptr, output_ptr, n_elements, divisor, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.repeat_interleave`

```python
def repeat_interleave_tensor_kernel(repeats_ptr, cumsum_ptr, out_ptr, size, BLOCK_SIZE: tl.constexpr)
    # Write source indices for tensor-based repeat_interleave.
```

`nbxtensor.ops.repeat_op`

```python
def repeat_1d_kernel(in_ptr, out_ptr, in_stride0, out_stride0, out_shape0, in_shape0, num_tasks, BLOCK_SIZE: tl.constexpr)
    # Repeat kernel for 1D tensors.
def repeat_2d_kernel(in_ptr, out_ptr, in_stride0, in_stride1, out_stride0, out_stride1, out_shape0, out_shape1, in_shape0, in_shape1, num_tasks, BLOCK_SIZE: tl.constexpr)
    # Repeat kernel for 2D tensors.
def repeat_3d_kernel(in_ptr, out_ptr, in_stride0, in_stride1, in_stride2, out_stride0, out_stride1, out_stride2, out_shape0, out_shape1, out_shape2, in_shape0, in_shape1, in_shape2, num_tasks, BLOCK_SIZE: tl.constexpr)
    # Repeat kernel for 3D tensors.
def repeat_4d_kernel(in_ptr, out_ptr, in_stride0, in_stride1, in_stride2, in_stride3, out_stride0, out_stride1, out_stride2, out_stride3, out_shape0, out_shape1, out_shape2, out_shape3, in_shape0, in_shape1, in_shape2, in_shape3, num_tasks, BLOCK_SIZE: tl.constexpr)
    # Repeat kernel for 4D tensors.
```

`nbxtensor.ops.residual_chain`

```python
class ChainSentinel
    # Zero-allocation sentinel returned by chain intermediate interceptors.
def resolve_chain_weights(chain_spec: Dict[str, Any], dag: Dict[str, Any], weights: Dict[str, Any])
    # Look up the weight tensors each chain op needs.
def band_streamed_chain_nbx(t_base: 'Any', chain_weights: Dict[str, Any], tile_factor: int, halo: int)
    # A residual chain run in bands over its spatial axis, on NBXTensor (torch-free).
```

`nbxtensor.ops.rmsnorm`

```python
def rms_norm_forward_kernel(input_ptr, weight_ptr, output_ptr, batch_dim, feat_dim, BATCH_BUCKET, FEAT_BUCKET, input_batch_stride, input_feat_stride, output_batch_stride, output_feat_stride, eps, scale_by_weight: tl.constexpr, BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr, SATURATE_F16: tl.constexpr=False)
    # RMS normalization forward.
```

`nbxtensor.ops.rope`

```python
def rope_forward_kernel(q_ptr, q_row_stride, k_ptr, k_row_stride, cos, cos_row_stride, sin, sin_row_stride, sl, bs: tl.constexpr, cos_bs: tl.constexpr, n_qh: tl.constexpr, n_kh: tl.constexpr, hd: tl.constexpr, pad_n_qh: tl.constexpr, pad_n_kh: tl.constexpr, pad_hd: tl.constexpr, BLOCK_SIZE: tl.constexpr, BACKWARD_PASS: tl.constexpr=False)
    # Fused RoPE kernel — applies rotary embedding in-place to Q and K.
```

`nbxtensor.ops.rsqrt`

```python
def rsqrt_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.scaled_softmax`

```python
def scaled_softmax_kernel(output_ptr, input_ptr, scale_factor, query_seq_len, key_seq_len, stride_b, stride_h, stride_q, BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr)
    # Scaled softmax
```

`nbxtensor.ops.scatter_op`

```python
def scatter_kernel(src, index, out, N, out_dim_stride, src_stride_outer, src_stride_dim, src_stride_inner, idx_stride_outer, idx_stride_dim, idx_stride_inner, out_stride_outer, out_stride_inner, outer_size, dim_size, inner_size, BLOCK_SIZE: tl.constexpr)
    # Scatter
def scatter_add_kernel(src, index, out, N, out_dim_stride, src_stride_outer, src_stride_dim, src_stride_inner, idx_stride_outer, idx_stride_dim, idx_stride_inner, out_stride_outer, out_stride_inner, outer_size, dim_size, inner_size, BLOCK_SIZE: tl.constexpr)
    # Scatter add
def scatter_reduce_amax_kernel(src, index, out, N, out_dim_stride, src_stride_outer, src_stride_dim, src_stride_inner, idx_stride_outer, idx_stride_dim, idx_stride_inner, out_stride_outer, out_stride_inner, outer_size, dim_size, inner_size, BLOCK_SIZE: tl.constexpr)
    # scatter_reduce with reduce='amax'
def scatter_reduce_amin_kernel(src, index, out, N, out_dim_stride, src_stride_outer, src_stride_dim, src_stride_inner, idx_stride_outer, idx_stride_dim, idx_stride_inner, out_stride_outer, out_stride_inner, outer_size, dim_size, inner_size, BLOCK_SIZE: tl.constexpr)
    # scatter_reduce with reduce='amin'
```

`nbxtensor.ops.screen_oracle`

```python
def oracle_row_bytes(tuner, meta)
    # What one flat output row costs THIS kernel's float64 reference on the host, or None for a kernel the row table does not window (the default
def fp64_footprint(buffers)
    # The host bytes a FULL reference would hold for these device buffers
def windowed_reference(tuner, meta, rows)
    # The fp64 reference for flat `rows` of the output, or None if this kernel cannot be windowed.
def provider(tuner, key, buffers, meta=None)
    # The correct contents of every screened buffer, or None.
def set_last_refusal(why: str)
def last_refusal()
def record_oracle_ms(key, ms: float)
def oracle_ms_of(key)
    # The oracle's measured CPU time for this key, or None if it never ran.
def announce_no_oracle(kernel_name: str, key, why: str='no oracle is registered for this kernel')
    # Say, once per process, that a screen ran on a consensus alone.
```

`nbxtensor.ops.selu`

```python
def selu_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.sigmoid`

```python
def sigmoid_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.silu`

```python
def silu_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.sin`

```python
def sin_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.softmax`

```python
def softmax_forward_kernel(input_ptr, output_ptr, batch_dim, feat_dim, BATCH_BUCKET, FEAT_BUCKET, input_batch_stride, input_feat_stride, output_batch_stride, output_feat_stride, log: tl.constexpr, BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr)
    # Softmax forward (or log_softmax if log=True).
def softmax_forward_looped_kernel(input_ptr, output_ptr, feat_dim, BATCH_BUCKET, FEAT_BUCKET, input_batch_stride, input_feat_stride, output_batch_stride, output_feat_stride, log: tl.constexpr, TILE_FEAT: tl.constexpr)
    # Online two-pass softmax for rows too large for one register tile.
```

`nbxtensor.ops.softplus`

```python
def softplus_forward_kernel(x_ptr, output_ptr, n_elements, beta, threshold, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.sort_op`

```python
def uint_to_uint(x, descending: tl.constexpr=False)
    # Unsigned → unsigned
def int_to_uint(x, descending: tl.constexpr=False)
    # Signed int → unsigned, preserving sort order.
def floating_to_uint(x, descending: tl.constexpr=False)
    # Float → unsigned, preserving sort order (handles sign bit correctly).
def convert_to_uint_preserve_order(x, descending: tl.constexpr=False)
    # Convert any numeric type to unsigned, preserving sort order.
def radix_sort_histogram_kernel(arr_ptr, out_ptr, num_passes, m, n, tiles_n_per_cta, TILE_N: tl.constexpr, TILE_R: tl.constexpr, num_bits_per_pass: tl.constexpr, descending: tl.constexpr)
    # Compute per-bin histograms for radix sort.
def radix_sort_tile_counts_kernel(arr_ptr, counts_ptr, m, N, OUT_N, TILE_N: tl.constexpr, TILE_R: tl.constexpr, k_bits: tl.constexpr, bit_offset, descending: tl.constexpr)
    # How many elements of each bin this TILE holds.
def radix_sort_tile_prefix_kernel(counts_ptr, prefix_ptr, num_tiles, OUT_N, TILES_POW2: tl.constexpr)
    # Exclusive scan of the per-tile counts, along the TILE axis.
def radix_sort_scatter_kernel(arr_ptr, associate_arr_ptr, out_ptr, associate_out_ptr, excumsum_bins_ptr, tile_prefix_ptr, n_passes, pass_id, bit_offset, m, N, OUT_N, TILE_N: tl.constexpr, TILE_R: tl.constexpr, k_bits: tl.constexpr, descending: tl.constexpr)
    # Scatter elements to their sorted positions.
```

`nbxtensor.ops.spectral_wrappers`

```python
def fft_c2c_wrapper(x, dim: int=-1, norm: str=None, forward: bool=True)
    # _fft_c2c
def fft_rfft_wrapper(x, n: int=None, dim: int=-1, norm: str=None)
    # rfft equivalent via Triton kernel.
def fft_irfft_wrapper(x, n: int=None, dim: int=-1, norm: str=None)
    # irfft equivalent via Triton kernel.
```

`nbxtensor.ops.sqrt`

```python
def sqrt_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.stack_op`

```python
def stack_copy_kernel(out_ptr, in_ptr_a, in_ptr_b, in_ptr_c, in_ptr_d, dim_size_in_a, dim_size_in_b, dim_size_in_c, dim_size_in_d, dim_size_out, dim_prod_post, dim_offset_a, dim_offset_b, dim_offset_c, dim_offset_d, total_elements_a, total_elements_b, total_elements_c, total_elements_d, BLOCK_X: tl.constexpr)
    # Copies up to 4 tensors into stacked output at correct offsets.
```

`nbxtensor.ops.std`

```python
def std_map_kernel(X, Tmp_sum, Tmp_sum_sq, N, BLOCK_N: tl.constexpr)
    # Pass 1
def std_reduce_kernel(Tmp_sum, Tmp_sum_sq, Out, N, correction, BLOCK_NUM, BLOCK_SIZE: tl.constexpr)
    # Pass 2
def std_dim_kernel(X, Out, stride_x_row, stride_x_col, M, N, correction, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # Compute std along compressed inner dim N.
```

`nbxtensor.ops.strided_copy`

```python
def strided_copy_kernel(src_ptr, dst_ptr, n_elements, shape_ptr, stride_ptr, BLOCK_SIZE: tl.constexpr, NDIM: tl.constexpr, WIDE: tl.constexpr=True)
    # Copy strided src → contiguous dst.
def strided_scatter_kernel(src_ptr, dst_ptr, n_elements, shape_ptr, stride_ptr, BLOCK_SIZE: tl.constexpr, NDIM: tl.constexpr, WIDE: tl.constexpr=True)
    # Scatter contiguous src → strided dst.
def strided_copy_nd_kernel(src_ptr, dst_ptr, n_elements, shape_ptr, src_stride_ptr, dst_stride_ptr, BLOCK_SIZE: tl.constexpr, NDIM: tl.constexpr, WIDE: tl.constexpr=True, SATURATE_F16: tl.constexpr=False)
    # Copy strided src → strided dst, casting on the store.
```

`nbxtensor.ops.sub`

```python
def sub_forward_kernel(x_ptr, y_ptr, output_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr)
    # out = x - alpha * y
def rsub_forward_kernel(x_ptr, output_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr)
    # out = scalar - x (reverse subtraction)
```

`nbxtensor.ops.sum`

```python
def sum_forward_kernel(input_ptr, output_ptr, batch_dim, feat_dim, BATCH_BUCKET, FEAT_BUCKET, input_batch_stride, input_feat_stride, BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr)
    # Sum reduction over the last dimension
```

`nbxtensor.ops.tan`

```python
def tan_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.tanh`

```python
def tanh_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.threshold`

```python
def threshold_forward_kernel(input_ptr, output_ptr, n_elements, threshold_val, value, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.topk`

```python
def topk_stage1_kernel(y_ptr, index_ptr, x_ptr, k, N: tl.constexpr, CHUNK_SIZE: tl.constexpr, DESCENDING: tl.constexpr)
def argsort(x, ids, dim: tl.constexpr, descending: core.constexpr)
def topk_stage2_kernel(y_ptr, index_ptr, chunk_x, chunk_index, sort_dim: tl.constexpr, k: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr, DESCENDING: tl.constexpr)
```

`nbxtensor.ops.trace_op`

```python
def trace_kernel(inp_ptr, out_ptr, num_diag, stride0, stride1, BLOCK_SIZE: tl.constexpr)
    # Sum diagonal elements of a 2D matrix.
```

`nbxtensor.ops.tril`

```python
def tril_kernel(X, Y, M, N, diagonal, M_BLOCK_SIZE: tl.constexpr, N_BLOCK_SIZE: tl.constexpr)
def tril_batch_kernel(X, Y, batch, MN, N, diagonal, BATCH_BLOCK_SIZE: tl.constexpr, MN_BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.triu`

```python
def triu_kernel(X, Y, M, N, diagonal, M_BLOCK_SIZE: tl.constexpr, N_BLOCK_SIZE: tl.constexpr)
def triu_batch_kernel(X, Y, batch, MN, N, diagonal, BATCH_BLOCK_SIZE: tl.constexpr, MN_BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.unfold_op`

```python
def unfold_backward_1d_kernel(grad_ptr, out_ptr, N_frames, size, step, L, MAX_OVERLAP: tl.constexpr, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.unique_op`

```python
def simple_unique_flat_kernel(sorted_data_ptr, sorted_indices_ptr, data_out_ptr, inverse_indices_ptr, idx_ptr, unique_size_ptr, return_inverse: tl.constexpr, return_counts: tl.constexpr, num_tasks, tile_size: tl.constexpr)
    # Single-block unique on pre-sorted data (for small inputs <= 8192).
def output_counts_kernel(idx_ptr, origin_num_tasks, counts_ptr, num_tasks, tile_size: tl.constexpr)
    # Compute counts from boundary indices.
```

`nbxtensor.ops.upsample_bilinear2d`

```python
def upsample_bilinear2d_kernel(input_ptr, output_ptr, N, C, IH, IW, OH, OW, scale_h, scale_w, offset, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr)
    # Bilinear interpolation upsampling.
```

`nbxtensor.ops.upsample_nearest2d`

```python
def upsample_nearest2d_kernel(ptr_o, ptr_i, N, C, OH, OW, IH, IW, s_n, s_c, s_h, s_w, reciprocal_scale_h, reciprocal_scale_w, BLOCK_SIZE: tl.constexpr)
```

`nbxtensor.ops.upsample_nearest3d`

```python
def upsample_nearest3d_kernel(ptr_o, ptr_i, N, C, OD, OH, OW, ID, IH, IW, reciprocal_scale_d, reciprocal_scale_h, reciprocal_scale_w, BLOCK_SIZE: tl.constexpr)
    # Nearest-neighbor 3D upsample.
```

`nbxtensor.ops.var`

```python
def welford_func(mean_x, count_x, M_x, mean_y, count_y, M_y)
    # Combine two Welford partial aggregates.
def var_kernel_1(X, Acc, Average, Count, N, BLOCK_N: tl.constexpr)
    # Pass 1
def var_kernel_2(Acc, Average, Count, Var, N, correction, BLOCK_NUM, BLOCK_N: tl.constexpr)
    # Pass 2
def var_welford_kernel(X, Var, M, N, correction, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # Welford variance along compressed inner dim N.
```

`nbxtensor.ops.vector_norm`

```python
def l2_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # L2 norm per row
def l1_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # L1 norm per row
def linf_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # L-infinity norm per row
def l0_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # L0 norm per row
def lp_norm_kernel(X, Out, M, N, ord, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # General Lp norm per row
def l2_norm_pass1_kernel(X, Mid, M, BLOCK_SIZE: tl.constexpr)
    # First pass of flat L2 norm
def l2_norm_pass2_kernel(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr)
    # Second pass
def linf_norm_pass1_kernel(X, Mid, M, BLOCK_SIZE: tl.constexpr)
    # First pass of flat Linf norm
def linf_norm_pass2_kernel(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr)
    # Second pass
```

`nbxtensor.ops.weight_norm`

```python
def weight_norm_kernel_first(output_ptr, norm_ptr, v_ptr, g_ptr, M, N, eps, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # Weight norm forward, normalizing along dim=0 (rows).
def weight_norm_kernel_last(output_ptr, norm_ptr, v_ptr, g_ptr, M, N, eps, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    # Weight norm forward, normalizing along last dim (columns).
```

`nbxtensor.ops.where`

```python
def where_forward_kernel(cond_ptr, x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
    # out = where(cond, x, y)
```

`nbxtensor.ops.wrappers`

```python
def has_native_bf16()
    # Expose the cached hardware capability flag for non-kernel callers.
def get_hardware_profile()
    # The hardware profile given to set_hardware_profile, or None.
def get_compute_dtype()
    # Return the active per-component compute dtype, or None if not set.
def set_activations_fp16_safe(safe: bool)
    # Set the activations-fp16-safe opt-in flag for the active component.
def get_activations_fp16_safe()
    # Return the activations-fp16-safe flag for the active component.
```

`nbxtensor.ops.zeros_op`

```python
def zeros_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
    # Fill output with zeros.
```

### `nbxtensor.backend.launcher_contract` — the contract a launch driver meets (BACKENDS.md section 1)

`nbxtensor.backend.launcher_contract`

```python
class ArgSlot
    # One argument the launcher must supply, in binding order.
class CompiledKernel(Protocol)
    # The result of compiling once, launched many times.
class LauncherDriver(Protocol)
    # What the launcher needs from a backend, and nothing more.
def verify_driver_contract(driver, jit_fn, signature, constexprs, make_buffer, read_buffer, free_buffer, grid, args_builder, expected)
    # Exercise a driver against the contract.
```

### `nbxtensor.quant` — quantized tensors

`nbxtensor.quant.quantized_tensor`

```python
class QuantizedTensor
    # int4-g128-asym weight triplet under one graph-facing handle.
def assemble_quantized(weights: dict)
    # Fold `.qweight`/`.scales`/`.qmins` triplets in a loaded weights dict into QuantizedTensor entries under the graph key `<base>.weight`.
```

### `nbxtensor.io.read_rate` — how fast the machine reads its weight files, by parallel reads

```python
PIECE = 16 << 20                                   # the loaders' read piece
def pieces(files, max_bytes) -> (list, int)        # (path, offset, bytes) pieces over `files`, and their total
def trial(pieces, workers) -> float                # bytes/s reading them with `workers` threads, the cache evicted first
def choose(rates, within) -> int                   # the smallest count within `within` of the best rate
def measure(files, max_bytes, max_workers, within=0.05) -> dict
    # {'rates': {1: .., 2: .., 4: ..}, 'workers': choose(...), 'steady': bool, 'bytes': .., 'last_one_worker': ..}
WITNESS_BYTES = 1 << 30                           # the witness written when the storage holds no weight file
START_BYTES = 2 << 30                             # the first trial: one shard's read, at most this
def workers_for(directory, max_workers=None, pattern='*/components/*/weights/*.safetensors') -> int
    # measured on the directory's storage (its largest weight files, else a witness it writes and removes):
    # the first trial reads one shard (at most START_BYTES = 2 GiB), then the trials grow while the measure is not stable; two steady draws decide (a third breaks a tie);
    # refused by name when no size is stable — never None, never a default
```

The cache is evicted before each trial and bypassed during it (macOS `msync(MS_INVALIDATE)` + `F_NOCACHE`, Linux
`posix_fadvise(DONTNEED)`): the rate is the storage path's, never the page cache's. A measure whose first and last
one-worker trials disagree is not steady and is not to be recorded.

### `nbxtensor.io.safetensors` — the host conversions of a loaded weight, split across the loader's workers

```python
def to_bf16_bits(arr: np.ndarray, io) -> np.ndarray   # float32 / float16 -> bfloat16 patterns (uint16), the
                                                      # bytes of one f32_to_bf16_bits call, one span per worker
```

`io = (workers, executor)` is the application's pool (DESIGN rule 14: its size is the profile's).

### `nbxtensor.dtype.policy` — precision by policy (DESIGN rule 12)

```python
@dataclass(frozen=True)
class Policy:                      # plain data: three dtype names, a flag, the op classes
    param: str = "float32"; compute: str = "float32"; output: str = "float32"
    float32_ops_half_io: bool = False
    classes: Tuple[Tuple[str, FrozenSet[str]], ...]          # default: policy.yml (PyTorch's autocast tables)
    def ops(self, cls: str) -> FrozenSet[str]                # an op class's names
    def op_class(self, op: str) -> Optional[str]             # "float32" / "lower" / "promote" / None
    def compute_dtype(self, op: str, input_dtypes: Sequence[str]) -> Optional[str]
    def fill_value(self, op: str, value, dtype: str)         # a fill scalar clamped to the dtype's finite range
    def replace(self, **changes) -> Policy; def as_dict(self) -> dict
def get_policy(spec: str, classes=None) -> Policy           # "params=float32,compute=float16,output=float32"
def default_classes() -> Dict[str, FrozenSet[str]]
CLASSES = ("float32", "lower", "promote", "float32_half_io", "scalar_fill", "creation_fill")
```

## 3. `nbxtensor.shape` — pure functions on integers (M2, Mac)

Each one is the code that runs today, moved and exported unchanged. Each is a pure function from ints to
ints, so a Rust port is mechanical.

```python
def numel(shape) -> int                                   # NBXTensor.numel; replaces triton/expert_store.py:42 and core/prism/work.py:56
def contiguous_strides(shape) -> tuple[int, ...]          # nbx_tensor.py:3161 _contiguous_strides
def is_contiguous(shape, strides) -> bool                 # nbx_tensor.py:3140 _is_dense_row_major (extent-1 dims carry any stride)
def view_strides(shape, strides, new_shape) -> tuple[int, ...] | None   # kernels/dispatch.py:156 _view_strides (ATen computeStride)
def broadcast_shapes(*shapes) -> tuple[int, ...]          # nbx_tensor.py:3170 _broadcast_shapes; refuses by name
def broadcast_strides(shape, strides, target) -> tuple[int, ...]        # the rule `expand` applies, as a function
def slice_bounds(start: int | None, stop: int | None, size: int) -> tuple[int, int]   # nbx_tensor.py:3197 (clamped, step 1 only)
def max_linear_offset(shape, strides) -> int              # nbx_tensor.py:2976
```

## 4. `nbxtensor.Tensor`

Read-only metadata, all of it plain data:

```python
t.shape: tuple[int, ...]     t.strides: tuple[int, ...]    t.offset: int        # elements
t.dtype: DType               t.device: Device              t.itemsize: int
t.numel: int                 t.nbytes: int                 t.is_contiguous: bool
t.data_ptr() -> int          t.__cuda_array_interface__    # kept (nbx_tensor.py:3514)
t.__dlpack__(stream=None)    t.__dlpack_device__()         # new: the only exchange protocol (rule 6)
```

`t.dtype` returns a `DType`, not a `triton.language` dtype. Today `.dtype` returns the Triton dtype
(nbx_tensor.py:3328) because the Triton JIT reads `arg.dtype` to specialize a kernel. In v1 every kernel
launch goes through the library's launcher, which already passes pointer integers and a signature
(triton/launcher_contract.py). The Triton dtype is then derived there, in `_tl_dtype(DType)`. This is a
precondition of M2; the Dell owns the launcher.

**Constructors** are module functions with an explicit device and dtype. There is no implicit current
device and no float32 default.

```python
nbxtensor.empty(shape, dtype: DType, device: Device) -> Tensor
nbxtensor.zeros(shape, dtype, device);  nbxtensor.ones(...);  nbxtensor.full(shape, value: float, dtype, device)
nbxtensor.empty_like(t, dtype: DType | None = None, device: Device | None = None)   # None: t's own
nbxtensor.from_numpy(a: np.ndarray, device: Device, dtype: DType | None = None)      # dtype given: a's bytes reinterpreted, sizes checked (bf16 from uint16 bits)
nbxtensor.to_numpy(t) -> np.ndarray      # bfloat16 comes back as uint16 bits; widen with convert.bf16_bits_to_f32
nbxtensor.from_dlpack(obj) -> Tensor     # any producer; the producer's library imports itself
nbxtensor.from_pointer(ptr: int, shape, strides, dtype, device, owner: object | None) -> Tensor   # NBXTensor.from_raw; `owner` keeps the memory alive
```

**Views.** No copy is made, and each view is a new Tensor sharing the storage. The rules they follow are
section 3's functions.

```python
view(t, shape)  reshape(t, shape)  permute(t, dims)  transpose(t, a, b)  expand(t, shape)
narrow(t, dim, start, length)  select(t, dim, index)  unsqueeze(t, dim)  squeeze(t, dim | None)
flatten(t, start, end)  unflatten(t, dim, sizes)  movedim(t, src, dst)  unfold(t, dim, size, step)
as_strided(t, shape, strides, offset)  view_as_real(t)  view_as_complex(t)
```

The same operations remain methods on `Tensor`, taking exactly these arguments: `t.view(shape)` takes a
tuple, not `*shape`. `t[...]` remains as sugar defined by `narrow` and `select` alone, with step 1 only, as
today.

**Copies, casts and transfers** are explicit verbs. The overloaded `.to(*args, **kwargs)` (nbx_tensor.py:4407)
is replaced:

```python
contiguous(t)        clone(t)        copy_(dst, src)
astype(t, dtype: DType) -> Tensor        # the device cast; narrowing to float16 saturates finite values (nbx_tensor.py:2821) — stated, not a flag
to_device(t, device: Device, *, pinned: bool = False, stream: int | None = None) -> Tensor   # to_cuda / to_cuda_async / to_cpu
pin(t) -> Tensor                         # pin_host
```

## 5. `nbxtensor.device` and capabilities (M2, Mac for Metal)

```python
nbxtensor.device.devices() -> list[Device]          # detection (nbx_tensor.py:2668 gpu_backend)
nbxtensor.device.capabilities(kind: str) -> Capabilities
@dataclass(frozen=True)
class Capabilities:          # one record per backend, today five dicts (nbx_tensor.py:345-490)
    traps_on_device_assert: bool; loads_pointers_from_memory: bool; memory_is_host_readable: bool
    has_fp64: bool; fa_min_tile: int; reduce_tile_max: int; reduce_per_thread: bool; threads_per_warp: int
nbxtensor.device.synchronize(device)    nbxtensor.device.mem_info(device) -> (free: int, total: int)
```

`set_backend_has_fp64` and `declare_backend_capability` (nbx_tensor.py:420,519) mutate module globals
today. In v1 the record is built once, when the backend is detected. A device's fp64 is a fact the
backend states (Metal: none). It is not something the engine sets afterwards.

**Metal note for `mem_info`.** `total` is `recommendedMaxWorkingSetSize` and `used` is the allocator's own
live count. `MTLDevice.currentAllocatedSize` is cumulative and cannot be read as live
(kernels/metal_device.py:381). The docstring must state this, so that no caller reads `total - free` as
the runtime's footprint.

### `nbxtensor.backend.cpu.pool` — every core (DESIGN rule 13)

```python
def measured_threads() -> int                     # Apple silicon: performance cores; else physical cores allowed
def threads() -> int                              # the set size, else the measurement
def set_threads(n: Optional[int]) -> None         # an application's size; None = the measurement
def numa_nodes() -> Dict[int, Tuple[int, ...]]    # Linux /sys/devices/system/node; {} elsewhere
def pieces(n: int, parts: int) -> List[Tuple[int, int]]          # contiguous, in order, sizes within one
def split_threshold(kind: str, probe=None) -> int                # set, or measured once (inline vs split)
def set_split_threshold(kind: str, n: Optional[int]) -> None
def parallel_for(n: int, run: Callable[[int, int], object], kind: str, probe=None) -> Sequence[object]
```

`nbxtensor.backend.cpu.blas` — the BLAS numpy calls, held to the pool's cores:

```python
def blas() -> Tuple[str, object]               # ("accelerate" | "openblas", the library); refused by name otherwise
def set_blas_threads(n: int) -> None           # OpenBLAS: the process's count; Accelerate: 1 = single-threaded here
def blas_threads_of_this_thread() -> Optional[int]   # 1 single-threaded, a count (OpenBLAS), None (Accelerate's own)
```

## 6. `nbxtensor.memory`

The allocator becomes one interface with one instance per device, not a class holding process-wide
dicts (`DeviceAllocator._cuda_live_bytes`, `_cuda_peak_bytes`, `_pool_free`, nbx_tensor.py:1040-2600).

```python
nbxtensor.memory.allocator(device: Device) -> Allocator
class Allocator:
    def allocate(self, nbytes: int, *, pooled: bool = True) -> int     # malloc_cuda; DeviceOOMError on refusal
    def free(self, ptr: int) -> None
    def stats(self) -> AllocatorStats    # live, peak, pool_cached, pool_peak, host_pinned, host_pinned_peak: plain ints
    def reset_peak(self) -> None
    def empty_pool(self) -> int          # empty_cache_pool
    def holds(self, ptr: int) -> bool
nbxtensor.memory.plan(buffers: Sequence[tuple[int, int, int]]) -> Plan   # (nbytes, first_use, last_use) -> offsets + peak (static planner, M3)
nbxtensor.memory.copy(dst: int, src: int, nbytes: int, kind: str, stream: int | None)  # memcpy / memcpy_async
class DeviceOOMError(RuntimeError): requested: int; device: Device; live: int; pool_cached: int; driver_free: int | None; driver_total: int | None
```

The replay `SlabAllocator`, the arena and the pool sit behind this interface. That is the Dell's M2 work;
this section only names the surface.

## 7. `nbxtensor.ops` (surface for M2/M3; the Dell owns the move)

```python
nbxtensor.ops.get(name: str) -> Op       # kernels/dispatch.py:1213 dispatch(op_type); unknown name: KeyError naming it
op(*tensors, **attrs) -> Tensor | tuple[Tensor, ...]   # attrs: plain values only
op.launch_keys(shapes, dtypes, attrs, device) -> list[tuple]   # launch_keys.py functions, pure
op.cost(shapes, dtypes, attrs, device) -> Cost                 # M3: outputs, workspace and scratch with lifetimes, bytes moved, flops
```

Process-wide state stays out of `ops`. Today the wrappers carry a compute dtype, an fp16-safe flag and a
hardware profile set per run (`wrappers.set_compute_dtype`, `set_hardware_profile`). A v1 op receives
every dtype it computes in as arguments. The compute dtype is the DtypeEngine's decision, so it stays in
the engine and is passed in.

## 8. `nbxtensor.random`

`nbxtensor.random.Generator(seed: int, device: Device)`, an explicit object with `normal(shape, dtype)` and
`uniform(shape, dtype)`. It replaces the run-scoped stream armed through module state
(kernels/rng_stream.py, rng_pin.py).

## 9. What v1 does not keep (rule 11), from today's code

| today | where | v1 |
|---|---|---|
| `.to(*args, **kwargs)`: a dtype, a device string or a tensor, guessed from `str()` | nbx_tensor.py:4407 | `astype`, `to_device` |
| `.dtype` returns a `triton.language` dtype | nbx_tensor.py:3328 | `DType`; Triton dtype derived inside the launcher |
| `.device` returns the tensor itself | nbx_tensor.py:3338 | `Device` |
| `parse_dtype(d)` takes any object through `str(d)`, `"torch."` stripped | nbx_tensor.py:310 | `dtype(name: str)`; torch spellings normalised at the engine's graph boundary |
| `canonical_name` accepts a torch dtype by its `__module__` | triton/itemsize.py `canonical_name` | the engine's mirror keeps it; the library takes `DType` or `str` |
| `empty(shape, dtype=None, device='cuda')`: float32 and the current device by default | nbx_tensor.py:3527 | dtype and device required |
| capability dicts mutated by `set_backend_has_fp64` / `declare_backend_capability` | nbx_tensor.py:420,519 | frozen `Capabilities` |
| allocator state in class-level dicts | nbx_tensor.py:1040 | `Allocator` per device |
| device fault channel in module globals (`_FAULT_MESSAGES`, `_FAULT_BUFFERS`) | nbx_tensor.py:586 | a field of the device runtime object (M2) |
| current device cached in `threading.local` | nbx_tensor.py:975 | explicit `Device` everywhere |
| `numpy_staging_dtype(dt)` falls back to float32 for any dtype it does not list | triton/dtype.py:529 (`.get(dtype, np.float32)`) | removed: `to_numpy` / `from_numpy` own the host mapping and refuse by name |
| torch hand-off (`nbx_to_torch`, `nbx_dtype_to_torch`, `set_torch_device`, `torch_device_str`) | nbx_tensor.py:326,693-800 | stays in the engine until DLPack replaces it (rule 6) |
| torch-style conveniences (`float()`, `half()`, `long()`, `type_as`, `new_zeros`, `item`, `size()`, `dim()`) | NBXTensor | engine compatibility only, not exported; they disappear with their callers |
| arithmetic dunders (`__add__`, `__mul__` …) launching kernels | NBXTensor | `nbxtensor.ops`; the dunders stay as sugar calling the op |

## 10. Open points for the supervisor

1. **The enum and the table disagree.** The table carries `uint16`, `uint32`, `uint64`, `float8_e4m3fn`,
   `float8_e5m2` and `tfloat32`, which the enum lacks. This proposal appends the storage dtypes to the enum.
   `tfloat32` is a compute format with no storage of its own, so it is not a `DType`.
2. **Policy mixed with mechanism in triton/dtype.py.** The AMP op sets and `TritonDtypeEngine` are policy
   and stay in the engine (rule 3). `matrix_unit_representation` (triton/dtype.py:312) is a device fact
   read as a plain dict and moves to `ops`. `contraction_accumulator_dtype` and `saturate_scalar` are
   mechanism and move to `dtype`.
3. **Dependency on the Dell's launcher.** The `.dtype` change (section 4) needs the launcher to stop
   reading the Triton dtype off the argument. It is listed as a precondition of the `Tensor` part of M2.

## 11. The boundary: every host service is generic in the library or goes back to the engine (Hocine 29/10 16:28)

nbxtensor is the alternative to torch, and only that: the published library holds nothing of NeuroBrix. M1 passed
every fact the library did not hold yet through `nbxtensor.host`: the application provided a service by name and the
library refused a missing one by name. That was right for the move. A library published that way does nothing alone.

Each service now takes one of two fates:

- **(a) GENERIC, in the library.** Either a fact nbxtensor measures on the device itself (capabilities, the
  allocator's envelope, rates), or an optional setting whose default comes from a measurement (an autotune store at
  the user's standard cache location, a memory limit the user MAY impose). A setting is plain data set through the
  public API (rule 11), never a callback into the application.
- **(b) BACK IN THE ENGINE**, when it is NeuroBrix's own (NeuroBrix profiles and vendor files, rungs, the
  certification census, the classification of traced graphs). The engine applies it by calling the public API; the
  library no longer knows it.

**Exit criterion.** `nbxtensor.host` disappears, or keeps only documented optional hooks; the README examples pass in
an environment where NeuroBrix is NOT installed, on CUDA and on Metal; a guard
(`tests/unit/nbxtensor_boundary/test_the_public_api_names_nothing_of_neurobrix.py`) refuses a NeuroBrix name
(vendor file, rung, census, NeuroBrix profile, memory class) in the public API, its list of today's offenders
shrinking only.

| service today | what it answers | fate | the library after | state, owner |
|---|---|---|---|---|
| `memory_state` | the host's free memory | (a) fact, measured | `nbxtensor.backend.host_memory` reads the operating system | **landed** (Mac, be3acf2b); the service goes |
| `host_shares_memory_with_device` | unified memory | (a) fact, measured | `backend.device_facts`: Metal `hasUnifiedMemory`, CUDA `cudaDevAttrIntegrated` | **landed** (Mac); the service goes |
| `profile_triton_has_fp64` | the device computes float64 | (a) fact | `backend.device_facts.kernels_carry_fp64()` — the backend's answer, no profile argument | **landed** (fact: Mac; the last reader, `ops.wrappers.set_hardware_profile`: Dell) — the service is gone; an application that states the capability otherwise declares it with `backend.capabilities.set_backend_has_fp64` (NeuroBrix: `triton.dtype.declare_hardware_profile`, from its vendor files) |
| `certified_dir` | where proven kernel settings live | (a) setting, default measured | default: the user's cache (`$XDG_CACHE_HOME`/`~/.cache` `/nbxtensor/certified/`), filled by the autotune; `nbxtensor.ops.autotune.set_store(path)` for another | default **landed** (Mac); the setter replaces the service (Mac) |
| `tiling_policy` | the split sizes the tiling rules read | (a) setting, default | the library's table (`ops/tiling.yml`); `nbxtensor.ops.set_tiling(dict)` to replace values | table **landed** (Mac); the setter replaces the service (Mac) |
| `vendor_config`, `vendor_param`, `vendors_dir` | the kernel-relevant device description (tiles, shared memory, registers, warp size, matrix unit, dispatch limits, identification) | (a) fact, per device | `nbxtensor.backend.device_files`: the library's device files `backend/devices/<vendor>/<name>.yml` hold every key the census lists (`docs/nbxtensor/device_keys.txt`: static ast, string literals, a traced run); a device is found and described with no application. An application's own half of a device (`vendors_dir`, now OPTIONAL) is merged, a key in both refused by name; `tools/split_vendor_files.py` splits a vendor's files and proves each half merges back to the original | **landed**: apple split (Mac, 21 files); `vendor_config` / `vendor_param` are no services — the library reads `device_files.describe` / `param` (both halves; NVIDIA and AMD read the application's half alone until their split, Dell); `vendors_dir` stays an OPTIONAL hook |
| `device_arch_string` | the key a device file is found by | (a) fact | `backend.device_names.arch_string` | **landed** (Mac, M2) |
| `imposed_rung_mb` | a memory ceiling imposed on the allocator | (a) setting, optional | `nbxtensor.memory.set_limit(bytes)` — a limit the user MAY impose (default: none, the device's own budget); the engine sets its rung through it (`core.prism.memory_budget.apply_imposed_rung`, at start when `NBX_PRISM_BUDGET_MB` is set) | **landed** (Dell, 0fb617eb); the service goes, the library reads no NeuroBrix variable |
| `deferred_drain_policy_of` | when deferred frees go back | (b) engine | the deferred dead-tensor queue is the engine's Triton hot loop (`triton.sequence`), not the library's: no library code read the policy but its own reader, which only the engine called — no setter is needed; the engine reads its vendor file itself (`core.prism.memory_budget.deferred_drain_policy`, refused by name with no profile declared, where the library's reader returned 2 GB in silence) | **landed** (Dell); the service and `ops.elementwise_wrappers.deferred_drain_policy` go |
| `attention_operand_dtypes`, `attention_operand_alignment` | the dtypes and alignment an attention's operands run at | (a) default + explicit arguments | default: the operands' common dtype by `dtype.promotion.binary_result`, the device's own alignment; the engine's DtypeEngine passes its decision as the op's arguments | Mac (dtype zone) with the Dell (attention wrappers) |
| `matrix_unit_representation`, `matrix_unit_flash_representation`, `matrix_unit_split_scale` | how the matrix unit carries a GEMM's or an attention's operands | (a) default + explicit arguments | default from the device file's matrix-unit section and the dtype table; the DtypeEngine's choice passed as arguments | Mac (dtype) with the Dell (linalg, attention wrappers) |
| `canonical_aten` | an op name's canonical ATen spelling | (b) engine | the library's ops have their own names; the traced graph's spelling is the engine's dispatch (`neurobrix.kernels.classification.kernel_for`; `ops.dispatch.dispatch` maps the library's own names only) | **landed** (Dell, 504de8d5); the service goes |
| `memory_class_gb`, `memory_class_from_profile_id` | the memory a certified entry is served for | (a) fact + (b) engine | the library's certified store keys entries by its OWN device's memory: `backend.device_facts.memory_gb(mb)` (the one rounding), `device_memory_gb(idx)` (declared, else the driver's total, read once), `declare_device_memory(idx, bytes)` for an application (a census shadow, a plan's profile — NeuroBrix: `triton.dtype.declare_hardware_profile`); store functions renamed `proof_memory_gb`, `entry_for_memory_gb`, `executing_memory_gb`, `covered_memory_gbs`, keyword `memory_gb=`; the profile-NAME parse stays in the engine (no committed proof needs it: 76 133 of 76 133 carry their device record) | **landed** (Dell); both services go |
| `census`, `census_table` | the certification census | (b) engine, through two documented optional hooks | `ops.launch_observer.set_launch_observer(obs)` — a dry run: while `obs.active()`, an autotuned kernel launches nothing, hands its key to `obs.record(tuned, key)` and calls `obs.shadow_run`, and a count read back from the device is taken from the op's contract; `ops.autotune_settings.set_miss_explainer(fn)` — an application's account of a key a confirmation run misses, appended to `KeyNotCertified`. NeuroBrix installs its census as both (`kernels/census.py`, `kernels/census_table.census_row`, moved out of the library) | **landed** (Dell); both services go |
| `installation_refusals` | NeuroBrix's workshop rule on where a backend build may live | (b) engine | none | Mac (its one reader is the Metal backend selection) |

A service still refused in the environment without NeuroBrix names its row here.

## 12. The rest of the public surface, module by module (dtype, shape, random, host, backends)

Every public name of these modules, with its signature and what it answers (the first line of its docstring);
the guard `tests/unit/nbxtensor_boundary/test_every_public_name_is_in_the_api.py` keeps this complete.

### `nbxtensor.backend.capabilities`

```python
def backend_has_fp64() -> bool   # Whether the backend's kernels handle float64 elements, as declared by
def backend_loads_pointers_from_memory() -> bool   # True where a kernel may dereference an address it LOADED from a tensor.
def backend_traps_on_device_assert() -> bool   # True where `tl.device_assert` actually stops the kernel.
def reduced_extent_max(num_warps: int) -> Optional[int]   # The largest reduced extent a program of `num_warps` stages on this backend, or None when its lowering bounds
```

### `nbxtensor.backend.cpu_backend`

```python
class TritonCPUNotInstalledError   # Raised when `--triton` is invoked on a CPU-only profile and the
def ensure_triton_cpu_or_raise() -> None   # Verify `triton-cpu` is installed when we're on a CPU-only profile.
def get_install_instructions() -> str   # Return the install instructions string. Useful for CLI `--help`
def is_cpu_triton_active() -> bool   # True iff triton-cpu has been selected for this process.
```

### `nbxtensor.backend.faults`

```python
def check_device_faults() -> None   # Raise if any kernel reported a broken contract since the last check.
def device_fault_buffer(device_idx: int) -> 'NBXTensor'   # This device's zeroed one-word status buffer.
def device_fault_code(message: str, *, armed: bool=None) -> int   # The code a kernel should be compiled with: its own where the backend
def device_fault_code_cached(message: str) -> int   # `device_fault_code` memoised per message — it runs at every launch of
def fault_channel(message: str, spare: 'NBXTensor') -> tuple   # `(pointer_argument, FAULT_CODE)` for a kernel that carries the channel.
def register_device_fault(message: str) -> int   # Reserve a fault code for `message`. Called at import by the kernel
```

### `nbxtensor.backend.host_memory`

```python
class MemoryState   # One reading of the machine, at one instant.
```

### `nbxtensor.backend.metal_backend`

```python
class BackendSelectionRefused   # This seam refused to choose a Metal backend, and said why.
class TritonMetalNotInstalledError   # Raised when `--triton` is invoked on an Apple GPU and no Metal Triton
class TritonMetalShaderCompilerMissingError   # Raised when the Metal backend IS installed but Apple's offline shader
def backend_compiler_class()   # The selected backend's Triton backend class — the object a driver patches.
def backend_fallback_markers()   # Phrases that mean the selected backend did not emit for the device.
def backend_module(name: str) -> str   # The top-level module Metal backend `name` is installed as (what an application inspects to judge where the
def backend_refusal_types() -> tuple   # The exception types a Metal backend raises to say 'I cannot compile this
def backend_target_name() -> str   # The Triton GPUTarget backend NAME the selected implementation answers to.
def ensure_metal_toolchain_selected() -> None   # Work around Xcode 26/27's broken default-toolchain delegation, once.
def ensure_triton_metal_or_raise() -> None   # Verify a Metal Triton backend is present before the triton path starts.
def is_apple_silicon() -> bool   # True on an Apple-Silicon Mac.
def is_backend_refusal(exc: BaseException) -> bool   # True when `exc` is a Metal backend saying it cannot compile this config —
def metal_shader_compiler_available() -> bool   # True when Apple's offline shader compiler (`xcrun metal`) can be run.
def metal_target()   # The Triton target for this machine, without activating a driver.
def nbx_driver_module() -> str   # The library's launcher driver module implementing the SELECTED backend's
def selected_metal_backend() -> str   # WHICH Metal backend this machine runs, decided by the PROFILE.
def triton_metal_available() -> bool   # Probe for a Metal Triton backend without importing it.
```

### `nbxtensor.backend.metal_device`

```python
class MetalRuntime   # One Metal device, presented as the runtime object the seam expects.
class MetalUnavailableError   # Metal, or its Python bindings, are not usable on this machine.
def declare_new_family_ownership() -> None   # Tell pyobjc that the `new*` Metal selectors return an already-retained object.
def metal_device_available() -> bool   # True when a Metal device can actually be opened.
def reset_runtime_for_tests() -> None   # Drop the cached runtime — and everything compiled against it.
```

### `nbxtensor.backend.replay_graph`

```python
class CapturedPlan   # One bucket's action list as an instantiated CUDA graph.
class GraphReplayError   # 
class SegmentedPlan   # A multi-card plan's action list as one CUDA graph per stretch of actions on one card (S3, 2026-10-28).
```

### `nbxtensor.backend.runtime`

```python
def device_label(tensor) -> str   # How a tensor's placement should READ in a message, e.g. "metal:0".
def device_memory()   # The memory primitives of the backend in force (`memory.device_memory.DeviceMemory`): the account
def invalidate_current_device_cache() -> None   # Drop the ensure_triton_device fast-path cache (this thread).
```

### `nbxtensor.backend.triton_ext_driver`

```python
class TritonExtDriver   # Loads a metallib and launches it the way triton-ext's own launcher does.
class pinned_addresses   # Pin the Metal wraps behind these tensors, so an address a kernel CAPTURES
def pinned_gpu_address(addr: int) -> int   # The address a SHADER may read at `addr`, valid while its pin is held.
```

### `nbxtensor.dtype.convert`

```python
def bf16_abs_max(bits: np.ndarray) -> float   # The largest magnitude among bfloat16 patterns, NaN ignored (an infinity is infinite): what a caller compares to
def bf16_bits(a: np.ndarray) -> np.ndarray   # The `uint16` patterns of a bfloat16 host carrier: the array itself when it already is `uint16`, a view when
```

### `nbxtensor.dtype.itemsize`

```python
def dtype_table() -> Dict[str, Any]   # The whole table as written (`dtypes`, `packed`, and every other section), a copy: the reader a client keeps of
def from_autotune_key(key: str) -> str   # The dtype name of a launch-key spelling ("fp16" -> "float16", "int1" -> "bool").
def itemsize_table() -> Dict[str, int]   # {name: bytes per element} over every dense dtype (a copy: the caller may not edit the table).
def known_dtypes() -> Tuple[str, ...]   # Every dense dtype the table carries, sorted.
```

### `nbxtensor.dtype.nbx_dtype`

```python
def dtype_size(dtype: NBXDtype) -> int   # 
```

### `nbxtensor.host`

```python
def need(name: str) -> Callable   # Service `name` as the application provided it; refused by name when none did.
def provide(name: str, fn: Callable) -> None   # The application provides service `name` (one of `SERVICES`). Providing it again replaces it.
```

### `nbxtensor.random.mt_randn`

```python
class MT19937   # The generator as torch's CPU `Generator` runs it: `manual_seed(s)` = `init_genrand(s & 0xffffffff)`.
def mt_randn(seed: int, shape: Iterable[int]) -> np.ndarray   # A fresh generator seeded with `seed`, its first `prod(shape)` normals reshaped — the vendor's
```

### `nbxtensor.random.rng_pin`

```python
def pinned_normal(shape)   # Standard-normal N(0,1) float32 array of the given shape.
def pinned_seed()   # Return the configured seed (int) or None when pinning is disabled.
def pinned_uniform(shape)   # Uniform [0,1) float32 array of the given shape.
```

### `nbxtensor.random.rng_stream`

```python
def active() -> bool   # 
def init_draw(shape)   # The run's next initial-noise draw as a float32 numpy array of `shape` — the vendor's CPU draw for
def next_normal(shape)   # The run stream's next standard normals as a float32 numpy array of `shape` (torch's CPU `randn` for the
def next_seed() -> int   # Next per-draw kernel seed: splitmix64 of (run_seed, counter), 31-bit.
def next_uniform(shape)   # The run stream's next [0, 1) uniforms as a float32 numpy array of `shape` (torch's CPU `rand`, continued).
def set_run_seed(seed)   # Arm (or disarm with None) the run stream; resets the draw counter and the init-draw generator.
```

### `nbxtensor.random.seeded_draw`

```python
class SeededDrawStream   # One seeded uniform stream; the draw ORDER is the R30 contract.
def apply_repetition_penalty(logits: np.ndarray, seen_ids, penalty: float) -> np.ndarray   # RepetitionPenaltyLogitsProcessor semantics on fp64 logits (in place).
def seeded_gaussian(seed: int, shape) -> np.ndarray   # Seeded fp32 standard-normal draw — the latent-init twin of
```

### `nbxtensor.shape`

```python
MAX_NDIM = ...   # (constant)
```

### `nbxtensor.backend.device_files` — the library's device files (API.md section 11)

```python
DEVICES: Path                                         # nbxtensor/backend/devices: <vendor>/<name>.yml
def names() -> Iterator[Tuple[str, str, Path]]        # every device described, by the library or the application
def read(vendor: str, name: str) -> dict              # the library's half of one device ({} when none)
def merge(library: dict, application: dict, where="") -> dict   # one description; a key in both refused
def merged(vendor: str, name: str) -> dict            # the library's half with the application's (vendors_dir)
def describe(vendor: str, name: str) -> dict          # the device whole, refused by name when nothing describes it
def param(vendor, name, section, key, default)        # one key of it (the per-op reader: merged once per process)
def in_force() -> dict                                # the device the launch target names ({} when none matches)
def for_arch(arch: str) -> dict                       # the device an arch string names, read without a target
def forget() -> None                                  # the next reader resolves the device in force again
```

### `nbxtensor.backend.launcher.arch`

```python
def arch() -> int                                     # the target's arch as an int (70 for sm_70): target().arch
def bind_target(compile_target) -> None               # the target stated without a driver (a derivation that launches nothing)
```

### `nbxtensor.backend.runtime` — the backend in force

```python
def gpu_backend() -> str          # "cuda", "hip" or "metal": the vendor runtime this process can load (or NBX_GPU_BACKEND)
def gpu_runtime()                 # that backend's runtime: Metal's runtime object, or the vendor library through its C ABI
def backend_table() -> dict       # the backend's entry-point name table (CUDA and HIP share one C ABI; Metal has its own)
def set_device(tensor) -> None    # bind the driver's current device to the tensor's before a launch
def device_memory()               # the memory primitives of the backend in force (memory.device_memory.DeviceMemory)
```
