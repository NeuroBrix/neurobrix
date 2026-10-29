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
`core/dtype/itemsize.py` stays in the engine for the PyTorch branch: it is a mirror, held equal by test.

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
nbxtensor.device.devices() -> list[Device]          # detection (nbx_tensor.py:2668 _detect_gpu_backend)
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

## 11. The host services and what each becomes (M2/M3; supervisor, 29/10 02:50)

M1 passed every fact the library does not hold yet through `nbxtensor.host`. The application provides a service by
name, and the library refuses by name any service that is missing. That is a good boundary for the move. But a
library published that way does nothing on its own: a developer who runs `import nbxtensor` has no vendor files, no
certified store and no tiling policy.

The rule for M2/M3 is that each service becomes one of three things:

- **a fact the library owns or measures itself;**
- **an optional policy**, whose default comes from a measurement;
- **an optional hook**, only for what truly belongs to the application.

| service (today) | what it answers | becomes | how |
|---|---|---|---|
| `vendor_config`, `vendor_param`, `vendors_dir`, `device_arch_string` | the kernel-relevant device description: tiles, shared memory, registers, thread counts, the matrix unit | **fact, owned** | the kernel sections of the vendor files move into the library (`nbxtensor/backend/devices/<vendor>/<arch>.yml`); what the driver reports is read at backend start (Metal `maxThreadgroupMemoryLength`, `maxBufferLength`; CUDA device attributes). The engine's vendor files keep only engine policy (Prism's budgets) |
| `device_arch_string` | the key device files are found by | **fact, owned** | one function of the device's own name, in `backend` |
| `matrix_unit_representation`, `matrix_unit_flash_representation`, `matrix_unit_split_scale` | how the matrix unit carries a GEMM's or an attention's operands | **fact, owned** | mechanism plus the device file's matrix-unit section, both in the library |
| `profile_triton_has_fp64` | whether the device computes float64 | **fact, measured** | asked of the backend at start (a one-element fp64 kernel; Metal: none), kept in the frozen `Capabilities` |
| `host_shares_memory_with_device` | unified memory | **fact, measured** | Metal `hasUnifiedMemory`, CUDA `cudaDevAttrIntegrated` |
| `memory_state` | the host's free memory | **fact, measured** | the library reads the operating system (`sysctl` / `/proc/meminfo`); no NeuroBrix reader |
| `certified_dir` | where certified kernel settings live | **policy, optional** | default: the library's own per-device store in the user cache (`~/.cache/nbxtensor/<device>/`), filled by autotune at first use; an application may point it at a committed directory (NeuroBrix: its certified store) |
| `tiling_policy` | the split sizes the tiling rules read | **policy, optional** | default derived from the measured device limits (maximum buffer, shared memory); an application may override |
| `deferred_drain_policy_of` | when freed blocks are returned | **policy, optional** | default: the allocator's measured envelope (drain on a refused allocation); NeuroBrix keeps its cliff |
| `imposed_rung_mb` | a memory rung imposed on the allocator | **generic setting** (landed) | `nbxtensor.memory.set_limit(bytes, device=None)`; default: none (the physical card); NeuroBrix applies its `NBX_PRISM_BUDGET_MB` through it when it starts (`core.prism.memory_budget.apply_imposed_rung`) |
| `attention_operand_dtypes`, `attention_operand_alignment` | the dtypes an attention's operands are run at | **policy, optional** | default: the operands' common dtype by `dtype.promotion.binary_result`; NeuroBrix's DtypeEngine overrides |
| `canonical_aten` | a traced graph's ATen alias (`multiply` for `mul`) | **back in the engine** (landed) | `ops.dispatch.dispatch` maps the library's own op names only and asks for nothing; NeuroBrix resolves the alias first (`neurobrix.kernels.classification.kernel_for`) |
| `memory_class_gb`, `memory_class_from_profile_id` | NeuroBrix's memory classes of a card | **hook, optional** | only the census keys read them; absent, keys carry no class |
| `installation_refusals` | NeuroBrix's workshop rule on where a backend build may live | **hook, optional** | absent: no refusal |
| `census`, `census_table` | the launch census | **hook, optional** | already optional (`_NoCensus`) |

**Added to M3's exit criterion.** A test runs every README example in an environment where NeuroBrix is NOT
installed (a fresh venv holding only the nbxtensor wheel and its dependencies), on CUDA and on Metal. A service still
refused there names the row of this table that has not landed.
