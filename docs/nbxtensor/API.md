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

`nbxtensor.dtype.convert.saturate_scalar(value, dtype)`: a Python number clamped to a half float's finite range (the table's `max`); anything else unchanged.

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
```

### `nbxtensor.backend.launcher.arch`

```python
def arch() -> int                                     # the target's arch as an int (70 for sm_70): target().arch
```

### `nbxtensor.backend.runtime` — the backend in force

```python
def gpu_backend() -> str          # "cuda", "hip" or "metal": the vendor runtime this process can load (or NBX_GPU_BACKEND)
def gpu_runtime()                 # that backend's runtime: Metal's runtime object, or the vendor library through its C ABI
def backend_table() -> dict       # the backend's entry-point name table (CUDA and HIP share one C ABI; Metal has its own)
def set_device(tensor) -> None    # bind the driver's current device to the tensor's before a launch
def device_memory()               # the memory primitives of the backend in force (memory.device_memory.DeviceMemory)
```
