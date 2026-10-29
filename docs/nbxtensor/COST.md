# nbxtensor — the cost of an op, declared (M3 design, prototype)

Written by the Mac agent on 29/10/2026 for the supervisor's order of 01:45, on branch `nbxtensor` 9ff939ab. It
implements rule 4 ("every op declares its cost; Prism asks instead of re-deriving") under rule 11 (plain data at
the edges). It has three parts:

- the declaration;
- the list of what Prism re-derives today, and what replaces each one;
- the prototype (`nbxtensor/src/nbxtensor/ops/_cost.py`) and its proof.

## 1. The declaration

```python
Spec(shape: tuple[int, ...], strides: tuple[int, ...], dtype: str)        # an operand, as data
Buffer(name: str, nbytes: int, first: int, last: int, returned: bool)    # one allocation of the op
Cost(buffers: tuple[Buffer, ...], bytes_moved: int, flops: int)
    .outputs       # the returned buffers (they outlive the op)
    .workspace     # the rest: casts, contiguous copies, accumulators, scores, folds
    .peak_bytes    # the most the op holds at once beyond its operands, outputs included
```

**Lifetimes** are steps of the op's own sequence. Each allocation takes the next step, and a buffer lives over
[`first`, `last`]. Today every transient lives until the op returns: the wrappers keep the cast, the contiguous
copy and the accumulator in local names, and Python releases those at the return. The declaration says so rather
than imagining an earlier free. A wrapper that releases a transient early (a `del`, a rebind) moves that buffer's
`last`, and the declaration says that too.

**Bytes moved** counts every byte read and written, copies included. **Flops** counts the arithmetic the kernel
performs, per element, as its source reads (stated in each function's docstring). Contractions use Prism's existing
rule, two per multiply-add (`core/prism/work.py op_flops`).

**One source of decisions.** A declaration reads the same decisions the kernel launch reads:

- the layout rules (`nbxtensor.shape`: contiguity, the view rule, broadcasting);
- the widths (`nbxtensor.dtype`);
- once `launch_keys` and the wrappers are in the library (the Dell's core move), the launch decisions themselves:
  the attention route, a reduction's split, the conv fold, the matrix-unit representation.

Until then, a decision the library does not hold yet is passed in by the caller. Examples: the binary result dtype
(`wrappers._wider_dtype`) and the DtypeEngine's output dtype. Each such argument is named in the function's
docstring and listed in §4.

**How Prism uses it.** The activation walk (`ActivationProfiler.estimate_peak_memory`) keeps its liveness, which is
the graph's: when a tensor is born and when it dies. At each op it replaces its in-op terms with the op's
declaration. The in-op peak becomes:

`live before the op + cost.peak_bytes`

computed with the op's resolved operand specs and the plan's widths. The walk's own outputs bookkeeping goes on
from `cost.outputs`. The time estimate reads `cost.flops` and `cost.bytes_moved` in place of `work.op_flops` and
the walk's byte sums.

## 2. What Prism re-derives today, and what replaces each

| Prism today | what it re-derives | replaced by (library) | Prism's call instead |
|---|---|---|---|
| `solver._arena_mb` | the bytes of a component's weight arena: weights times the device's cost multiplier, with no alignment | `nbxtensor.memory.arena.layout(weight specs, dtype) -> Layout(total_bytes, offsets)`, the very layout the loader allocates (`ComponentArena`: one buffer, 256-byte aligned per tensor; `triton/memory_pool.py`) | `layout(...).total_bytes` |
| `solver._arena_over_allocation` | whether that arena fits one allocation of the device | `nbxtensor.device.capabilities(device).max_allocation_bytes` (a measured device fact: Metal `MTLDevice.maxBufferLength`) | `layout(...).total_bytes <= caps.max_allocation_bytes`. The decision (refuse, stream) stays Prism's; both facts are the library's |
| `solver._store_arena_over_allocation` | the same for an expert store's dense part | the same `layout` over the store's dense tensors (the expert store's own list, `triton/expert_store.py`) | the same comparison |
| `solver._arena_door` | applies the two rows above over a plan | — (policy) | stays Prism's, reading the two facts above |
| `memory_estimator.get_dtype_bytes_per_element` | a dtype's width, plus "absent dtype = 4 bytes" | `nbxtensor.dtype.itemsize.itemsize` | `itemsize(name)`. The one absent-dtype case (`InputConfig()` names none) states its float32 at that call: `itemsize(ic.dtype or "float32")` |
| the 1.3 / per-device `triton_arena_activation_factor` | what the Triton allocator holds over the walked live set: the pool's parked blocks (measured on HAT-L x4, M4 Pro, 2026-10-28: live tensors 0.62x the walk, parked blocks 3 540 MB) | `nbxtensor.memory.simulate(events, allocator_policy) -> Simulation(peak_bytes, parked_bytes)`. It replays the walk's allocations and frees (outputs plus each op's declared buffers) through the allocator's own rules: the pool's exact-or-slack take-back (`alloc_pool_fit_slack_fraction`), the parked cap (`alloc_pool_parked_cap_fraction`), the flush on device growth (S4b). It is the same code the allocator runs, as a pure function of the event list | the component's peak = `simulate(walk events).peak_bytes`; the factor is deleted |
| the 0.92 whole-component fraction | an unpriced envelope: runtime growth, the drain queue, the pool, and the attention scores (Wan2.2 16g, 2026-10-28) | the terms it hid, each declared: the runtime growth (`device.runtime_growth_bytes`, measured by the library on every run, unified devices included); the pool (the simulation above); the attention scores (the `sdpa` declaration below). The drain queue stays the engine's (its execution mode's cliff) | the budget = the rung less the measured growth; the component's peak from the simulation; no fraction |
| `attention_scratch.scratch_bytes` (compiled) and the Dell's Triton scores pricing (`sdpa_route`) | the scores an attention writes, per route | `nbxtensor.ops.sdpa.cost(q, k, v, mask, device)`. It reads `launch_keys.sdpa_route` itself and declares, per route: math (fp32 scores plus softmax output, the two-buffer peak after the 28/10 fix, the q copy); chunked (one chunk's of those); flash (the log-sum-exp rows and the bias view) | `cost.peak_bytes` in the walk; `_attention_scratch_of` deleted |
| `profiler` in-op terms: operand casts (`op_casts`), the conv3d folds (`conv3d_one_shot_peak`), a band-tiled conv's band (`tiled_conv_band_bytes`), the per-op workspace table (`op_workspace`) | each op's transients, re-derived beside the kernel | each op's `cost(...).workspace` | the walk's `in_op += cost.peak_bytes - outputs already counted` |
| `work.op_flops` and the walk's byte sums | the time estimate's work | `cost.flops`, `cost.bytes_moved` | `time_estimate` reads them per op |

The compiled engine (torch's ATen kernels) is not nbxtensor's. Its terms stay Prism's, read from the vendor files
as today: `compiled_sdpa_full_scores`, `compiled_op_workspace`, `compiled_allocator_retention`. Rule 3: the
library declares its own ops only.

## 3. The prototype

`nbxtensor/src/nbxtensor/ops/_cost.py` declares the families whose kernels are in the library and whose wrappers
allocate in a few readable lines:

- the element-wise ops: unary, and binary with a dtype cast and broadcast views;
- `softmax`;
- `rms_norm`, including its in-place write into its own contiguous copy;
- `native_layer_norm`, with its float32 row statistics;
- `sum` and `mean`, with the float32 accumulator and its cast to the result dtype.

Each declaration mirrors its wrapper (NeuroBrix `kernels/wrappers.py`) allocation by allocation.

**Proof.** The declaration is held to the execution, not just to itself:

- `nbxtensor/tests/test_ops_cost.py`: card-less, on hand-derived figures.
- `tests/unit/kernels/test_an_op_allocates_what_it_declares.py`: each family runs once on the device with the
  allocator's pool emptied, after one warm-up call, and the allocator's own peak above its entry level must EQUAL
  the declared `peak_bytes`. Shapes are chosen so every buffer is a whole number of device pages.
- Result on the M4 Pro, Metal: 10 of 10 equal, among them a strided unary, a cast binary, softmax over the last and
  over the first dim, rms_norm contiguous and strided, layer_norm, and sum over both dims.
- Seen red: with the contiguous-copy rule removed from the declarations, 3 of 10 fail.

## 4. What the prototype still takes from its caller, and when that ends

| argument | today's owner | it becomes |
|---|---|---|
| `elementwise_binary(common_dtype=...)` | `wrappers._wider_dtype` (`_DTYPE_PRIORITY`) | `nbxtensor.dtype.promotion.binary_result(a, b)`, the wrapper's own rule moved with the core, in one place with `promoted` |
| `rms_norm(out_dtype=...)` | the DtypeEngine (policy, engine) | stays an argument: the output dtype is the engine's decision (rule 3) |
| `reduction(out_dtype=..., cast_to=...)` | aten's `dtype` keyword and `sum`'s int64 rule | the keyword stays an argument (the graph's); the int64 rule moves to `dtype.promotion` |
| launch decisions (the attention route, splits, folds) | `kernels/launch_keys.py` | read by `cost` from `nbxtensor.ops.launch_keys` once the core has moved |

**Next families, after the core move:** GEMM (`mm`, `addmm`, `bmm` with the matrix-unit split's hi/lo operands),
convolutions (the band and fold transients), `sdpa` (per route), the copies and casts (`copy_`, `_to_copy`),
gather/scatter/index, and the fused MoE op. Each lands with its device equality test, like the five above.
