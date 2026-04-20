# PixArt triton — arena inter-run corruption (April 2026, OPEN)

**Status**: open. Route A (periodic `_deferred` drain) landed as a prerequisite; this bug is orthogonal and blocks PixArt triton end-to-end.
**Affected models**: PixArt-Sigma-XL-2-1024-MS, PixArt-XL-2-1024-MS (alpha). Both native 1024 × 1024 green at `8ae49dd`.
**Blocks**: PixArt-Sigma/Alpha triton 3-gate validation (coherent image, `cos(native, triton) ≥ 0.95` on post-CFG latent step 0, CFG preservation).
**Follow-up chain**: depends on NeuroBrix arena/liveness inspection tooling; will likely also benefit Sana triton multi-step, Flex.1 triton, SANA-Video triton, upscaler tiling paths — any triton model that calls `run()` more than once per request.

## Baseline before Route A

PixArt-Sigma-XL-2-1024-MS triton, V100 32 GB, CFG batch=2, 4 scheduler steps:

```
OOM inside first transformer run() at aten._scaled_dot_product_efficient_attention::49
Requested alloc: 37 748 736 bytes
Live VRAM at crash: 30.97 GB  (30.85 GB of retained _deferred tensors)
Peak live allocs at crash: 947  (age distribution: min=6, median=1472, max=2942 events)
```

Allocation log (`NBX_MALLOC_TRACE`) aggregated at peak shows the top sites are all arena-output slots retained by `_deferred`:

```
10.30 GB  249 × 41.4 MB  kernels/wrappers.py:1410 addmm         (result tensor)
 8.34 GB  248 × 33.6 MB  kernels/wrappers.py:414  _prepare_binary  (result)
 3.63 GB   25 × 145 MB   kernels/wrappers.py:486  gelu
 1.85 GB  103 × 18 MB    kernels/wrappers.py:381  _prepare_binary  (result)
 1.85 GB   49 × 38 MB    kernels/wrappers.py:873  native_layer_norm
 1.85 GB   49 × 38 MB    kernels/wrappers.py:4456 SDPA workspace
 ... (tail persistent memory_pool + CFG engine state, < 1.5 GB total)
```

`age[min=6 ... max=2942]` — outputs from ops hundreds-to-thousands of events ago are still live. That is `_deferred` accumulating until end-of-run.

## What Route A does

`triton/sequence.py::_run_single_device` and `_run_multi_device` gain periodic drain of `_deferred` when bytes ≥ `NBX_DEFERRED_DRAIN_BYTES` (default 2 GB) OR count ≥ `NBX_DEFERRED_DRAIN_COUNT` (default 512). Each drain does the same sync-then-free the existing end-of-run drain does. Same correctness, bounded peak.

Empirical verification:

```
TinyLlama-1.1B-Chat-v1.0 --triton, drain count=8 (aggressive):
  198 drains across 15 decode tokens. Output coherent
  ("I'd be happy to help you..."). Sanity: drain in isolation is safe.

PixArt-Sigma --triton, drain 2 GB threshold:
  36 drains across 4 transformer calls (9 per call on average).
  Transformer completes 4 times with NO OOM.
  Peak VRAM during drain window stays ≤ ~18 GB.
```

But PixArt then crashes on a **separate** bug, described below.

## The inter-run bug

With Route A active and first-crash eliminated, PixArt triton advances past the first transformer call and crashes on the **third** `run()`:

### PixArt-Sigma (`CUDA_LAUNCH_BLOCKING=1`)

```
Failed at aten.clone::28 (aten::clone): Triton Error [CUDA]: an illegal
memory access was encountered
```

`aten.clone::28` reads `self.data_ptr()` and memcpy's it. Error 700 =
`cudaErrorIllegalAddress` — the pointer points into a `cudaFree`-d region.

### PixArt-Alpha

```
Failed at aten.cat::2 (aten::cat): 'NoneType' object has no attribute 'ndim'
```

`aten.cat::2` consumes `aten.sin::1::out_0` and `aten.cos::1::out_0`. In run 3,
one of these two arena slots returns `None` to the `cat` dispatch. `cat::2` is
op 44 in execution order — well past the first 43 ops that should have already
repopulated those slots.

Both crashes appear at slightly different ops but share the same signature:
**run 3 reads an arena state that run 2 left behind, and that state is corrupt.**

## Why native doesn't crash

Native uses PyTorch `torch.Tensor` backed by the PyTorch caching allocator.
When a tensor is dropped, its memory is retained in PyTorch's bucket pool —
`data_ptr()` remains a valid mapping until PyTorch gives the block to
something else. A dangling view in the arena will still read coherent memory
for many subsequent runs, so the bug is masked.

Triton uses `NBXTensor` backed by raw `cudaMalloc` / `cudaFree` (by design,
per the "zero torch in triton" rule). When `__del__` fires, the driver
reclaims the page. A dangling view's `data_ptr()` points at unmapped
memory and the next kernel faults. The bug is surfaced immediately.

**Conclusion**: the bug existed latently in the cross-run arena machinery
before Route A. Route A is a prerequisite for *exposing* it — without
Route A, PixArt OOMs before the third run, so we never see the symptom.

## Likely root causes (to be confirmed by Piste 1 in the next session)

1. `_base` chain break. A view created somewhere (maybe a `detach()`, a
   `contiguous()` fast-path, a slow-path tensor transfer in
   `_transfer_tensor`, a MoE scatter, or a custom-fused op output) loses
   its `_base=parent` attribute. Later, the parent is `kill_slot`'d and
   drained, leaving the orphan view dangling.

2. Cross-run liveness omission. `_compute_liveness` is computed per-run.
   If slot S is `kill_slot`'d in run N but run N+1's execution order
   reads S *before* any run-N+1 op writes to it, run N+1 sees None.
   This would require a slot to be treated as input-from-prior-run
   semantics the liveness analysis doesn't know about.

3. Protected-set omission. A slot that should be in the `protected =
   weights ∪ inputs ∪ outputs` set of `_compute_liveness` is missing —
   e.g., a buffer that is shared between pre-loop and loop (scheduler
   state, CFG conditional embedding cache, noise schedule table).

## Three candidate fix routes analysed

**Piste 1 — diagnose-first (chosen)**
Instrument the arena: dump every slot's `(data_ptr, shape, strides,
dtype, device_idx, pinned, _base chain to root)` at `bind_weights`,
after every `bind_inputs`, after every `gather_outputs`. Run PixArt-Alpha
with the dump, diff arena state at run 2 boundary vs run 3 entry. The
slot that is `None` on alpha, or whose `data_ptr` moved on sigma,
points directly at the root. Fix at that root (view producer, liveness
rule, or protected-set omission). Universal across native + triton.

**Piste 2 — reset arena between runs (rejected)**
Between each `run()`, set all non-weight, non-input arena slots to None,
forcing each run to fully repopulate its intermediates. Hides the symptom;
does not fix the root. Violates the "performance + correctness at the
source, no half solutions" rule in CLAUDE.md.

**Piste 3 — caching pool in `DeviceAllocator` (rejected for this bug)**
User-space pool for `cudaMalloc` / `cudaFree`. Would mask the bug the
same way PyTorch's caching allocator masks it in native. Does not fix
the root. Independent of the arena inter-run issue; remains an
interesting project for fragmentation on long-running zero3 / KV growth,
but not for this bug.

## Prerequisites already in place (this commit)

- Route A drain in `triton/sequence.py` (single-dev + multi-dev paths).
- `NBX_MALLOC_TRACE` allocation-site logger in `kernels/nbx_tensor.py`.
- Both env-gated, zero cost when off.
- Docstrings in both files reference this follow-up so the next session
  can navigate the stack without rediscovering the context.

## Next session scope

Implement `NBX_ARENA_DUMP=<path>` that writes one JSON file per boundary
(`bind_weights`, `bind_inputs[run_idx]`, `gather_outputs[run_idx]`).
Each file maps slot_idx → `{data_ptr, shape, strides, dtype, device_idx,
pinned, base_chain: [root_data_ptr, root_shape, ...]}`. Run PixArt-Alpha
with dumps on. Diff run 2's post-gather dump vs run 3's bind_inputs
dump. The slot that degrades is the root symptom. Trace back the code
that produced the degradation. Fix universally. Validate the 3 gates on
both PixArt models + regression harness.
