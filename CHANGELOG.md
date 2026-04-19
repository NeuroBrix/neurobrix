# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`NBX_FORCE_STRATEGY` env var for deterministic Prism strategy selection** (`core/prism/solver.py`): setting `NBX_FORCE_STRATEGY=<strategy>` short-circuits the score cascade to the named strategy. If the strategy is unknown → `RuntimeError` listing the 9 valid values (`single_gpu`, `single_gpu_lifecycle`, `component_placement`, `pipeline_parallel`, `block_scatter`, `weight_sharding`, `component_placement_lazy`, `lazy_sequential`, `zero3`). If the strategy is valid but unavailable for the given device count (multi-GPU strategy on a single-GPU profile) → `RuntimeError` distinguishing the two cases. If the strategy is valid and in-set but its `_try_*` method cannot fit the model → `RuntimeError: ZERO FALLBACK: NBX_FORCE_STRATEGY={strategy} cannot fit ...` — no silent fallback to an alternative, because the operator asked for X specifically and picking Y without notice would hide bugs. Without the env var, behaviour is unchanged. Unit tests (`tests/scratch/prism_matrix_validation/test_force_strategy.py`) exercise all 5 code paths against cached TinyLlama-1.1B + Qwen3-30B fixtures. Unblocks the Phase 2 strategy matrix (`tests/scratch/prism_matrix_validation/matrix_driver.py` + `MATRIX_REPORT.md`) and any future per-strategy regression harness.
- **`NBXTensor.is_expanded()`** (`kernels/nbx_tensor.py`): pure metadata predicate — `any(stride == 0 and shape > 1 ...)`. One canonical helper per R2; used by `to_cuda` / `to_cuda_async` and by `TritonSequence._transfer_tensor` to materialise expand views before H2D.
- **`NBXTensor._contiguous_cpu()`** (`kernels/nbx_tensor.py`): CPU-side strided-copy helper. Wraps the view's byte window via `np.ctypeslib.as_array` + `np.lib.stride_tricks.as_strided` with esz-byte granularity, runs `np.ascontiguousarray` in C, and `ctypes.memmove`s into a fresh `NBXTensor.empty_cpu` preserving pinned/unpinned backing. Handles expand views automatically (numpy replays along stride-0 axes). Zero triton, zero torch.
- **Heterogeneous 2 × V100 16 GB NVLink hardware profile** (`config/hardware/v100-16g-x2-01.yml`): describes cuda:0 + cuda:1 on the Dell C4140 host, leaving cuda:2 / cuda:3 free for parallel workloads (regression harness). Used by the Phase 2 multi-GPU strategy matrix.
- **Phase 2 strategy matrix validation artefacts** (`tests/scratch/prism_matrix_validation/`): `matrix_driver.py` (spawns `neurobrix run` per strategy × mode, classifies outcomes, emits JSON + Markdown), `test_force_strategy.py` (5 unit tests over the env var), `MATRIX_REPORT.md` (8 / 16 🟢; the 4 reds are documented as expected "strategy-not-applicable" behaviour for TinyLlama — each red is sourced to the `_try_*` method + line that correctly returned None).
- **Item 1 + Item 2 scratch artefacts** (`tests/scratch/dette_technique_session/`): `test_item1_contiguous_cpu.py` (5 / 5 byte-correct vs torch), `test_item2_expand_h2d.py` (5 / 5 byte-identical to torch reference), `SESSION_REPORT.md` (prior-session context for the continuation).
- **Zero3 block-wise ratchet pipelining (native + triton)** — zero3 is no longer a per-op CPU→GPU transfer "slow path". `Zero3Strategy` now drives a sliding 2-block window (current + prefetched) on the compiled op sequence. At each op-to-block transition it (a) ensures block N+1's weights are resident via an async H2D on a dedicated transfer stream, (b) binds the arena to block N's GPU copy, (c) evicts block N-1 after materializing any dependent intermediates, returning VRAM to the driver immediately. Multi-pass autoregressive decode resets the ratchet via `_post_run_hook` so VRAM stays flat across tokens. Polymorphic on tensor flavor — same strategy code handles both `torch.Tensor` + `CompiledSequence` and `NBXTensor` + `TritonSequence` via the parity APIs. Closes Qwen3-30B-A3B-Thinking-2507 Test A on 16 GB V100: peak VRAM bounded to two blocks' weights + KV cache + activations (~6.7 GB target), prefill dramatically faster than the per-op slow path. The legacy per-op slow path survives only for non-pipelined weights (embeddings, final norm, lm_head — sub-percent of total weight mass).
- **MoE fusion — Pass 2 output-side dead-op sweep** (`core/runtime/graph/moe_fusion.py`): the existing post-fusion dead-op elim only removed ops whose INPUTS were in `removed_producers`. It missed ops whose only consumer (typically `aten::mm` for expert projections) was collapsed into `custom::moe_fused` — notably the `aten::t(expert_weight)` that precedes each expert matmul. Left in the compiled sequence these ran at execution, stored a `.t()` view in the arena, and the view's `_base` pinned the CPU-offloaded expert weight to GPU memory under zero3 (~1.15 GB leaked per block on Qwen3-30B). New Pass 2 iterates to fixed point removing any op in the surviving execution order whose outputs have no active consumer and are not DAG-level outputs. Shared-expert paths and the fused op itself are protected. Empirical: Qwen3-30B-A3B sees 25,009 orphaned ops removed (18,432 are `aten::t` on expert weights — exact 128 × 3 × 48 match; remaining 6,577 are cascaded routing orphans: `aten::unsqueeze` on topk outputs, `aten::zeros` allocator buffers, one `aten::arange`). TinyLlama and other dense LLMs: 0 (no MoE fusion runs). Log controlled by `NBX_MOE_FUSION_LOG=1`.
- **Async stream + event primitives in `DeviceAllocator`** (`kernels/nbx_tensor.py`): `create_stream` / `destroy_stream` / `stream_synchronize`, `create_event` / `destroy_event`, `record_event(event, stream)`, `stream_wait_event(stream, event)`, `memcpy_async(dst, src, nbytes, kind, stream)` — thin ctypes wrappers over `cudaStreamCreate` / `cudaEventCreate` / `cudaMemcpyAsync` (HIP equivalents on ROCm). Backend mapping extended in `_GPU_BACKENDS`. Zero torch; zero allocation overhead for stream handles (opaque ints). Enables the zero3 ratchet to overlap H2D(N+1) with compute(N) on the default stream.
- **`NBXTensor.to_cuda_async(device_idx, stream)`** (`kernels/nbx_tensor.py`): non-blocking variant of `to_cuda` that enqueues the H2D on a user-supplied stream. Caller is responsible for sequencing via `stream_wait_event` or `stream_synchronize`. Documented that the source tensor MUST be pinned for the driver to actually overlap on a non-default stream.
- **`materialize_slots_depending_on(weight_slot_ids)`** — parity API on both `CompiledSequence` (`core/runtime/graph/compiled_sequence.py`) and `TritonSequence` (`triton/sequence.py`). Copies every arena slot whose tensor aliases a weight slot (via PyTorch storage identity / NBXTensor `_base`) into fresh storage via `.contiguous()` so the weight can be evicted without leaving dangling pointers. Under the Pass-2-fixed MoE graph this typically returns 0 — dead views never land in the arena to begin with — but remains as the correct defensive primitive for legitimate future cases (non-MoE patterns, computed-view aliases).
- **MoE `_ptr_cache` fingerprint + LRU** (`triton/moe.py`): the pointer-table cache key was previously `f"moe_{gate_weights[0].data_ptr()}_{num_experts}"`. Under zero3 pipelining weights are freed and reallocated between blocks, frequently at identical virtual addresses for same-sized buffers, so this static key would resolve to a stale `PtrTables` and cause silent garbage or illegal memory accesses. New `_ptr_cache_fingerprint(gate, up, down, num_experts)` hashes ALL 3 × `num_experts` expert `data_ptr()`s so any swap invalidates. `_ptr_cache_get` / `_ptr_cache_put` implement an LRU bounded to `_PTR_CACHE_MAXSIZE = 256` entries (OrderedDict with `move_to_end` on hit, `popitem(last=False)` on overflow) — keeps cache memory flat across hundreds of blocks × devices during long pipelined runs.
- **`TensorArena.__len__` / `Arena.__len__`**: both the native (`core/runtime/graph/compiled_sequence.py`) and triton (`triton/arena.py`) arenas now expose `__len__`, needed by `materialize_slots_depending_on` for slot iteration and by diagnostic scripts that enumerate the arena.
- **Zero3 universality in triton mode** — Qwen3-30B-A3B-Thinking-2507 now runs end-to-end with `--triton --hardware v100-16g` (zero3 cascade, 30 B weights on 16 GB GPU). Previously the feature was non-functional in triton: weight load failed with a 60 GB `cudaMalloc` attempt because the `shard_map[*]="cpu"` contract was silently ignored, and the TritonSequence hot loop had no parity with the correctness APIs added for native zero3. This release lands the full infrastructure + the specific memory-footprint fix that closes Test A.
- **`NBXTensor` CPU form** (`kernels/nbx_tensor.py`): `NBXTensor.empty_cpu(shape, dtype, pinned=False)` / `to_cuda(device_idx)` / `to_cpu(pinned=False)` / `pin_host()` — numpy-backed by default, `cudaMallocHost`-backed when `pinned=True` for fast non-blocking H2D DMA. `__del__` dispatches to `free_host_pinned` or lets numpy GC handle unpinned backing. New accessors: `is_cpu`, `is_pinned`. Zero torch dependency.
- **`DeviceAllocator` accounting + pinned host** (`kernels/nbx_tensor.py`): `malloc_host_pinned` / `free_host_pinned` wrap `cudaMallocHost` / `cudaFreeHost`. Running byte counters: `memory_allocated(device_idx)`, `peak_memory_allocated`, `reset_peak_memory`, `host_pinned_allocated`, `host_pinned_peak`. Enables per-NeuroBrix memory diagnostics without round-tripping through `cudaMemGetInfo` (which reports device-wide usage including other processes).
- **`DeviceAllocator._cuda_ptr_device` per-allocation device tracking** (`kernels/nbx_tensor.py`): every `malloc_cuda` records the allocating device so that `free_cuda`'s `_cuda_live_bytes` decrement always targets the owning device, regardless of which device is current at the GC site. Closes a reporting-only accounting drift that surfaced while instrumenting the zero3 leak investigation. (A stream-ordered `cudaMallocAsync`/`cudaFreeAsync` path was prototyped in the same session and reverted — the cross-device pool semantics broke `pipeline_parallel` multi-GPU on Qwen3-30B; the `_find_cuda_arg` fix below was what actually closed the arena retention leak, not the allocator swap.)
- **Triton weight_loader CPU partition** (`triton/weight_loader.py`): detects Prism's `shard_map = {shard_path: "cpu"}` convention used by zero3 and partitions weights via the `_BLOCK_RE` regex. Block weights (`block.N.*` / `blocks.N.*` / `layers.N.*`) land in pinned host memory as CPU-backed `NBXTensor` via `_load_to_pinned_cpu`; non-block weights (embeddings, final norm, lm_head) stay on the GPU arena because they are accessed directly by flow handlers (notably `GraphLMSession.prefill` → `w.embedding`) that bypass the compiled sequence and expect GPU pointers.
- **`TritonSequence` zero3 parity API** (`triton/sequence.py`) — mirrors commit ea90d66's CompiledSequence additions in triton, same signatures and semantics:
  - `rebind_partial(partial_map) → List[int]` — swap a subset of weight slots, honours `_pretranspose_weights`.
  - `recompute_op_devices_for_slots(modified_slots)` — patch `op.device_idx` / `op.needs_transfer` for only the ops whose weight inputs intersect the modified set. Treats CPU-backed tensors correctly (no conflation with cuda:0 via shared `_device_idx=0`).
  - `get_op_blocks()` — group ops by transformer block index using the shared `_BLOCK_RE`. Cached post-compile.
  - `override_weightless_op_devices(device_idx)` — force tensor-creation ops (arange, scalar_tensor, full) to allocate on the execution device instead of inheriting CPU from the activation-device chain.
  - `mark_cpu_weighted_ops_for_transfer(exec_device_idx)` — flag every CPU-weighted op for the slow path.
  - `run(pre_op_callback=…)` / `_run_multi_device(pre_op_callback=…)` — optional hook fires BEFORE each op's args are resolved, used by zero3 priming and reserved for future pipelining.
- **`GraphExecutor` triton callback plumbing** (`core/runtime/graph_executor.py`): `_run_triton_compiled` now threads `self._pre_op_callback or self._persistent_pre_op_callback` into `TritonSequence.run(...)`. `_ensure_weights_loaded` installs zero3 hooks on both native (`_compiled_seq`) and triton (`_triton_seq`) executors, so flow handlers that bypass `strategy.execute_component` (autoregressive LLM prefill) get the fix transparently in either mode.
- **`Zero3Strategy` universal pin + install** (`core/strategies/zero3.py`): `_pin_cpu_weights` branches polymorphically on `torch.Tensor` vs `NBXTensor` — native uses `.contiguous().pin_memory()`, triton uses `NBXTensor.pin_host()` (no torch leak). `_install` callback resolves either `_compiled_seq` or `_triton_seq` on the executor and calls the matching parity API.
- **`ExecutionStrategy.transfer_tensor` polymorphism** (`core/strategies/base.py`): duck-types on `hasattr(tensor, 'to_cuda') and hasattr(tensor, '_device')` to route NBXTensor through the zero-torch path (`to_cuda(dev_idx)` / `to_cpu()`), and torch.Tensor through `.to(device)`. Single public API, no mode-specific method fork. Zero torch in the NBXTensor branch.
- **MoE fused CPU promotion** (`triton/moe.py`): `execute_moe_fused` detects CPU-backed expert weights and promotes them to the activation device via `NBXTensor.to_cuda(act_dev)` BEFORE `_build_ptr_tables` bakes raw pointers into the GPU int64 table. Under zero3, `_ptr_cache` is bypassed per-call because the promoted tensor addresses are fresh; under normal multi-GPU paths the cache is unchanged. Explicit `del` + `gc.collect` of the promoted lists + `tables` at function exit ensures the 768 MB of expert weights released before the next MoE call.
- **`ComponentArena.__del__` safety** (`triton/memory_pool.py`): `_base_ptr` is initialised to `0` before the `malloc_cuda` call so `__del__` is a safe no-op when `__init__` raises on a failed allocation (secondary issue surfaced during the zero3 leak investigation — previously a failed 60 GB cudaMalloc in `ComponentArena(...)` triggered a second exception from `__del__`).
- **Zero3 correctness path + block-pipelining groundwork**: `Zero3Strategy` now installs a per-executor priming hook that fires on the first op of the first run and flips `op.device` / `op.needs_transfer` on every CPU-weighted and weightless op so the multi-device slow path transfers weights on-the-fly (working set = one op). Previous implementation had four bugs (wrong prefetch target when non-block weights existed, missing wait on the async second prefetch, no per-block loop driving the ratchet, arena never rebound so the prefetched GPU tensors were never consumed) that left Qwen3-30B in zero3 crashing at `aten.mm::0` with `mat2 on cpu`. Hooks install at weight-load time via `RuntimeExecutor._ensure_weights_loaded`, so flow handlers that bypass `strategy.execute_component` (notably `GraphLMSession.prefill` for autoregressive LLMs, which calls `executor.run` directly) get the fix transparently. Verified end-to-end: Qwen3-30B-A3B-Thinking-2507 on a single 16 GB V100 (forced via `--hardware v100-16g`) generates coherent tokens where it previously crashed on the first matmul.
- **`CompiledSequence.rebind_partial(partial_map) → List[int]`**: replace a subset of weight slots on the arena without touching the rest. Honours the same `_pretranspose_weights` contract as `bind_weights`. Returns the list of modified slot indices.
- **`CompiledSequence.recompute_op_devices_for_slots(modified_slots)`**: patch per-op `op.device` + `op.needs_transfer` for exactly the ops that read the modified slots. Complements `compute_op_devices()` for post-bind changes without a full rescan.
- **`CompiledSequence.get_op_blocks() → Dict[int, Dict]`**: introspect the compiled op list and group ops by transformer block index (`block.N.` / `blocks.N.` / `layers.N.` / `model.layers.N.` / `encoder.layers.N.` / `decoder.layers.N.`). Non-block weights → `-1`; weightless ops inherit predecessor's block. Result cached on the sequence (immutable post-compile).
- **`CompiledSequence.override_weightless_op_devices(device)`**: zero3 helper that forces tensor-creation ops (arange, scalar_tensor, full, attn-mask casts) to allocate on the execution GPU instead of inheriting device from the (CPU-correct-for-FGP-but-wrong-for-zero3) activation-device chain built by `compute_op_devices()`.
- **`CompiledSequence.mark_cpu_weighted_ops_for_transfer(exec_device)`**: flip `needs_transfer=True` for every weighted op whose weight is currently on CPU, so the multi-device slow path handles the per-op transfer. Returns the count of flips for diagnostic.
- **Optional `pre_op_callback` on `CompiledSequence.run` / `_run_inner` / `_run_inner_multi_device`**, plumbed through `GraphExecutor.run` via new `_persistent_pre_op_callback` and `_post_run_hook` attributes on GraphExecutor. Explicit per-call callback wins over the persistent one. The multi-device hot loop invokes the callback with `(op_idx, op)` before arg resolution; fast-path single-device ignores it to preserve zero overhead when unused.
- **Hardware-gated fp16 overflow protection (WIP, architectural surface only)**: `PrismProfile.has_native_bf16` property data-driven from `devices_support_dtype("bfloat16")` (covers all vendors). `kernels/wrappers.set_hardware_profile()` propagates the flag into a module-level `_NBX_HAS_NATIVE_BF16` gate; on pre-Ampere hardware (no native bf16), `mm`/`bmm`/`addmm` upcast fp16 inputs and land output in fp32. Triton `matmul_kernel`/`addmm_kernel` gain `IEEE_PRECISION` constexpr to force `tl.dot(input_precision="ieee")` when inputs were promoted to fp32. **Known incomplete**: openaudio DualAR still crashes (upstream `_to_copy(fp32→fp16)` clamps to Inf before mm); perf of per-call weight upcast not yet measured; Ampere+ no-op path not yet mock-verified.
- Flow-aware CLI dispatch in regression harness: STT models now auto-dispatch `--audio`, TTS-with-reference models auto-dispatch reference audio. Unblocks whisper, parakeet, canary-qwen, Voxtral, granite-speech, Kokoro native in automated testing.
- New kernel wrappers in Triton dispatch: `linear`, `isin`, `is_nonzero`, `layer_norm` alias. Enables chatterbox Triton LM stage and openaudio DualAR entry.
- NBXTensor→numpy D2H helper (`_to_numpy`) for flow handlers that need host-side arrays without going through torch.

### Fixed
- **NBXTensor.contiguous() on CPU tensors silently promoted to GPU** (Item 1, `kernels/nbx_tensor.py:1247`): the method hard-coded `NBXTensor.empty(self._shape, self._dtype, f"cuda:{self._device_idx}")` for the destination, then dispatched `_strided_copy` — a Triton kernel — against the source's pointer. On a CPU-backed `NBXTensor` (e.g. zero3-offloaded block weight, or any host-allocated buffer via `empty_cpu`) the destination was silently allocated on GPU and the Triton kernel read host addresses as device pointers, producing undefined behaviour. Fix branches on `self._device`: CPU tensors now go through a new `_contiguous_cpu()` helper that materialises the view in numpy (`as_strided` over the byte window, `ascontiguousarray`, `ctypes.memmove` into a fresh `empty_cpu(..., pinned=self._pinned)`) — CPU stays on CPU, pinnability preserved. GPU path unchanged. Validated in `tests/scratch/dette_technique_session/test_item1_contiguous_cpu.py` (5 / 5 byte-correct vs `torch.Tensor.contiguous().numpy()` across transpose, expand, narrow + permute, pinned-preservation, and the already-contiguous short-circuit).
- **NBXTensor expand views (`stride == 0` axes) over-read backing storage during H2D / D2D** (Item 2, `kernels/nbx_tensor.py` + `triton/sequence.py`): `to_cuda`, `to_cuda_async`, and `TritonSequence._transfer_tensor` all memcpy'd `tensor._nbytes = numel × element_size` bytes from `data_ptr()` into a same-sized GPU allocation. For an expand view — e.g. `(1, 768).expand(2, 512, 768)`, stride `(0, 0, 1)` — numel counts the expanded shape while the source allocation holds only the unbroadcast elements; the memcpy ran past the real buffer and stamped garbage into the GPU tensor. Fix: new `NBXTensor.is_expanded()` predicate (`any(st == 0 and sh > 1 ...)`) gates a `contiguous()` materialisation before the memcpy in all three call sites. Pure transposes (non-zero strides, `nbytes == backing_bytes`) are untouched so the zero3 `.t()` pre-transpose contract survives. Validated in `tests/scratch/dette_technique_session/test_item2_expand_h2d.py` (5 / 5 byte-identical to torch reference across bias broadcast, scalar broadcast, multi-dim expand, async variant, and a non-expand transpose negative control). Supersedes the "Deferred — NBXTensor expand views on CPU" note that was in this file in the earlier revision.
- **`--triton-sequential` did not thread the zero3 pre-op callback** (Item 4, `core/runtime/graph_executor.py:~1481`): `_run_triton_sequential` dispatched ops directly against the store with no callback hook, while `_run_triton_compiled` passes `pre_op_callback=cb` into `TritonSequence.run`. Any strategy that installed a persistent hook — notably zero3 — therefore never fired on the sequential path and zero3-weighted ops ran against CPU pointers. Fix resolves `self._pre_op_callback or self._persistent_pre_op_callback` once outside the op loop and invokes it with `(op_idx, op_data)` before every op, wrapped in a try / except to match the `TritonSequence.run` contract where callback failures are isolated from the compute loop. Smoke-verified with TinyLlama-1.1B-Chat-v1.0 `--triton-sequential --hardware v100-16g-x2-01` — coherent tokens ("Certainly! Here"). Full zero3 end-to-end under `--triton-sequential` on Qwen3-30B is deferred to the Phase 2 matrix extension (the installed branch exists and is exercised on a fitting model; a zero3-forcing spec belongs in that matrix when `NBX_FORCE_STRATEGY=zero3 --triton-sequential` is added there).
- **`_to_copy(fp32 → fp16)` saturated to ±Inf for values outside fp16 range** (Item 5, `core/dtype/engine.py` in both branches of `_make_to_copy`): when the graph emits an explicit fp16-targeted `aten::_to_copy` and the input contains fp32 / fp64 / bf16 values beyond ±65504 (observed upstream of the OpenAudio DualAR pre-projection, flagged in the v0.1.5 known-incomplete list), `inp.to(torch.float16)` produced ±Inf and the next `mm` propagated NaN through the rest of the graph. Fix clamps to `[-65504, 65504]` before the narrowing cast, matching the pattern already established in `core/dtype/converter.py::safe_dtype_convert` line 45. Clamp is identity for in-range values (no numerical change for models that don't overflow); only the saturating path is modified, and it now produces ±65504 instead of ±Inf, which the next `mm` handles correctly. TinyLlama native decode unchanged (3.59 s, "Certainly! Here").
- **Triton zero3 numerical divergence — every pretransposed CPU weight silently untransposed**: `TritonSequence._transfer_tensor` built the destination GPU NBXTensor via `NBXTensor.empty(tensor._shape, tensor._dtype, f"cuda:{target_dev}")`, which hard-sets `_contiguous_strides(shape)` on the destination regardless of the source's actual strides. The memcpy then copied the source's raw backing bytes (which describe the *original*, pre-transpose row-major layout) to the destination, and the destination's contiguous strides re-indexed those bytes as if they were the transposed layout. Effect: every linear weight that was pre-transposed at bind time by `_eliminate_weight_transpose_ops` (i.e. all of them, across all 48 blocks and 128 MoE experts for Qwen3-30B) arrived on the GPU with dims 0 and 1 swapped back to original; every `mm` then computed `act @ W` instead of `act @ W.t()`. Test A (`Qwen3-30B-A3B-Thinking-2507 --triton --hardware v100-16g --prompt "2+2="`) produced `"OTTéraquate RED"` with a flat, ~half-magnitude logit distribution (top1 +17.42 vs native +37.31, disjoint top-10) — the model was running forward with structurally wrong weights. Fix: build the destination directly via `NBXTensor(dst_ptr, tensor._shape, tensor._strides, tensor._dtype, 'cuda', owns_data=True, device_idx=target_dev, offset=0)` so the stride semantics of the view survive the transfer; downstream `.contiguous()` inside `wrappers.mm` then correctly materialises via `_strided_copy`, matching how `torch.Tensor.to(device)` preserves strides on the native side. Validation: Test A triton now produces `"Okay, the user"` (identical to native), cosine similarity of step-0 logits = 0.999999, argmax match, top-10 same order with max per-token delta of 0.13 (fp16 rounding). Bonus: deepseek-moe-16b-chat `--triton` previously returned gibberish (`"eses самоу思acular"`) — root cause was the same bug via MoE expert-weight promotion through `_transfer_tensor`; it now matches native (`" 4"`). Investigation log in `tests/scratch/divergence_inv/DIVERGENCE_REPORT.md`; regression coverage in `tests/scratch/zero3_triton_impl/test_triton_seq_parity.py::test_transfer_tensor_preserves_transposed_view` (+ contiguous-source sanity + GPU-to-GPU variants). Pipeline_parallel triton (all-GPU weights, historical caller of `_transfer_tensor` on contig activations only) unchanged and confirmed non-regressed.
- **Triton zero3 arena retention leak (+1.28 GB per transformer block → OOM at block 7/9 on V100 16 GB)**: the slow path in `TritonSequence._run_multi_device` was unconditionally promoting CPU-backed weight inputs to the execution GPU, even for metadata ops (`aten::t`, `aten::view`, `aten::reshape`, `aten::permute`, `aten::unsqueeze`) that only manipulate strides and produce VIEWS. Each metadata op on a CPU-resident zero3 weight therefore allocated a fresh 3 MB GPU temp, then returned a view whose `NBXTensor._base` held that temp alive for the lifetime of the arena slot — ~389 unfreed allocations accumulated per transformer block, exactly the block's MoE expert weight size. Native CompiledSequence's slow path scans args for a CUDA tensor first and only promotes when compute on GPU is actually required; ported to triton via `_transfer_args` + a new `_find_cuda_arg(args)` helper. Investigation methodology in `tests/scratch/zero3_triton_impl/LEAK_PINPOINT_REPORT.md`; arena slot histogram confirming the fix in `analyze_histograms_v2.py` (triton `aten::t` total: 2415 MB of unique storages before fix, 0 MB after — matching native). Per-block growth post-fix: ~67 MB (identical to native, matches KV-cache accumulation rate).
- `MemoryManager.unload_weights` silent use-after-free: `device_sync()` was called AFTER `weights_dict.clear()` (too late — `clear()` already triggered `ComponentArena`/`NBXTensor` finalizers that call `cudaFree`) and with no device argument (no-op on multi-GPU — `device_utils.py:27` returns early on `None`). If a kernel was still in-flight on the stream when its buffer was freed, the CUDA context was silently corrupted and the next `cudaMalloc` failed with `cudaErrorIllegalAddress` (err 700), which the allocator wrapper misreported as "GPU malloc failed". The cache flush at the end of the same function had the exact same bug (`device_empty_cache()` with no argument also returns early — `device_utils.py:43`). Fix enumerates every device in the dict (`_arenas`, `NBXTensor`, `torch.Tensor`) once, syncs each BEFORE clearing refs, then flushes each device's cache AFTER gc. Exposed by lifecycle / lazy strategies that actually unload between phases.
- Pre-Ampere LLM decode regression introduced by the wip fp16 overflow protection: weights now upcast to fp32 once at bind time (when VRAM permits) instead of on every matmul call. TinyLlama 1.1B `--triton` decode on V100 returns to the v0.1.5 baseline (matmul ~28 ms/step vs ~285 ms/step in the regressed wip). Models too large for fp32 weights (e.g. Qwen3-30B) fall back silently to per-call upcast.
- Janus-Pro-7B Triton: autoregressive flow now family-aware, no longer tries to apply `chat_template` on image-generation models.
- Zero-torch contract in `triton/flow/audio.py`: `_get_compute_dtype` now returns a string; torch conversion pushed to stage handlers (`core/flow/stages/`) where torch is accepted as boundary.

### Changed
- **NBXTensor metadata ops propagate `pinned=self._pinned`** (`kernels/nbx_tensor.py`, 10 sites — `view`, `unsqueeze`, `squeeze`, `permute`, `transpose`, `expand`, `narrow`, `select`, `unfold`, `as_strided`): view constructors previously dropped the pinned flag, so a transposed / reshaped view of host memory allocated via `cudaMallocHost` lost its `_pinned` marker. `_contiguous_cpu()` (Item 1) reads this flag to decide whether the materialised contiguous copy should also be pinned, so correct propagation is a prerequisite for pinned-roundtrip byte-correctness. All 10 sites changed by a single `replace_all` — pattern was identical. Validated by `test_item1_contiguous_cpu.py::test_pinned_preservation`.
- **Zero3 execution model — from per-op slow path to block-wise pipelining**: prefill and decode under the zero3 cascade are substantially faster and use bounded VRAM (two blocks' weights + KV cache + activations) regardless of model size. The per-op transfer path is retained only for the small non-block weights (embeddings, final norm, lm_head) which stay on CPU through the whole run.
- Stage handlers (`core/flow/stages/kokoro.py`, `vibevoice.py`): added `_coerce_torch_dtype` helper to accept both string (from Triton engine) and `torch.dtype` (from native engine).

### Removed
- **Unused locals in `core/prism/solver.py::solve`**: `total_mem` (line 473), `total_vram` (line 480), and `scale_factor` (line 923 inside `_apply_attention_correction`) — computed but never referenced in the method bodies. Kept the computation pipeline around them (`sum(component_memory.items())`, device preparation) because those remain load-bearing. ~3 lines freed.
- **Duplicate local `import` sites consolidated to top-level** in `core/prism/solver.py`: `import logging` (was re-imported inside the serve-mode fallback branch), and two redeclarations of `from neurobrix.core.prism.structure import AllocationStrategy` inside `_score_strategy` and `_build_plan`. The top-of-file imports (`import logging`, `from neurobrix.core.prism.structure import AllocationStrategy, ...`) now cover these use sites once. ~4 lines freed.
- **Unused `except RuntimeError as e:`** in `core/prism/solver.py::solve` KV-cache fit-check (the exception object was never referenced in the except body). Replaced by bare `except RuntimeError:` — same behaviour, 1 character saved and Pyright no longer flags the unused binding.
- **Dead zero3 pipelining scaffolding** that never worked: `_prefetch_block`, `_wait_prefetch`, `_evict_block_from_gpu`, `_gpu_weight_cache`, `_block_groups`, `_group_weights_by_block`, and the module-level `_BLOCK_RE` regex (regex moved to `compiled_sequence.py` where it serves the general-purpose `get_op_blocks` API). These methods were never reached correctly at runtime — the prefetched GPU tensors ended up in a dict that the compiled sequence never consulted, so the existing CPU→GPU slow path was carrying all the work anyway.

### Deferred
- **Item 3 — fp16 bind-time upcast safety gate** (session SESSION_REPORT.md): the naive implementation (gate the bind-time upcast on `_scan_bf16_fp16_safety`) was prototyped, validated byte-correct on TinyLlama but regressed per-token decode ~85 % (0.246 ms/token → 0.452 ms/token). Root cause: `wrappers.mm` at `_NBX_HAS_NATIVE_BF16 == False` upcasts activations to fp32 per call, and the "dtype alignment" block (`wrappers.py` lines 1053–1059) then widens the fp16 weight to fp32 per call too when bind-time upcast hasn't happened. Skipping bind-time upcast shifts the fp32 weight copy cost from load-once to per-call-many. Fix requires kernel-level work (teach `matmul_kernel` to accept mixed fp32_act × fp16_weight natively OR switch to op-level `AMP_FP32_OPS` classification on the triton side instead of the current blanket pre-Ampere gate). Tracked as a standalone kernel session; reverted cleanly in this commit — working tree restored to prior upcast behaviour on `weight_loader`, no half-merged residue.
- **Phase 2 red verdicts for multi-GPU strategies** (`tests/scratch/prism_matrix_validation/MATRIX_REPORT.md`, 4 / 8 strategies red in both modes): `component_placement`, `pipeline_parallel`, `block_scatter`, `weight_sharding` all return None from their `_try_*` methods when TinyLlama-1.1B is forced onto `v100-16g-x2-01` because the model is small enough to fit a single GPU, which the strategy-specific fit heuristics correctly reject. Sourced per-strategy in the report. These are NOT Prism regressions — reds are produced by the new `ZERO FALLBACK: NBX_FORCE_STRATEGY=X cannot fit` path (added in this session). Demonstrating these strategies' green behaviour requires a larger (model, hardware) pairing (Qwen3-30B on 4 × 16 GB or DeepSeek-MoE on 2 × 32 GB). Matrix extension belongs in a dedicated session so it can run Qwen3-30B × 10 multi-GPU runs without harness-cuda:2 contention.

## [0.1.5] - 2026-04-15

### Added
- Regression harness (`tests/regression/`) — automated model×mode matrix, golden output comparison, pytest-based with `--runslow` flag for heavy models.
- Three graph-level fusion passes for Triton decode optimization:
  - Dead causal mask elimination: removes ~132 ops/step (ones→tril→logical_not→where chain feeding SDPA attn_mask, replaced by kernel-native IS_CAUSAL).
  - SwiGLU fusion: collapses silu+mul into single `custom::swiglu_fused` kernel (~22 ops/step).
  - RoPE fusion: replaces 18-op rotate_half chain per layer (slice×4, neg×2, cat×2, mul×4, add×2) with single `custom::rope_fused` kernel backed by Liger-Kernel's `rope_forward_kernel` (~396 ops dropped for 22-layer models).
- Cumulative Triton decode performance (TinyLlama V100 fp16): step time 460 ms → 94 ms (4.9× faster), element-wise ops 684 → 376 (−45%).

### Fixed
- Sana diffusion transformer NaN in Triton mode: `bmm` attention scores overflowed fp16 on V100. `bmm` now always outputs fp32 for half-precision inputs (attention intermediates are temporary, no OOM impact). SDPA wrapper aligns Q/K/V dtypes before kernel launch.
- Native CFG engine crash on diffusion models (Sana, PixArt-Sigma): string dtype from Prism's allocation was passed to `torch.Tensor.to()` which interpreted it as device name. Added `_resolve_torch_dtype` helper.
- Kokoro Triton startup crash on 1-D constant tensors in models without a `seq_len` symbol.

### Changed
- Weight transpose elimination (`_eliminate_weight_transpose_ops`) ported from native to Triton — 154 fewer ops/step, structural parity with native CompiledSequence.
- Orphan `rope_wrapper` removed (incompatible with kernel, zero call sites in any model graph). Replaced by `rope_fused_wrapper` with correct Liger kernel signature.

### Documented
- WARNING blocks added to `stages/kokoro.py` and `stages/vibevoice.py` flagging runtime dependency violations (phonemizer/espeak-ng imports, PyTorch native bypass of TensorDAG).
- `KNOWN_FAILURES` in regression harness `conftest.py` with exact reasons for each xfail.

## [0.1.4] - 2026-04-14

### Changed
- DeepSeek benchmark script now requires `HF_TOKEN` to be provided via the shell environment or a gitignored `.env` file; no token is ever hardcoded. This replaces the previous version of the same file, which shipped with a hardcoded token — users who pulled `0.1.3` from the sdist should upgrade.
- GitLab CI `publish-pypi` stage switched to `when: manual`. New version tags no longer trigger an automatic PyPI upload; an operator now reviews the build artefacts on GitLab and clicks the job explicitly.

### Security
- Rotate the credential that was shipped in the `neurobrix-0.1.3.tar.gz` sdist on PyPI (hardcoded HuggingFace access token in `benchmarks/profile_hf_deepseek.py`). The sdist has been yanked; `pip install neurobrix` now resolves to `0.1.4` by default. Users who installed `0.1.3` from the sdist (not the wheel — the wheel does not include benchmarks) should upgrade to `0.1.4`.

## [0.1.3] - 2026-04-14

### Added
- `--triton` mode: DeepSeek-MoE-16B now supported end-to-end (greedy output `" Hello! How can I help you today?"` on `"Hello"`, matching the native path semantically). Joins TinyLlama-1.1B and Qwen3-30B-A3B as fully working LLMs in the Triton runtime.
- `--triton` decode speedups for LLMs across the board (V100 numbers, fp16):
  - TinyLlama-1.1B: full decode step 443 → 160 ms (2.8× faster).
  - Per-matmul on the decode hot path: 2.02 → 0.20 ms (10–18× depending on shape).
  - Per-SDPA on decode with GQA: 1.58 → 0.94 ms (1.7×).
- Decode-aware output precision for matrix multiplication: when running one token at a time, accumulation now lands in fp32, preventing silent overflow on very deep MoE stacks (Qwen3-30B observed crash → now stable). Prefill and image/video spatial matmuls keep their fp16 output — no memory regression on diffusion.
- Decode-aware attention: short-query attention now uses a compact block size (no more 99% wasted compute when generating one token at a time). GQA models compute in place — K/V are no longer expanded to the Q head count in front of every attention call.
- Triton profiling harness (opt-in, off by default):
  - `NBX_TRITON_PROF=1` — per-category ms/op breakdown (matmul / sdpa / elem / meta / embed / other) for every run.
  - `NBX_DUMP_TIDS=<path>` + `NBX_DUMP_TIDS_FILTER=<substrings>` — dump any op output as JSON for side-by-side native vs Triton numerical diff.
  - `NBX_MOE_DIAG=1` — dump MoE routing intermediates on the first forward pass.
- New benchmark: `benchmarks/profile_hf_deepseek.py` — reference timings against the HuggingFace + Accelerate device_map=auto baseline on the same hardware.

### Changed
- Triton dtype policy simplified: only `div` still forces an fp32 input upcast. Matmul ops (`mm`, `bmm`, `addmm`) now cooperate with the new decode-aware output precision instead of forcing every input to fp32 per call — removes a ~3.5 GB per-decode-step weight-copy cost that was silently capping throughput.
- Triton graph-load pipeline is more permissive about trace-shaped vs declared-shaped buffers: models that ship position-indexed lookup tables sized from the trace sample (DeepSeek's per-block rotary cache is the reference case) now load instead of crashing on a shape mismatch.
- Embedding wrapper accepts any scalar index dtype and casts internally. Fixes diffusion timestep → embedding paths (PixArt-Sigma and similar DiTs) that used to crash with a pointer/float type error.

### Fixed
- DeepSeek-MoE-16B `--triton`: previously produced gibberish at decode (`"роко"` / `"!!!!!"`). The three root causes were all addressed in this release:
  1. The model's MoE routing normalisation flag was silently ignored in Triton mode (defaulted to the Qwen3 convention), collapsing routed-expert magnitudes ~20×.
  2. Top-k selection over a softmax with non-power-of-two k returned a corrupted tail — the fix skips a redundant sort stage when the input fits in a single chunk. Side effect: one fewer kernel launch per MoE layer for every model whose expert count fits in that chunk.
  3. RoPE position indexing collapsed after the first decode step when cos/sin were recomputed per forward (DeepSeek's pattern). The runtime now pins the RoPE chain at its traced size so subsequent decode positions stay in-bounds.
- Qwen3-30B-A3B `--triton` now runs noticeably faster at decode from the new attention block-size heuristic and the GQA-in-place kernel path.
- TinyLlama-1.1B `--triton` decode is faster end-to-end (2.8× step time) and keeps the same output as before on greedy runs.
- PixArt-Sigma `--triton` no longer crashes on the embedding kernel (timestep dtype) or on the first `aten::add` after the timestep path (computable buffers now enter the Triton runtime in the expected tensor type). Further progress is blocked by an SDPA VRAM allocation failure partway through the transformer on 16 GB V100s — tracked as a separate issue; the native path has an unrelated config bug on the same setup.

### Removed
- Per-attention-call GQA materialization (`unsqueeze → expand → reshape → contiguous` of K and V). Replaced by kernel-native stride indirection, active only when the model has GQA; non-GQA models are bit-identical to before.

## [0.1.2] - 2026-04-03

### Added
- `--triton` mode — compiled Triton kernel inference (136 kernel files, 128 dispatch entries)
- `--triton-sequential` mode — sequential Triton execution for kernel debugging
- `TritonSequentialDispatcher` — extends NativeATenDispatcher, routes compute ops to Triton kernels
- 136 pure `@triton.jit` kernel files extracted from FlagGems, attorch, Liger-Kernel, Flash-Attention (Dao-AILab)
- NBXTensor — lightweight tensor descriptor for zero-PyTorch metadata ops
- Universal launch layer: `_prepare_binary`, `_prepare_comparison` — broadcasting, scalar handling, device context for all ops
- `_cuda_guard` in dispatch — handles multi-GPU + Zero3 CPU offloading transparently
- Metal GPU detection: `--triton` on Apple Silicon shows "not compatible" message
- CPU Triton backend: auto-enables `TRITON_CPU_BACKEND=1` on CPU-only machines
- Symbolic shape patching for sequential mode: `_patch_seq_len_in_ops` resolves trace-time seq_len in creation ops
- Pure Triton inference mode for LLM autoregressive generation (`--triton` flag)
- Zero-torch flow handler: autoregressive.py, samplers.py, generator.py, session.py
- Triton sequential debug mode (`--triton-sequential` flag)
- KV cache with GQA support for Triton decode (O(1) per token)
- Strided scatter kernel for non-contiguous KV cache writes
- NBXTensor boundary functions: nbx_to_torch(), nbx_dtype_to_torch()

### Changed
- DtypeEngine: merged `FP16_PRECISION_OPS` into `_FP16_NEED_FP32` (subset of `AMP_FP16_OPS`), eliminated duplicate sets
- DtypeEngine: `amp_cast_inputs()` now handles `_FP16_NEED_FP32` (was only in `compile_op`)
- Conv2d kernel: replaced FlagGems with attorch (V100 `num_stages` compatibility)
- Conv1d: routes through conv2d via unsqueeze (V100 safe)
- `pow` kernel: uses `libdevice.pow` for negative base handling (was `exp(e*log(x))` → NaN)
- `compiled_ops.py` enforces: missing Triton kernel for compute op = crash with descriptive error
- Remove @triton.autotune from 100+ element-wise kernels (fixed BLOCK_SIZE=1024)
- Remove @triton.autotune from 36 compute-bound kernels (fixed conservative configs)
- Cold start reduced from 8+ minutes to ~5 seconds

### Fixed
- NBXTensor.cat() called is_contiguous as attribute instead of method — corrupted RoPE
- Symbolic promotion skipped when multiple symbols share trace_value (s1/s3 ambiguity)
- SDPA double-masking when graph passes explicit causal mask with is_causal=False
- NBXTensor.__setitem__ used flat copy_kernel on non-contiguous narrow view — corrupted KV cache
- NBXTensor.contiguous() used memcpy instead of strided copy for non-contiguous views

### Removed
- `kernels/adapter.py` (1181 lines) — replaced by `dispatch.py` + `wrappers.py`
- `kernels/mapping.py` (155 lines), `kernels/resolver.py` (316 lines), `kernels/registry.py` (68 lines), `kernels/exceptions.py` (15 lines)
- `kernels/ops_legacy/` directory, `kernels/arch/` directory, `kernels/spec.py`
- `_execute_triton_op`, `_precompile_dispatch_table`, `_exec_type_map` from graph_executor.py
- Apple Silicon (MPS) support — M1 through M5 Ultra, unified memory, auto-detection
- `DeviceBrand.APPLE` with `"mps"` device prefix in Prism hardware abstraction
- Apple Silicon chip database (20 variants: M1-M5 base/Pro/Max/Ultra with GPU cores, bandwidth, memory)
- `device_utils.py` — unified device abstraction (`device_sync`, `device_empty_cache`, `device_seed`, `device_memory_stats`, `device_multinomial`)
- No-silent-fallback guardrail hook (blocks `PYTORCH_ENABLE_MPS_FALLBACK` and try/except device swallowing)
- Single-GPU strategy shortcut in Prism solver (skips multi-GPU cascade for 1-device hardware)
- `neurobrix doctor` command with OS-specific PATH fix instructions
- GitLab CI/CD pipeline for PyPI publishing (OIDC trusted publisher + API token fallback)

### Changed
- All `torch.cuda.empty_cache()` calls replaced with device-agnostic `device_empty_cache()` (26 call sites across flow handlers, strategies, graph executor, serving engine)
- All `torch.cuda.synchronize()` for timing replaced with `device_sync()` (serving engine, strategy base)
- All `torch.cuda.manual_seed_all()` replaced with `device_seed()` (serving engine)
- VRAM reporting in serving engine uses `device_memory_stats()` (supports CUDA + MPS)
- `torch.multinomial` replaced with `device_multinomial()` — CPU round-trip on MPS (9 call sites)
- Removed hardcoded `"cuda:0"` defaults from loaders and strategies — crash explicitly if Prism provides no device
- All repository URLs migrated from GitHub to GitLab (`gitlab.com/neurobrix/Neurobrix`)
- Dependencies updated: added `pydantic`, `packaging`, `torchaudio`, `snac`, `phonemizer`, `imageio-ffmpeg`, `transformers`, `mistral-common`, `tiktoken` — all families work out of the box
- bf16 dtype support gated by Apple chip generation (M2+ with macOS 14+)

### Removed
- `licenses.py` — hardcoded license classifications deleted. Hub is the single source of truth.

### Fixed
- License gating desync between CLI and hub — CLI now reads `gated`/`licenseName`/`licenseUrl` from hub API
- Serving engine crash on `ExecutionPlan.allocations` — use `primary_device` property
- Prism profile loader mapped unknown brands to NVIDIA silently — Apple got `cuda:0` instead of `mps:0`. Now crashes on unknown brand.
- Weight loader only transferred weights to CUDA GPUs — MPS weights stayed on CPU, triggering multi-device path. Now transfers to any GPU device.
- macOS daemon used `os.fork()` + `os.setsid()` which breaks Metal GPU access (MTLCompilerService is per-session). Now uses `subprocess.Popen` like Windows.
- False `avx2` ISA warning on Apple Silicon — ARM chips use NEON, not x86 ISA. Skip check for arm64.
- Apple M2+ now prefers bf16 (not fp16) — bf16 has fp32 exponent range, prevents overflow in matmul/conv accumulation that caused blurry image output
- MPS dtype flow: AMP stays ON (same rules as CUDA). fp32 precision chain flows through single-input ops (pow, mean, rsqrt) safely. Multi-input ops (mm, addmm) cast inputs to compute_dtype via AMP FP16 wrappers. No mixed dtype at multi-input op boundaries.
- SNAC audio decoder had silent `except ImportError` fallback returning zeros — now crashes explicitly
- `python -m neurobrix` shows PATH hint when CLI not on PATH

## [0.1.0] - 2026-03-26

First stable release of NeuroBrix — universal deep learning inference engine.

### Added
- NBX container format with TensorDAG, topology, manifest
- Prism hardware solver with multi-GPU allocation (11 strategies: single_gpu through zero3)
- CompiledSequence zero-overhead execution engine (eliminates all Python dict lookups)
- DtypeEngine with automatic mixed precision (standard PyTorch AMP rules)
- 4 model families: image (diffusion + VQ), LLM, audio, video
- CLI commands: `run`, `serve`, `chat`, `stop`, `hub`, `import`, `list`, `remove`, `clean`, `inspect`, `validate`, `info`, `doctor`
- MoE (Mixture of Experts) fused dispatch with NOP propagation
- KV cache with data-driven sizing and on-demand growth
- Triton GPU kernel framework
- NeuroBrix model registry at neurobrix.es
- Support for 34 models across 4 families (LLM, image, audio, video)
- Audio family: all 11 models working — Whisper, Whisper V3 Turbo, Parakeet, Orpheus, Canary-Qwen, Kokoro-82M, VibeVoice-1.5B, Voxtral, OpenAudio-S1, Granite Speech, Chatterbox
- Audio flow handlers: encoder_decoder, audio_llm, dual_ar, rnnt, tts_llm
- Universal AudioEngine with data-driven flow routing
- SANA-Video 720p support (video generation)
- Persistent model serving: `neurobrix serve`, `neurobrix chat`, `neurobrix stop`
- Multi-turn conversation with context management and automatic summarization
- Universal hardware auto-detection — `--hardware` flag is optional
- Cross-platform support: Windows, macOS, and Linux
- Platform-adaptive IPC: AF_UNIX on Unix/macOS, TCP localhost on Windows
- Universal TilingEngine — data-driven per-component tiling with accumulate-and-divide blending
- Symbolic spatial dims in compiled graphs — view/reshape ops use expression trees for multi-resolution
- ExprArg in CompiledSequence — runtime resolves symbolic expressions
- 12+ GPU hardware profiles (RTX 20/30/40 series, A10, A100, H100, L40S, T4, V100)
- Pipeline parallel, block scatter, weight sharding allocation strategies
- Prism hot/cold budget split for serve vs run mode
- Zero3 layer-wise pipelining with dual CUDA streams
- License system for model distribution with acceptance flow in `neurobrix import`
- Enterprise-grade documentation system (MkDocs Material)
- `neurobrix doctor` command for diagnosing PATH and installation issues

### Security
- Enforced `weights_only=True` for torch.load (prevents pickle RCE)
- Zip-slip path traversal validation in registry import
- Safe arithmetic parser replacing `eval()` in shape resolver

[Unreleased]: https://gitlab.com/neurobrix/Neurobrix/-/compare/v0.1.0...main
[0.1.0]: https://gitlab.com/neurobrix/Neurobrix/-/releases/v0.1.0
