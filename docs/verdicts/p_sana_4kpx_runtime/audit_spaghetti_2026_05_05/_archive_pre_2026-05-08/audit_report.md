# Spaghetti audit — triton subtree + kernels/wrappers + nbx_tensor

Date: 2026-05-05
Scope: P-SANA-4KPX-RUNTIME — pre-bisection static audit before launching
sequential / triton_sequential Sana 4Kpx with instrumentation.

Files audited:
  - src/neurobrix/triton/sequence.py            3141 lines
  - src/neurobrix/triton/sequential.py           204 lines
  - src/neurobrix/kernels/wrappers.py           5320 lines
  - src/neurobrix/kernels/nbx_tensor.py         2057 lines
  - src/neurobrix/triton/flow/iterative_process.py
  - src/neurobrix/core/runtime/graph_executor.py
  - src/neurobrix/core/runtime/graph/sequential_dispatcher.py
  - src/neurobrix/kernels/ops/fused_upsample_conv.py
  - src/neurobrix/kernels/ops/depthwise_conv2d.py
  - src/neurobrix/kernels/ops/conv_depthwise2d.py

═══════════════════════════════════════════════════════════════════════
(A) NBX_ env-var inventory — factual count: 40 distinct vars
═══════════════════════════════════════════════════════════════════════

Hocine's "80 NBX_*" was likely cumulative across both `os.environ` env
vars AND `_NBX_*` module-level globals (`_NBX_HW_PROFILE`,
`_NBX_HAS_NATIVE_BF16`, `_NBX_COMPUTE_DTYPE`, `_NBX_ACTIVATIONS_FP16_SAFE`,
`_NBX_CONV2D_BAND_BYTES`, `_NBX_CONV2D_TRACE`). The factual env-var
inventory (string-literal grep for `"NBX_*"`):

| # | var | files | classification | rationale |
|---|---|---|---|---|
| 1 | NBX_DEBUG | core/runtime/debug.py + 3 | **PROD** | runtime debug log toggle, documented in CLAUDE.md §8 |
| 2 | NBX_DEBUG_DECODE | core/flow/autoregressive.py | **DEBUG-OPT-IN** | LLM decode trace — keep, gated |
| 3 | NBX_FORCE_STRATEGY | core/prism/solver.py | **PROD** | manual Prism strategy override — legitimate ops escape hatch |
| 4 | NBX_DISABLE_MOE_FUSION | core/runtime/graph/moe_fusion.py | **DEBUG-OPT-IN** | regression toggle for MoE fusion — keep |
| 5 | NBX_MOE_FUSION_LOG | core/runtime/graph/moe_fusion.py | **DEBUG-OPT-IN** | gated log — keep |
| 6 | NBX_MOE_DIAG | triton/moe.py | **DEBUG-OPT-IN** | gated log — keep |
| 7 | NBX_Z3_TRITON_DIAG | triton/moe.py | **DEBUG-OPT-IN** | gated log — keep |
| 8 | NBX_ZERO3_VRAM_LOG | core/strategies/zero3.py | **DEBUG-OPT-IN** | gated log — keep |
| 9 | NBX_DUMP_LOGITS | flow/autoregressive.py × 2 | **DEBUG-OPT-IN** | gated log — keep |
| 10 | NBX_DUMP_TIDS | triton/sequence.py + compiled_sequence | **DEBUG-OPT-IN** | per-op tensor dump for triton↔native diff — keep, useful |
| 11 | NBX_DUMP_TIDS_FILTER | triton/sequence.py + compiled_sequence | **DEBUG-OPT-IN** | filter for above — keep |
| 12 | NBX_NAN_GUARD | compiled_sequence | **DEBUG-OPT-IN** | runtime NaN guard — keep |
| 13 | NBX_NAN_GUARD_VERBOSE | compiled_sequence | **DEBUG-OPT-IN** | verbose NaN guard — keep |
| 14 | NBX_TRITON_TRACE_NAN | triton/sequence.py | **DEBUG-OPT-IN** | first-Inf/NaN tracer — keep, gated |
| 15 | NBX_TRACE_NAN | compiled_sequence | **DEBUG-OPT-IN** | compiled-mode equivalent — keep |
| 16 | NBX_TRACE_RANGES | compiled_sequence + tools | **DEBUG-OPT-IN** | activation range probe — keep |
| 17 | NBX_RANGE_LOG | compiled_sequence + tools | **DEBUG-OPT-IN** | range log — keep |
| 18 | NBX_TRACE_ZEROS | compiled_sequence | **DEBUG-OPT-IN** | zero-output detector — keep |
| 19 | NBX_STRIDE_SKIP | core/runtime/graph_executor.py | **PROD** | stride safety toggle — keep |
| 20 | NBX_MM_SHAPES | kernels/wrappers.py | **DEBUG-OPT-IN** | matmul shape histogram — keep |
| 21 | NBX_TRITON_PROF | triton/sequence.py | **DEBUG-OPT-IN** | per-op profiling buckets — keep |
| 22 | NBX_IO_LOG | core/io/memory.py | **DEBUG-OPT-IN** | I/O timing log — keep |
| 23 | NBX_IO_WORKERS | core/io/memory.py + weight_loader.py | **PROD-YAML-CANDIDATE** | thread-pool size; SHOULD live in vendor YAML |
| 24 | NBX_PINNED_MEMORY | core/io/memory.py | **PROD-YAML-CANDIDATE** | pinned-mem toggle; YAML candidate |
| 25 | NBX_PREFETCH | core/io/memory.py | **PROD-YAML-CANDIDATE** | prefetch toggle; YAML candidate |
| 26 | NBX_PREFETCH_SIZE | core/io/memory.py | **PROD-YAML-CANDIDATE** | prefetch queue size; YAML candidate |
| 27 | NBX_MALLOC_TRACE | kernels/nbx_tensor.py | **DEBUG-OPT-IN** | tsv malloc trace — keep |
| 28 | NBX_UNLOAD_DIAG | core/runtime/graph_executor.py + triton/flow/iterative_process.py | **DEBUG-OPT-IN** | unload diagnostic — keep |
| 29 | NBX_CONV2D_TRACE | kernels/wrappers.py | **DEBUG-OPT-IN** | per-call conv2d shape log — keep, gated |
| 30 | NBX_CONV2D_BAND_BYTES | kernels/wrappers.py | **DEBUG-OPT-IN→YAML-CANDIDATE** | band-stream threshold (default 4 GiB); should be vendor YAML once Phase 1.5 closes |
| 31 | NBX_DEPTHWISE_DISABLE | kernels/wrappers.py | **TEMPORARY-DEAD** | escape hatch from depthwise specialization rollout; can DROP after Sana 4Kpx triton works (commit b7c0773 has been validated) |
| 32 | NBX_DEFERRED_DRAIN_BYTES | triton/sequence.py | **PROD-YAML-CANDIDATE** | Route A drain threshold (2 GB); YAML candidate per task #44 |
| 33 | NBX_DEFERRED_DRAIN_COUNT | triton/sequence.py | **PROD-YAML-CANDIDATE** | Route A count threshold; YAML candidate |
| 34 | NBX_DEFERRED_DRAIN_DIAG | triton/sequence.py | **DEBUG-OPT-IN** | gated drain log — keep |
| 35 | NBX_ALLOC_POOL | kernels/nbx_tensor.py | **DEBUG-OPT-IN** | Phase 2 caching pool toggle (validated empty pool at OOM = nothing to recover); KEEP gated, do not enable by default |
| 36 | NBX_GC_ON_OOM | kernels/nbx_tensor.py | **DEAD VESTIGE** | added 2026-05-05 commit b2ba978 to test ref-cycle hypothesis; proven NO help (Sana 4Kpx still OOMs); CAN DROP — but kept harmless under default-off |
| 37 | NBX_LIVE_DUMP_EVERY | triton/sequence.py | **DEBUG-OPT-IN** | periodic live-tracker — keep, very useful for bisection |
| 38 | NBX_LIVE_DUMP_ON_OOM | triton/sequence.py + iterative_process.py | **DEBUG-OPT-IN** | OOM forensics dump — keep, very useful |
| 39 | NBX_FORCE_GC | triton/sequence.py | **DEAD VESTIGE** | added to test cycle hypothesis; proven NO help; CAN DROP |
| 40 | NBX_ACTIVATIONS_FP16_SAFE | (string-literal in dtype.py docs only) | **DOC ONLY** | the actual mechanism is a per-component flag set via `set_activations_fp16_safe()`, NOT an env var; the string in docs is informational |

Summary:
  - Production-critical: 3 (NBX_DEBUG, NBX_FORCE_STRATEGY, NBX_STRIDE_SKIP)
  - Debug-opt-in (gated, default off, useful — KEEP): 23
  - YAML candidates (config-grade params currently env-vars): 6
    (NBX_IO_WORKERS, NBX_PINNED_MEMORY, NBX_PREFETCH, NBX_PREFETCH_SIZE,
     NBX_CONV2D_BAND_BYTES, NBX_DEFERRED_DRAIN_BYTES,
     NBX_DEFERRED_DRAIN_COUNT)
  - Dead vestiges (failed hypotheses, can drop): 2
    (NBX_GC_ON_OOM, NBX_FORCE_GC)
  - Temporary escape hatches (drop once feature stabilizes): 1
    (NBX_DEPTHWISE_DISABLE)
  - Doc-only string literal: 1 (NBX_ACTIVATIONS_FP16_SAFE)

**Verdict A**: 40 env vars is on the high side for a runtime, but the
breakdown is healthy: 23 are gated diagnostics that cost nothing at
default-off and provided real factual value in the chantier (live-block
walks, tid dumps, NaN tracers). Only **2 are dead vestiges** (failed
hypothesis tests). Cleanup recommendation: drop NBX_GC_ON_OOM and
NBX_FORCE_GC, plus revisit NBX_DEPTHWISE_DISABLE once b7c0773 is fully
validated. The 6 YAML-candidate vars deserve a separate config-grade
chantier (Phase 3+, not this session).

═══════════════════════════════════════════════════════════════════════
(B) Dead code + diagnostics in triton/sequence.py + kernels/wrappers.py
═══════════════════════════════════════════════════════════════════════

### B.1 — triton/sequence.py (3141 lines)

The file's structure is healthy: two-phase compile/run mirror of
`compiled_sequence.py`, op-elimination passes (detach, weight-transpose,
dead-causal-mask, swiglu-fuse, rope-fuse), liveness analysis, single-
device + multi-device hot loops, output gather. Nothing rolled-back is
left lying around — all committed changes either landed (Route A drain,
op_idx loop variable) or were reverted in the same commit chain.

Diagnostic blocks (all gated, default-off, can stay):
  - lines 2540-2544, 2870-2875: drain stats counters (gated NBX_DEFERRED_DRAIN_DIAG)
  - lines 2547-2570, 2818-2842: per-op profiling buckets (gated NBX_TRITON_PROF)
  - lines 2600-2611: periodic live-track log (gated NBX_LIVE_DUMP_EVERY)
  - lines 2615-2706: OOM-time arena dump + live-blocks walk + big-tensor
    referrer scan (gated NBX_LIVE_DUMP_ON_OOM) — VERY useful, keep
  - lines 2752-2756, 2950-2954: per-op NaN/Inf trace + tid dump (gated)
  - lines 2773-2783: per-op gc.collect (gated NBX_FORCE_GC) — DEAD,
    proven no help, can drop as part of vestige cleanup

Findings:
  - **NO dead-code paths** detected in the hot loop. Every conditional
    branch is exercised by at least one mode (NOP propagation, MoE
    fusion, multi-device transfer, deferred drain trigger).
  - **NO orphan `_LEGACY` markers** or commented-out blocks left over
    from band-loop closure refactor (commits 5a9f5d5 + 690bad9 reverted
    cleanly via b2ba978).
  - **NO duplicate dispatch logic**: `_run_single_device` and
    `_run_multi_device` share the same NOP/output/kill-slot/drain
    pattern with the only delta being device-switching + transfer slow
    path. Acceptable factual divergence.

### B.2 — kernels/wrappers.py (5320 lines)

Wrappers file is large (5320 lines) because it hosts every Triton op
wrapper signature. Each wrapper is small, self-contained, and reads
the same global pattern (`_NBX_COMPUTE_DTYPE`, `_NBX_HAS_NATIVE_BF16`).

Findings:
  - **DUPLICATE depthwise wrappers**: TWO depthwise kernels coexist:
    - `kernels/ops/conv_depthwise2d.py` (154 lines, FlagGems vendoring)
      — registered via `dispatch.py:496-498` for op names
      `depthwise_conv2d`, `conv_depthwise2d`, `conv_transpose2d` →
      `conv_depthwise2d_wrapper` (wrappers.py:3903)
    - `kernels/ops/depthwise_conv2d.py` (107 lines, NEW commit b7c0773)
      — used by `_depthwise_conv2d_dispatch` (wrappers.py:2418) called
      from `conv2d_wrapper` (wrappers.py:2369) when groups==in_c==out_c
    The NEW one is the 453× speedup on Sana 4Kpx VAE. The OLD one is
    only reachable when the DAG names the op directly as
    `aten::depthwise_conv2d` (rare; modern PyTorch traces emit
    `aten::convolution` then routes via `conv2d_wrapper` which
    catches the depthwise pattern and routes to the NEW kernel).
    **DECISION needed**: keep both (back-compat for legacy traces) OR
    deprecate the old one (force everything through `conv2d_wrapper`
    detection). Recommendation: **keep both for now** — the OLD one
    is in dispatch.py for legacy DAGs and costs nothing if not called.
    Mark with comment that `conv2d_wrapper:2369` is the preferred path.
  - **Étape 1 band-streaming wrapper** (`_conv2d_band_streamed`,
    wrappers.py:2456): factual status = dead code on Sana 4Kpx because
    op-level tiling already keeps per-op output ≤ 700 MB before this
    threshold (4 GiB) trips. Kept for OTHER models with tall conv
    outputs that bypass op-level tiling. **NOT dead architecturally**,
    just dormant on Sana 4Kpx — keep.
  - **NO NBX↔torch round-trip** detected in wrappers.py for Triton
    kernel wrappers (R33 audit clean).

### B.3 — kernels/ops/fused_upsample_conv.py (648 lines)

This file contains the dual-backend dispatcher pattern documented in
CLAUDE.md R33: `fused_upsample_conv2d` (line 110) detects backend by
arg type, routes to `_fused_upsample_conv2d_torch` or
`_fused_upsample_conv2d_nbx`. Same pattern for `tiled_conv2d_spatial`
(line 306) → `_tiled_conv2d_spatial_torch` / `_tiled_conv2d_spatial_nbx`.

Findings:
  - **R33 compliance**: dispatcher routes by `isinstance(ref, NBXTensor)`.
    The two backends are SEALED — no `_torch` call from the `_nbx`
    branch and vice versa. Clean.
  - The dispatcher's own `import torch` (line 132) is the documented
    R33 exception (router file, not a compute kernel).
  - **NOT spaghetti**, but worth noting: this is the ONLY file in
    `kernels/ops/` that contains torch compute logic (because the
    compiled-mode VAE upsample-fusion path lives here). Future
    refactor opportunity = move `_xxx_torch` variants to `core/`
    once the patterns crystallise — but not now.

### B.4 — kernels/nbx_tensor.py (2057 lines)

DeviceAllocator with Phase 2 caching pool added 2026-05-05 (12802be).
Pool is opt-in (NBX_ALLOC_POOL=1). All new code is properly gated and
self-documents the rationale (free-list pool, OOM-flush-and-retry,
cycle-collect retry under NBX_GC_ON_OOM).

Findings:
  - **Pool implementation is clean** — 200 lines of well-documented
    code (lines 334-580). No dead branches.
  - **NBX_GC_ON_OOM retry block** (lines 501-505): proven not useful
    on Sana 4Kpx (factual: OOM still raises with same shortfall).
    Can drop, but harmless if kept opt-in.
  - **OOM diagnostic block** (lines 507-533): added in commit b2ba978,
    very useful — gives factual breakdown
    (live_tracked / pool_cached / driver_free / driver_total) inside
    the RuntimeError message. KEEP.

═══════════════════════════════════════════════════════════════════════
(C) R30 verification — triton ↔ sequential parity audit
═══════════════════════════════════════════════════════════════════════

**Hocine's question**: are triton_sequential and sequential running the
SAME flow_handler with the SAME DAG, only swapping the backend
(NBXTensor vs torch.Tensor)? Or are there divergences of LOGIC?

### C.1 — Flow-handler dispatch

`core/runtime/executor.py:444-625` shows the factual dispatch:

```
if ctx.mode in ("triton", "triton_sequential"):
    return TritonXxxHandler(...)         # triton subtree
return XxxHandler(...)                   # core subtree
```

This means **`triton` AND `triton_sequential` share the SAME flow
handler** (the Triton-side mirror, e.g. `TritonIterativeProcessHandler`).
Similarly **`compiled` AND `sequential` share the SAME flow handler**
(the core-side, e.g. `IterativeProcessHandler`).

→ The split is **flow-handler-level by mode-pair**, not per-mode. So
the diff `sequential vs triton_sequential` ALSO diffs the flow handler
(`core/flow/iterative_process.py` vs `triton/flow/iterative_process.py`),
NOT just the backend kernels. This is a NON-TRIVIAL divergence
documented per the CLAUDE.md two-modes doctrine — the two subtrees are
DELIBERATELY duplicated to keep them sealed.

### C.2 — Op dispatch within each mode

Inside `graph_executor.py:1395-1500`:
  - `mode in ("triton","triton_sequential")` → `_run_triton(...)` →
    branches on `mode == "triton_sequential"`:
      - YES: `_run_triton_sequential` → `TritonSequentialDispatcher`
        (triton/sequential.py, 204 lines) — op-by-op
      - NO:  `_run_triton_compiled` → `TritonSequence` (triton/sequence.py)
        — pre-compiled hot loop
  - `mode == "compiled"` → `_execute_compiled_graph()` →
    `CompiledSequence` (core/runtime/graph/compiled_sequence.py) —
    pre-compiled hot loop
  - `mode == "sequential"` → `_execute_all_ops()` → per-op
    `_dispatch_op` → `NativeATenDispatcher`
    (core/runtime/graph/sequential_dispatcher.py)

So the **4-mode matrix** is:

| mode | flow handler | op dispatcher | tensor type | kernel backend |
|---|---|---|---|---|
| sequential | core/flow/* | NativeATenDispatcher | torch.Tensor | torch + cuDNN/cuBLAS |
| compiled | core/flow/* | CompiledSequence | torch.Tensor | torch + cuDNN/cuBLAS |
| triton_sequential | triton/flow/* | TritonSequentialDispatcher | NBXTensor | Triton kernels |
| triton | triton/flow/* | TritonSequence | NBXTensor | Triton kernels |

**R30 reading**: the four modes share the SAME DAG (graph.json) and
the SAME execution_order. They diverge on:
  1. Flow-handler subtree (core/ vs triton/) — by-design, sealed
  2. Op dispatch substrate (per-op torch vs compiled-closure torch vs
     per-op NBX vs compiled-closure NBX) — by-design
  3. Backend kernels (cuDNN/cuBLAS/F.* vs Triton @triton.jit) —
     by-design

**No accidental divergence** detected. The diff between
`sequential` and `triton_sequential` is THREE legitimate axes
simultaneously, not one. Hocine's bisection plan is therefore exactly
right: comparing them isolates the AGGREGATE divergence between the
two subtrees, not a single variable.

### C.3 — Liveness analysis parity

Both `_run_triton_sequential` (graph_executor.py:1598-1652) and
`TritonSequence._compute_liveness` (sequence.py) compute `dead_at_op`
the same way (last_use map + protected set including
inputs/weights/outputs). So tensor eviction policy is FACTUALLY
identical between triton_sequential and triton.

`NativeATenDispatcher` doesn't compute liveness — instead
`graph_executor._cleanup_finished_tensors` (line 2140) evicts based on
a `last_use_tid` map computed by `_compute_last_use`. Same logic, same
result, different code path. So sequential AND triton_sequential have
**equivalent eviction semantics** — confirming this isn't a
liveness-divergence issue at the diff layer.

### C.4 — Output finalization parity

Both paths end with `_gather_outputs` / output_tids → store mapping.
The triton path writes through to `variable_resolver.resolved` via
`OutputExtractor.store_component_outputs`. Native does the same via
`_gather_outputs` then the executor's loop. No divergence.

### C.5 — Verdict on R30

The four modes are **structurally aligned at the contract surface**
(DAG, topology, variable_resolver, flow_handler interface) but
**deliberately duplicated at the compute substrate** (core/ vs triton/
subtrees). Hocine's "Neurobrix ne fait pas la même chose pour les deux
modes" is FACTUALLY TRUE in the sense that mode 1 (compiled/sequential)
and mode 2 (triton/triton_sequential) traverse separate code subtrees
end-to-end — but this is the EXPLICIT architectural design, not a bug.
The sealing is what allows mode 1 to be debranched without touching
mode 2 (per CLAUDE.md two-modes doctrine).

→ Bisection plan is sound. Comparing `sequential` vs `triton_sequential`
will diff three axes simultaneously (flow handler subtree + op
dispatcher + kernel backend), but liveness and output semantics are
factually equivalent so any live-set divergence at boundary VAE
op 600-650 is meaningful diagnostic data.

═══════════════════════════════════════════════════════════════════════
RECOMMENDATIONS BEFORE BISECTION RUN
═══════════════════════════════════════════════════════════════════════

**Option A — Drop dead vestiges first (10 min cleanup)**:
  - Delete NBX_GC_ON_OOM block in nbx_tensor.py:501-505
  - Delete NBX_FORCE_GC block in triton/sequence.py:2773-2783
  - Optionally delete NBX_DEPTHWISE_DISABLE escape hatch since b7c0773
    is validated; or leave as guard for future regression detection
  - Single dedicated commit "chore: drop failed-hypothesis env vars"
  - Then proceed with Étape 1 sequential bisection

**Option B — Skip cleanup, proceed with bisection now**:
  - The dead vestiges cost nothing at default-off and don't affect the
    bisection diff
  - Cleanup can land as a follow-on commit after Sana 4Kpx triton works
  - Faster to factual answer

**Recommendation**: **Option B**. The audit found the codebase is
healthier than the line counts suggest — no spaghetti, no rolled-back
debris, R33 compliance clean, R30 architectural sealing as designed.
The 2 dead env vars and 1 escape hatch are noise, not blockers.
Bisection should proceed.

═══════════════════════════════════════════════════════════════════════
ETA FOR BISECTION STEPS (post-audit)
═══════════════════════════════════════════════════════════════════════

| Step | Mode | Sana 4Kpx wall-clock estimate | Rationale |
|---|---|---|---|
| 1 | sequential | 10-15 min cold; 8-10 min hot | torch eager + cuDNN; reference oracle |
| 2 | triton_sequential | 20-30 min cold (autotune compile); 12-18 min hot | Triton + NBXTensor op-by-op + autotune cache miss first run |
| 3 | diff analysis | static, ~5 min | grep / diff per-op live tracker |
| 4 | (conditional) compiled vs triton | 6-8 min × 2 | only if 1+2 PASS but 4+ FAIL |
| 5 | R29 4-PNG generation | 30-45 min | one PNG per mode if all pass |

Total budget if Steps 1+2 PASS: ~1h-1h15. If they FAIL with same
symptom, root cause is in the sub-graph or backend, not in compile-mode
fusion — and Step 3 diff identifies which tensors triton_sequential
holds that sequential frees. If they DIVERGE (one passes, one fails),
the diff at boundary identifies the cause directly.

EOF
