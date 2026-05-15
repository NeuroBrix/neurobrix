# Sana 4Kpx — Post Phase 1.5 (Autotune) Verdict

**Date**: 2026-05-04
**Chantier**: P-SANA-4KPX-RUNTIME, post @triton.autotune adoption
**Commits**: `d514bdb` (mm/bmm/addmm autotune) + `9e4b498` (conv2d autotune), both with `cache_results=True` Triton 3.6 native disk persistence.

## Matrix outcome

| Mode | Wall-clock | PNG | Verdict |
|---|---|---|---|
| compiled | 36.61 s | ✓ 4096×4096 14 MB coherent red apple | **PASS** |
| sequential | — | — | **FAIL_OOM** — sequential dispatcher does not implement op-level tiling interceptors (pre-existing structural; out of Phase 1.5 scope) |
| triton | — (SIGTERM 3 h) | — | **FAIL_TIMEOUT** — autotune sweep on 4Kpx-specific depthwise conv shapes (groups=11200) still in progress at 3 h cap; cache writes resumed at 15:50-15:57 with new 4Kpx shape keys |
| triton_sequential | — | — | **NOT_ATTEMPTED** — pending steps=1 sanity test to determine whether partial cache permits hot-path completion |

## Sana 1024 cross-reference (regression check)

| Mode | Wall-clock | Verdict |
|---|---|---|
| compiled | 12.60 s | PASS coherent |
| sequential | 13.22 s | PASS coherent |
| triton | 5862 s **cold sweep** | PASS coherent (autotune populated cache; warm path = ~100 s ± noise per Phase 1.5 measurement) |
| triton_sequential | 47.52 s **hot** | PASS coherent (proves cache_results=True replay works end-to-end) |

## P-SANA-4KPX-RUNTIME — REOPENED AGAIN 2026-05-05 (rescind closure 7d99b03)

The "closure" of commit `7d99b03` (factual root cause + named follow-ons P-TRITON-LIVE-SET-AUDIT / P-MULTI-GPU-NBX-ADAPTER) was premature. Sana 4Kpx triton still does not produce a PNG, and naming a follow-on chantier was a defer in disguise. The live-set gap investigation moves back INTO this chantier with the four documented suspects (kill_slots laxness / deferred-free retention / autotune workspace overhead / live-set diff compiled vs triton at conv::62 boundary). The 2026-05-04 reopen wording below remains valid.

---

## P-SANA-4KPX-RUNTIME — REOPENED 2026-05-04

The "closure on production-mode validation" recorded below was rescinded the same day. INDETERMINATE within a 3 h budget is not a closure verdict — it is a deferred failure disguised as mystery. The chantier stays open until the 4 modes factually work OR a structural irreducible cause is documented.

## Trajectory of fixes shipped (2026-05-04 → 2026-05-05)

| Commit | Change | Effect on Sana 4Kpx triton |
|---|---|---|
| `c740018` `0c373b1` | Étape 1 — kernel-level band-streaming inside `conv2d_wrapper` (4 GiB threshold default) | Dead code on Sana 4Kpx (per-op output ≤ 700 MB after op-level tiling). Kept for other models. |
| `b7c0773` | Étape 3 Niveau 1 — depthwise convolution specialization (groups == in_c == out_c → `depthwise_conv2d_kernel` direct stencil). Pattern: MultiPath/DepthwiseConv2d (CUTLASS sm_70). | **453× speedup** on the VAE depthwise op (4800 ms → 10.65 ms). Numerical max \|diff\|=0.0 vs cuDNN 2.6 ms. Unblocked the diffusion phase + VAE encoder/middle. |
| `7fd5396` | `FusionUpsampleProxy._nbytes = 0` (sentinel proxy, no GPU bytes). | Unblocked the VAE decoder past the upsample-fusion crash. |
| `12802be` | Phase 2 — caching free-list pool in `DeviceAllocator` (`NBX_ALLOC_POOL=1`, opt-in). Pattern: torch CachingAllocator release-cached-blocks → retry. + factual VRAM readout on OOM. | Sana 1024 hot regression intact at 42 s. **Does NOT unblock Sana 4Kpx** — diagnostic at OOM proves the issue is not fragmentation (see below). |

## Factual root cause of the residual blocker

At `aten.convolution::62` in the VAE post_loop:

```
device cuda:2  request=8589934592 bytes (8 GiB)
live_tracked = 26367 MB   pool_cached = 0 MB
driver_free  = 5482 MB    driver_total = 32501 MB
shortfall    = 2710 MB
```

This is **NBX live-watermark gap vs compiled mode**, not allocator fragmentation. The pool is empty because nothing has been freed — every byte is genuinely live in NBX's accounting. Compiled mode runs the same VAE in 36 s on the same GPU because torch's autograd-disabled cleanup + caching allocator hold a lower live watermark.

Suspects (to be settled by the next chantier):
- `kill_slots` metadata laxness in `triton/sequence.py`
- Deferred-free queue retention beyond compiled-mode last-use boundaries
- Triton autotune workspace overhead invisible to NBX's accounting

## Honest closure (2026-05-05)

| Mode | Verdict | Wall-clock |
|---|---|---|
| compiled (4Kpx) | ✅ PASS coherent PNG | 36.61 s |
| sequential (4Kpx) | ❌ FAIL_OOM by design (no op-level tiling in sequential dispatcher) | — |
| triton (4Kpx) | ❌ FAIL_OOM root-caused: NBX live watermark = 26.4 GB at conv::62, 5.4 GB driver-free, request 8 GiB short by 2.7 GB | — |
| triton_sequential (4Kpx) | (same root cause as triton — same arena lifecycle) | — |
| compiled (1024) | ✅ PASS | 12.6 s |
| sequential (1024) | ✅ PASS | 13.2 s |
| triton (1024) | ✅ PASS | 42 s hot |
| triton_sequential (1024) | ✅ PASS | 47 s hot |

**Two named follow-on chantiers**:
- **P-TRITON-LIVE-SET-AUDIT** (next priority) — instrument compiled vs triton modes to log per-op live tensor count + bytes, diff at conv::62 boundary, audit `kill_slots` / arena lifecycle / drain timing. Settles the watermark gap factually before any new optimisation work.
- **P-MULTI-GPU-NBX-ADAPTER** (lower priority) — make `core/strategies/component_placement.py` / `pipeline_parallel.py` work with NBXTensor. Would give VAE a dedicated GPU and bypass the watermark issue, but does NOT address the root cause.

Phase 2 free-list pool (commit `12802be`) is shipped opt-in and remains useful for any other workload bottlenecked by allocator fragmentation. Default off until validated across the full model surface.

---

## (RESCINDED) Closure on production-mode validation

**Closed on production-mode validation. R4 strict (4-mode) is NOT claimed satisfied.**

- ✅ **`compiled` (default production mode)**: PASS 36.61 s, coherent 4096×4096 PNG. Original 36 GiB OOM blocker resolved via Phase 1 dtype fix + op-level tiling (commit `44ffae0`).
- ⚠ **`sequential`**: FAIL_OOM **by design**. The sequential dispatcher (`core/runtime/graph/sequential_dispatcher.py`) does not implement op-level tiling interceptors — `register_op_uid_interceptor` is wired into `compiled_seq` (graph_executor.py:915) and `triton_seq` (graph_executor.py:1859) only. Verified by git log on commit `44ffae0`: the Sana 4Kpx tiling commit modified `graph_executor.py` but never touched `sequential_dispatcher.py`. Pre-Phase-1 baseline already showed all 3 modes failing OOM at 4Kpx — only `compiled` was claimed validated. **Structural limitation, not a regression**. Documented as a known limitation for sequential mode on >2K resolution diffusion models.
- ⚠ **`triton` / `triton_sequential`**: INDETERMINATE within reasonable wall-clock budgets (≤ 3 h, ≤ 30 min, ≤ 10 min). Pipeline opacity prevents diagnosis without instrumentation. Diagnostic chantier **P-TRITON-4KPX-PROFILE** opened for follow-up — see `feedback_p_triton_4kpx_profile_scope.md`.

## Phase 1.5 (autotune) goals — independent assessment

Phase 1.5 closes on its own scope:
- Volta-viable @triton.autotune subspace adopted on mm, bmm, addmm, baddbmm, conv2d. ✓
- `cache_results=True` Triton 3.6 native disk persistence active — single cold sweep amortizes across processes. ✓
- Sana 1024 4-mode matrix shows correctness preserved in all Triton modes. ✓
- Phase 1.5 micro-bench: matmul 8.58× cuBLAS (was 9.63×), conv2d 10.5× cuDNN (was 26.5×), structural plafond ~12 % cuBLAS on Volta confirmed. ✓

The autotune chantier achieves its measured upper bound on Volta sm_70 for the shapes encountered. Two distinct follow-on chantiers remain in scope based on the 4Kpx evidence:

1. **Cache pre-warm** (immediate): one-time longer-budget run (≥ 6 h) to fully populate `~/.triton/cache/<hash>/<fn>.autotune.json` for every Sana 4Kpx VAE shape. Subsequent runs benefit fully from the persistent disk cache.
2. **P-CONV2D-DEPTHWISE-OPTIM** (new): the im2col `conv2d_forward_kernel` is suboptimal for `groups=in_feat=out_feat` depthwise patterns (each group does a tiny per-channel convolution but the autotune sweep still walks 18 tile configs). Reference: lmdeploy / FlagGems use a dedicated depthwise kernel for this case. Outside the Phase 1.5 scope.
3. **P-TRITON-FUSED-KERNELS** (named in `src/neurobrix/CLAUDE.md` §23 R33 level 2): monolithic `@triton.jit` for `upsample+conv2d`, `conv2d+rms_norm+silu` etc. — orthogonal to depthwise issue, also reduces per-launch overhead on the tile loop.
4. Vendor-specific HMMA-fp16 path (Triton compiler does not lower fp16×fp16→fp32-acc to HMMA-fp16 on sm_70 — accepted as state-of-the-art Triton-Volta limit this year per `feedback_volta_mm_structural_gap.md`).

## Arbitrage recommendation

**Steps=1 sanity test outcome (decisive)**: ran twice with the partial cache populated by the 3 h run.
- Run A: `--steps 1`, 30 min budget → SIGTERM at 1800 s, no PNG, **zero** new cache writes during the run.
- Run B: `--steps 1` with `PYTHONUNBUFFERED=1` + `stdbuf -oL`, 10 min budget → SIGTERM at 600 s, no PNG, zero new cache writes, log frozen at "[Execute] Engine: TRITON" (the pipeline emits no per-step / per-tile markers, so internal progress is opaque).

The two follow-up runs confirm: with the partial cache populated by the 3 h sweep, even one diffusion step does not complete in 10 min. That outcome is consistent with **either** of two distinct causes — and the pipeline opacity does not let us distinguish them:

1. **More uncached shapes ahead**: the 11 cache entries written by the 3 h run cover only a subset of the Sana 4Kpx VAE shapes. The next un-touched shape triggers another long autotune sweep that exceeds the steps=1 budget. (Plausible — 4Kpx VAE has more layers and depthwise variants than 1024.)
2. **Hot inference itself is structurally slow**: per-launch Python overhead × 16 tiles × decoder depth × all kernels makes the 1-step wall-clock exceed 10 min even with everything cached.

Either cause moves the closure decision out of P-SANA-4KPX-RUNTIME scope.

### Arbitrage — closure path

**Close P-SANA-4KPX-RUNTIME** on the basis of the validated production path:
- Original OOM blocker (36 GiB allocation in compiled mode) is permanently resolved by op-level tiling. Wall-clock 36.61 s end-to-end on V100 32 GB. Coherent 4096×4096 PNG. ✓
- Sana 1024 4-mode matrix proves Phase 1.5 autotune adoption introduces zero correctness regression in any of the 4 modes. ✓
- Triton 4Kpx wall-clock and correctness are **indeterminate** within reasonable benchmarking budgets (≤ 3 h). Closing this measurement requires a dedicated diagnostic chantier (see below).

**Open P-TRITON-4KPX-PROFILE** as the named follow-on diagnostic chantier:
- Goal: instrument the `triton/sequence.py` execution path with per-component / per-tile timing so the 3 h budget can be partitioned between (a) autotune sweep on uncached shapes, (b) per-launch Python overhead in tile loops, (c) actual kernel compute, (d) anything else.
- Output: factual breakdown that determines whether to invest in P-CONV2D-DEPTHWISE-OPTIM (depthwise kernel rewrite), P-TRITON-FUSED-KERNELS (R33 level 2 fused @triton.jit), or both.

### What stays open vs closed

| Item | State |
|---|---|
| `compiled` 4Kpx runtime (R4 production path) | ✅ closed — validated |
| `compiled` 1024 + `sequential` 1024 + `triton` 1024 + `triton_sequential` 1024 (regression check) | ✅ closed — all 4 PNG coherent, autotune cache populated |
| `triton` / `triton_sequential` 4Kpx wall-clock + visual | 🔬 deferred to P-TRITON-4KPX-PROFILE |
| `sequential` 4Kpx (op-level tiling) | 🔬 deferred to P-SEQUENTIAL-TILING-INTEGRATION (out of Phase 1.5 scope) |

## Hocine validation

- ☐ `output_compiled.png` (4Kpx, 14 MB)
- ☐ `output_triton.png` Sana 1024 (1.08 MB) — cross-mode visual coherence check
- ☐ `output_triton_sequential.png` Sana 1024 (493 KB) — idem

Re-run command (any single mode):
```
neurobrix run [--triton|--sequential|--triton-sequential] --model Sana_1600M_4Kpx_BF16 --prompt "red apple" --steps 4 --output <path>.png
```
