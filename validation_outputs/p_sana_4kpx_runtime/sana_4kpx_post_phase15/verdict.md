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

## P-SANA-4KPX-RUNTIME closure (honest wording)

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
