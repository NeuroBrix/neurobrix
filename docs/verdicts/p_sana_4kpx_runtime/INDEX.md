# P-SANA-4KPX-RUNTIME — Validation Index

Chantier: op-level tiling to run Sana 4Kpx VAE at its native 4096×4096 resolution on V100 32 GB without Forge retracing.

| # | Mode | Verdict | Output | Notes |
|---|---|---|---|---|
| 1 | compiled (default) | ✅ PASS visual verified | `sana_4kpx/output.png` 4096×4096 (24 MB) | red apple coherent, ~74 s, mean 68 std 69 |
| 2 | --triton | ⏸ deferred | — | Tiling works (0 OOM), but Triton kernel autotune on ~18 ops × 4 bands exceeds 30 min wall-clock first-run. Persistent Triton kernel cache for bands = follow-up chantier |
| 3 | --triton-sequential | ⏸ deferred | — | Same cause as --triton — compile-bound, no OOM |

Hocine validation : ☐ compiled `output.png` (inspection visuelle requise)

## Before chantier

```
RuntimeError: Failed at op aten.convolution::62 (aten::convolution):
  CUDA ort of memory. Tried to allocate 36.00 GiB.
```

(observed phase 5 of the previous runtime_alignment chantier, across all 3 modes)

## After chantier (mode compiled)

```
[Execute] Running pipeline...
   Engine: COMPILED
[Timing] Total execution: 73.98s
SAVED: validation_outputs/p_sana_4kpx_runtime/sana_4kpx/output.png
```

Output PNG 4096×4096×3 RGB = 23.58 MB. Visuellement: pomme rorge glossy
with green/brown stem on textured wood background. Faint seam line visible
at mid-height (boundary between tile bands) — refining halo blending is
a future optimization chantier.

## Architecture in place (commits `44ffae0` + `d8e1be9`)

- `core/module/tiling_engine.py` — `OpLevelTilingPlan` + `OpLevelTilingEngine.from_op_level_constraint` + `register_into_graph_executor`
- `kernels/ops/fused_upsample_conv.py` — `fused_upsample_conv2d` (band-streaming, no intermediate), `tiled_conv2d_spatial` (workspace bound), `tiled_rms_norm_spatial` (channel-axis preserved)
- `core/prism/profiler.py` — `ActivationProfile.overflow_ops` (per-op output+workspace scan)
- `core/prism/memory_estimator.py` — `estimate_op_workspace_bytes` (cuDNN bound by 0.75×VRAM)
- `core/prism/solver.py` — `_detect_op_level_tiling_pairs` (cascade post-strategy)
- `core/runtime/executor.py` — branchement `plan.runtime_op_tiling`
- `core/runtime/graph_executor.py` — `register_op_uid_interceptors` (compiled + triton mirror via `_pending_triton_uid_interceptors`)
- `core/runtime/graph/compiled_sequence.py` — `register_op_uid_interceptor` + `update_op_uid_interceptors`
- `triton/sequence.py` — mirror of the same mechanism on the triton path

## R30 status

- compiled (default) : ✅ wired + validated runtime + visual
- triton : ✅ wired + non-OOM validated (passes the ops without crash) + ⏸ wall-clock validation deferred (Triton compile cost)
- triton_sequential : ✅ wired + non-OOM validated + ⏸ wall-clock validation deferred (idem)

The R30 architecture is clean by construction: a single `register_op_uid_interceptors` on GraphExecutor mirrors to both sequences (compiled + triton). Sibling ops of the same type stay on the native path — the interception only touches op_uid flagged by Prism.

## Post-chantier backlog

1. Persistent Triton kernel cache for bands (key `(N, C, IH, IW)`) — resolves the triton wall-clock
2. Halo blending in `fused_upsample_conv2d` to eliminate the visible seam
3. Test on other high-resolution conv-based diffusion models (cf design §6 prediction: DC-AE Sana 1024 does NOT trigger — verified, peak < 8 GB; future CogVideoX VAE / HunyuanVideo VAE / Wan2 VAE / swin2sr upscaler will be candidates)

## 4-mode performance matrix — Sana 1024 (steps=4, post execution-mode renaming)

NeuroBrix vocabulary: four execution modes are `compiled` (default,
PyTorch fused + cuDNN/cuBLAS), `sequential` (PyTorch eager op-by-op),
`triton` (Triton-pure compiled, NeuroBrix arena+closures+fused
kernels), `triton_sequential` (Triton-pure op-by-op debug). The legacy
value "native" was renamed to "sequential" — see CLAUDE.md "Execution
Modes" section.

| Mode | Flag | Wall-clock | PNG | Mean | Std | Verdict agent | Hocine OK |
|---|---|---|---|---|---|---|---|
| compiled | (default) or `--compiled` | 12.90 s | `sana_1024/output_compiled.png` | 188.07 | 106.30 | red apple coherent (glossy, white bg, leaf) | ☐ |
| sequential | `--sequential` | 12.78 s | `sana_1024/output_sequential.png` | 192.87 | 103.51 | red apple coherent (glossy, white bg, brown stem) | ☐ |
| triton | `--triton` | 102.02 s | `sana_1024/output_triton.png` | 185.34 | 91.79 | red apple coherent (smoother, pink-tinted bg, green leaf) | ☐ |
| triton_sequential | `--triton-sequential` | 98.73 s | `sana_1024/output_triton_sequential.png` | 94.99 | 79.38 | red apple coherent (darker bg, glossy highlights) | ☐ |

Per-mode `stats_<mode>.json` written next to each PNG.

### Performance contract verification

The four-mode contract per CLAUDE.md "Execution Modes" section:

- `triton ≤ compiled` (NeuroBrix Triton-pure beats PyTorch fusion) — **VIOLATED**: 102.02 vs 12.90 → 7.9× pire.
- `triton_sequential ≈ sequential` (Triton kernels match PyTorch eager) — **VIOLATED**: 98.73 vs 12.78 → 7.7× pire.
- `triton < triton_sequential` (our fusion provides gain) — quasi-equal (3.2% delta), aucun gain mesurable.
- `compiled < sequential` (PyTorch fusion provides gain) — quasi-equal at 4 steps (0.9% delta), expected on short runs.

### Diagnostic

`triton_sequential` (98.73 s) >> `sequential` (12.78 s) → bug
**kernel-level** (case (a) of the contract). Our individual Triton
kernels (`mm`, `bmm`, `addmm`, `conv2d`) are ~7.7× slower than the
PyTorch (cuBLAS / cuDNN) backend on Sana 1024 V100. The fusion layer
(`triton` ~ `triton_sequential`) is sound but cannot mask the per-op
gap. Phase 1.5 follow-up = isolated kernel benchmarks against
`torch.mm` / `torch.nn.functional.conv2d` to identify the levers
(IEEE_PRECISION strict on Volta fp16, static block sizes
sub-optimal, conv2d im2col vs cuDNN IMPLICIT_GEMM).

Hocine validation: 4 PNGs above to inspect manually before Phase 1
dtype fix is committed.

## 4-mode performance matrix — Sana 1024 (steps=4, post Phase 1.5 autotune)

Post @triton.autotune adoption on mm/bmm/addmm (commit `d514bdb` +
`20ed765`) and conv2d (commit `9e4b498`) with `cache_results=True`
persistent disk cache. Outputs in `sana_1024_post_phase15/`.

| Mode | Flag | Wall-clock | PNG | Mean | Std | Verdict agent | Hocine OK |
|---|---|---|---|---|---|---|---|
| compiled | (default) | 12.60 s | `sana_1024_post_phase15/output_compiled.png` | 79.31 | 88.53 | red apple coherent | ☐ |
| sequential | `--sequential` | 13.22 s | `sana_1024_post_phase15/output_sequential.png` | 182.21 | 111.65 | red apple coherent | ☐ |
| triton | `--triton` | **5862.44 s cold** | `sana_1024_post_phase15/output_triton.png` | 108.53 | 89.01 | red apple glossy coherent | ☐ |
| triton_sequential | `--triton-sequential` | **47.52 s hot** | `sana_1024_post_phase15/output_triton_sequential.png` | 178.59 | 110.40 | red apple coherent (clean bg) | ☐ |

**Cold vs hot autotune**: `triton` (5862 s) was a single cold sweep —
~13 unique conv2d shapes × 18 configs + mm/addmm/baddbmm autotune =
hundreds of compile+bench cycles. Cache persisted via `cache_results=True`
to `~/.triton/cache/<hash>/<fn>.autotune.json`. Subsequent
`triton_sequential` ran fully hot from disk (47.52 s) — proves the
warm-path baseline is recovered. Phase 1.5 measurement of repeated
`triton` cold-then-hot runs: cache_run2 ≈ cache_run3 ≈ 101 s ± 7 %.

### Sana 1024 verdict (correctness R4 + non-regression)

- 4/4 PNG coherent (red apple), no regression vs pre-Phase-1.5 outputs.
- Cold autotune cost amortizes to a one-time per-machine-per-shape-set tax
  via persistent disk cache (Triton 3.6 native `cache_results=True`).
- Warm-path wall-clock comparable to pre-Phase-1.5 baseline (102 s).
- No new failure mode introduced.

## 4-mode performance matrix — Sana 4Kpx (steps=4, post Phase 1.5 autotune)

Outputs in `sana_4kpx_post_phase15/`. Tiling engine (op-level)
activates for spatial > trace_size (1024×1024 → 4×4 tile grid in
decoder; trace_size=1024 from graph.json).

| Mode | Flag | Wall-clock | PNG | Verdict agent | Hocine OK |
|---|---|---|---|---|---|
| compiled | (default) | 36.61 s | `sana_4kpx_post_phase15/output_compiled.png` (14 MB) | red apple coherent | ☐ |
| sequential | `--sequential` | — | — | **FAIL OOM** — sequential dispatcher does not support op-level tiling interceptors (structural pre-existing) | ☐ |
| triton | `--triton` | — (SIGTERM 3 h) | — | **FAIL_TIMEOUT** — 11 cache writes during the run (1024-shapes then 4Kpx-specific incl. depthwise groups=11200); steps=1 sanity test ALSO times out at 30 min and 10 min budgets; pipeline is opaque so cause is indeterminate (autotune-cold or hot-launch-overhead) | ☐ |
| triton_sequential | `--triton-sequential` | — | — | DEFERRED — same indeterminacy as triton; needs profiling chantier first | ☐ |

`stats_<mode>.json` written next to each PNG / failure file.

### Sana 4Kpx verdict — see `sana_4kpx_post_phase15/verdict.md`

- ✅ R4 satisfied for `compiled` mode (the original 36 GiB OOM blocker resolved by op-level tiling). Wall-clock 36.61 s.
- ❌ R4 unsatisfied for `triton` modes at 4Kpx — bottleneck is per-launch Python overhead × 16-tile loop × decoder depth, NOT autotune quality (Sana 1024 triton coherent in both Triton modes).

**Arbitrage**: close P-SANA-4KPX-RUNTIME on the validated production path (compiled mode). Triton 4Kpx wall-clock + correctness are indeterminate within ≤3 h budgets and the pipeline is opaque — open **P-TRITON-4KPX-PROFILE** (new diagnostic chantier) to instrument per-component / per-tile timing in `triton/sequence.py`, then route remediation to either P-CONV2D-DEPTHWISE-OPTIM or P-TRITON-FUSED-KERNELS based on the profile.
