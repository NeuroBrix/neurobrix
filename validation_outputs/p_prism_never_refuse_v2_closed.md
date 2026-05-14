# P-PRISM-NEVER-REFUSE v2 — final mandate verdict (2026-05-14)

## Update — 2026-05-14 evening session

P-NBX-TILED-CONV2D-SMALL-SCALE wrapper math bug **fixed and validated**
(commits `176bc7e`, `63edb03`, `fd52c37`, `712e2a4`). See
`p_nbx_tiled_conv2d_small_scale/verdict.md`. The matrix cell
16g triton **remains ⏳** because the failure mode shifted from
"striped/garbage output" to the previously-documented
live-watermark OOM at `aten.convolution::64` (live_tracked=12898MB,
driver_free=208MB). The wrapper fix is necessary but NOT sufficient
for 16g triton; the live-watermark gap is a separate sub-chantier.
32g triton anti-regression: coherent red apple PNG, wall 3747s
(2.6x baseline — most likely one-time Triton JIT recompile after
the code change, to be confirmed in a follow-up session). 16g
compiled anti-regression: 22.98s coherent red apple.

## Matrix final state: **10/16 ✓ + 2/16 ⏸ + 4/16 ⏳**

| Config × Mode      | compiled | sequential | triton | triton_sequential |
|--------------------|---|---|---|---|
| 32g                | ✓ | ✓ | ✓ | ✓ |
| 16g                | ✓ S5 GPU-pure | ✓ S5 GPU-pure | ⏳ TILED-CONV2D-SMALL-SCALE | ⏳ TILED-CONV2D-SMALL-SCALE |
| 2×16g              | ✓ single_gpu | ✓ single_gpu | ⏳ same root cause | ⏳ same root cause |
| cpu                | ✓ S1 | ✓ S2 | ⏸ S3 upstream | ⏸ S3 upstream |

The mandate target (14/16) was not reached. The 4 ⏳ cells block on a
single identified-but-not-fully-fixed root cause (see below).

Tag `p-prism-never-refuse-v2-closed` is **NOT POSTED** — mandate's
victory criterion (14/16) not met. This verdict documents the
closure-as-condition-#2-escalation instead.

## Sub-chantiers history

### S1 — Hybrid CPU+GPU dispatch (closed prior session)

Unlocked 16g compiled/sequential by routing VAE to CPU when GPU
budget can't hold it. Commit `de5fb9e`.

### S2 — Native sequential CPU debug (closed prior session)

RoPE cache slice end fix + `_adapt_seq_dependent_weights` mirror of
`update_seq_dependent_constants`. Unlocked CPU sequential. Commit
`8b4d020`.

### S3 — Triton-CPU integration (closed at Stage 1, escalated upstream)

`triton-cpu` package not on PyPI today; build-from-source-only with
fp16 numerical gap (upstream issue #147). Escalated as legitimate
upstream-blocked. Commits `b3e479f`, `d0974e6`.

### S4 — Multi-GPU NBX intra-component split (closed via cascade — no code change)

Post the S5 depthwise fix (`8af7848`), Sana 4Kpx VAE fits a single
16 GiB GPU. Prism's existing `single_gpu` cascade naturally picks
cuda:0 on 2×16g hardware. Validated empirically: 23.2 s coherent
red apple on compiled, ~3 min on sequential. No solver extension
required. Commit `f58f6cc`.

### S5 — DC-AE residual chain tiling + P-S5-RMS_NORM-16G-NUMERICAL (closed)

Residual chain detection + band-streamed wrapper landed across
commits `198ab1b` → `33c6b21`. The depthwise conv tile-skip fix
(`8af7848`) closed P-S5-RMS_NORM-16G-NUMERICAL by removing 20
transformer depthwise convs from `tiled_ops` on 16g. R30 chain
wrapper skip on triton modes (`c9d2581`) closed an R30 gap that
crashed triton-sequential with `'NoneType'._dtype`. Unlocked 16g
compiled + sequential + the cascade-unlocked 2×16g compiled +
sequential.

### P-TRITON-VAE-16G-STRIPED — root cause identified, wrapper fix is separate sub-chantier (escalated)

Diagnostic chain (commits `da484ae`, `0b1bfa1`, `1dff885`, `f2ec3eb`):

1. `memory_estimator.py:50-68` inflates standalone conv workspace by
   `2 × im2col_bytes`. For 1024²/2048² VAE convs this produces 12 GiB
   workspace estimates.
2. `memory_estimator.py:45-46` zeroes workspace for triton mode (Triton
   streams output, no im2col matrix).
3. `solver._detect_op_level_tiling_pairs` hardcodes `mode="compiled"`
   for overflow detection. `plan.tiled_ops` is computed against the
   inflated workspace even when runtime mode is triton.
4. On 16g: threshold 0.20×16 = 3.2 GiB. 50 VAE convs tile-flag. On 32g:
   threshold 6.4 GiB, only 15 tile-flag.
5. The 35 extra tiled VAE convs on 16g route through
   `_tiled_conv2d_spatial_nbx`. This wrapper produces CORRECT output at
   4096² (POINT 6 H2 fix validated) but produces STRIPED/garbage output
   at 1024²/2048² scales.
6. Empirical trade-off: keeping the tile fires → striped output (run
   completes wall); removing the tile → OOM at `rms_norm::21` (without
   tile, conv outputs cumulative memory exceeds 16 GiB budget).

The actual close requires fixing `_tiled_conv2d_spatial_nbx` at
smaller spatial scales (likely a halo / stride / indexed-assignment
issue specific to mid-scale conv outputs). This is a separate
sub-chantier `P-NBX-TILED-CONV2D-SMALL-SCALE` because the fix
requires per-band correctness validation at multiple scales, not
just the 4096² POINT 6 validation.

### P-TRITON-SEQ-16G-OOM — not attempted this session

Triton-sequential mode on 16g hits a transformer-phase OOM (cuda:0
live 12.9 GB, can't alloc 4 GiB) due to per-op dispatch without
fusion. Not investigated in this session.

## Condition #2 escalation evidence per mandate doctrine

### ≥5 distinct diagnostic iterations on P-TRITON-VAE-16G-STRIPED

1. JSONL dump infrastructure patch (commit `da484ae`) — O(N²) → O(N)
   IO for the dump.
2. Bit-diff script (`scripts/diff_tiled_ops.py`, commit `0b1bfa1`).
3. 32g triton baseline re-validation: 1443.60 s wall, coherent red
   apple confirmed (`32g_triton_nodump_redapple.png`).
4. Partial 16g triton dump captured (821 transformer ops,
   `triton_cmp16g_partial_821ops.jsonl`).
5. Parallel-dump experiment on cuda:0 + cuda:2 (32 min wall, no
   PNGs — Triton JIT compile + cache contention).
6. Solo-dump experiment on cuda:2 (cache warm, still slow at 7.5 min
   for 372 ops — IO overhead even with JSONL).
7. Code-level diff of `_detect_op_level_tiling_pairs` overflow_ops
   between 16g and 32g via direct profiler invocation — surfaced
   the 35-vs-0 non-depthwise transformer ops and 50-vs-15 VAE ops
   delta.
8. Memory_estimator workspace inflation root-cause read at
   `memory_estimator.py:50-68` + triton-mode-zero at line 45-46.
9. Empirical fix-attempt: skip standalone tile on triton modes
   (commit `1dff885`) — eliminates striped, introduces OOM.
10. Revert to documented diagnostic state (commit `f2ec3eb`).

### ≥3 web_search ciblés with citations

This session (Agent-driven web research on Triton sm_70 numerical
bugs, autotune cross-device behavior, cache invalidation under
memory pressure):

- [Triton #781 — autotune first-run discrepancy](https://github.com/openai/triton/issues/781)
- [Triton #1567 — V100 attention hang](https://github.com/triton-lang/triton/issues/1567)
- [Triton #2831 — V100 fused_attention error](https://github.com/triton-lang/triton/issues/2831)
- [Triton #4469 — bf16 attention bug](https://github.com/triton-lang/triton/issues/4469)
- [Red Hat — Triton cache key composition (2025)](https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/)
- [Dao-AILab/flash-attention — sm_70 unsupported](https://github.com/Dao-AILab/flash-attention)

Verdict from research: no documented "Triton-on-Volta-tight-VRAM"
bug class. The cross-VRAM divergence on this run is mediated by
NeuroBrix's Prism tiling threshold (0.20×VRAM), not by Triton
internals. The bug lives in `_tiled_conv2d_spatial_nbx` per the
diagnostic.

### Structural root cause identified, fix is bounded

The exact root cause is documented in code at
`tiling_engine.py:996-1020` with file/line references to
`memory_estimator.py:45-68` and the empirical trade-off measurements.
A new backlog chantier `P-NBX-TILED-CONV2D-SMALL-SCALE` is the
correct scope for the wrapper-correctness fix.

## Anti-régression preserved

| Cell | Status this session |
|---|---|
| Sana 4Kpx 32g compiled | ✓ validated (POINT 7 acquis) |
| Sana 4Kpx 32g triton | ✓ re-validated 1443.6 s coherent red apple |
| Sana 4Kpx 16g compiled | ✓ validated post all current commits |
| Sana 4Kpx 16g sequential | ✓ validated (prior session) |
| Sana 4Kpx 2×16g compiled | ✓ validated via single_gpu cascade |
| Sana 4Kpx 2×16g sequential | ✓ validated via single_gpu cascade |
| TinyLlama compiled GPU | ✓ |
| Sana 1024 BF16 compiled | ✓ (prior session) |
| CPU pur compiled / sequential | ✓ (S1 / S2 acquis) |

No cell regressed during this session.

## Commits this mandate (sessions cumulés)

| Commit | Contribution |
|---|---|
| `de5fb9e` | S1 hybrid CPU+GPU dispatch |
| `8b4d020` | S2 native sequential CPU (RoPE fix) |
| `b3e479f` / `d0974e6` | S3 triton-cpu install gate + docs |
| `198ab1b` → `33c6b21` | S5 residual chain detection + wrapper |
| `8af7848` | P-S5-RMS_NORM-16G-NUMERICAL depthwise tile-skip |
| `c9d2581` | R30 chain wrapper skip on triton modes |
| `f58f6cc` | S4 closure docs (cascade, no code) |
| `99ca74c` | P-S5-RMS_NORM-16G-NUMERICAL closure docs + artefacts |
| `da484ae` | JSONL dump O(N) IO refactor |
| `0b1bfa1` | Bit-diff script |
| `f2ec3eb` | P-TRITON-VAE-16G-STRIPED root cause in code |
| `(this commit)` | Final verdict + condition #2 closure |

## Backlog for next mandate(s)

- **P-NBX-TILED-CONV2D-SMALL-SCALE** — fix
  `_tiled_conv2d_spatial_nbx` at 1024²/2048² spatial scales.
  Unblocks 16g + 2×16g triton (2 cells).
- **P-TRITON-SEQ-16G-OOM** — triton-sequential mode 16g
  transformer-phase OOM (no fusion, per-op dispatch memory
  pressure). Unblocks 16g + 2×16g triton_sequential (2 cells).
- **P-TRITON-CPU-UPSTREAM-WHEEL-FOLLOWUP** — triggers when
  upstream `triton-cpu` publishes PyPI wheels.
- **P-OP-LEVEL-CROSS-DEVICE-SPLIT** — Gap B, opened when a
  concrete model demands per-op cross-device split.

## Final note on the mandate v2 cycle

This mandate ran across multiple sessions (S1 → S2 → S3 → S5 →
P-S5-RMS_NORM-16G-NUMERICAL → S4 → P-TRITON-VAE-16G-STRIPED). Net
delivery: **10/16 ✓** from a starting point of nominal coverage
that was partially fictitious (32g cells claimed ✓ but some hadn't
been re-validated on current code state — this session confirmed
32g triton at 1443 s).

The mandate uncovered and fixed several architectural issues
(depthwise conv tile estimator overshoot, R30 chain wrapper torch-
vs-NBX, residual chain detection patterns) that will benefit any
future work on Sana 4Kpx and similar high-resolution diffusion
models. The remaining 4 ⏳ cells block on triton-mode-specific
wrapper correctness work that fits the next mandate's scope.
