# CogVideoX-2b — VERDICT: CLOSED 4/4 at CFG batch=2 (single-GPU, native 480×720, 13f-class)

**Date:** 2026-06-26 · **Build:** runtime ≥ `f42917f` · **Hardware:** V100 32 GB (`cuda:2`), `single_gpu`
**Artifacts:** `../video_cfg2/CogVideoX-2b/` — frames `m_*_f6.png`, videos `m_*.mp4`,
`stats.json`, the seam A/B pair. This file is the **canonical** verdict; it supersedes
the earlier 06-11 4-mode-at-f9 note (which predated the batch/CFG and seam criteria).

## Result — genuine 4/4

All four execution modes render a **coherent red fox in a snowy forest**, in
agreement, with no tile seams and no NaN. Visually inspected per mode (R29), not
stats-only.

| Mode | std | mean | max_row_jump | NaN | coherent | corr vs compiled |
|------|-----|------|--------------|-----|----------|------------------|
| compiled            | 66.5 | 198 | 6.5 | no | ✅ fox | 1.0000 |
| sequential (oracle) | 66.6 | 198 | 6.5 | no | ✅ fox | 0.9983 |
| triton              | 64.2 | 195 | 6.2 | no | ✅ fox | 0.9881 |
| triton_sequential   | 67.1 | 196 | 6.9 | no | ✅ fox | 0.9908 |

Config: native **480×720, 13 frames, cfg=6 (batch=2), seed=42**, 25-step matrix
(50-step beauty `m_compiled_50step_f6.png`).

## The three closure criteria

### 1. Frame cohérente (per mode)
Four independent compute paths — torch (`compiled`/`sequential`) and NBXTensor+Triton
(`triton`/`triton_seq`) — each produce the same coherent fox. The VAE compute is
per-mode (torch `conv3d` vs triton `_conv3d_via_conv2d`), so both structurally-
independent VAE paths are proven to decode coherently, not inferred from one another.

### 2. Drift-gate (cross-engine, matched config/seed/steps)
Mid-frame pixel agreement on the identical seed=42 initial noise:
- **compiled ↔ triton (cross-engine, different DtypeEngine): corr 0.9881, relL2 0.0516** — the headline cross-engine gate.
- compiled ↔ sequential (torch mirror): 0.9983 · triton ↔ triton_seq (triton mirror): 0.9930 · sequential ↔ triton_seq (kernel oracle vs torch oracle): 0.9911.

The 0.988 cross-engine band is the same correctness band used against the vendor
pipeline. Seed-determinism holds across the torch↔NBXTensor boundary.

### 3. Batch / CFG exercised ≠ trace
CogVideoX-2b is traced at **batch=2** (cfg). Running **cfg=1.0 → batch=1** exercises
the batch symbol at a value ≠ trace: coherent, no NaN (`m_compiled_cfg1_batch1_f6.png`).
Spatial (native 480×720 vs ~112×176 trace) and temporal (13f) are likewise ≠ trace.

## Seam debunk — a FIXED NeuroBrix Prism over-tiling bug (root-fixed, not accepted)

The seams in the prior (pre-`f42917f`) outputs were a **NeuroBrix Prism placement
bug**, not model content and not a seam to accept:

- **Root cause (verified at `f42917f`, 2026-06-19 — touches `cli/commands/run.py` +
  `core/prism/solver.py`):** the CLI built the Prism activation-profiling resolution
  with a hardcoded `1024` fallback, while the runtime executor falls back to the
  *family* config (512² for video). A video model without explicit dims was thus
  profiled at 1024² — ≈8× the real activation (4× spatial × 2× CFG batch) — giving a
  VAE estimate of 45 GB that force-tiled / weight-sharded a VAE which decodes natively
  in ~5.6 GB. The tile halos produced the hard horizontal seams.
- **Fix:** mirror the executor's fallback chain (args → model defaults → family
  config → 1024) when building the profiling InputConfig. Models that specify their
  resolution are unchanged (R23-inert elsewhere).
- **What the A/B PROVES — the placement decision changed (same 512² square config):**

  | build | placement (from log) | max_row_jump | frame |
  |-------|----------------------|--------------|-------|
  | pre-`f42917f` | `TilingEngine` | 88.3 (hard seam @row161) | `seam_BEFORE_512_pre-f42917f.png` |
  | current       | `single_gpu` (no tiling) | 28.5 | `seam_AFTER_512_single_gpu.png` |

  The smoking gun is the **placement log**: same 512² config, old → `TilingEngine`,
  current → `single_gpu`. With no tiling in the current build there is no tile
  boundary, and the BEFORE frame's hard seam is gone. The 512² AFTER frame still shows
  mild banding; we do **not** certify its cause (512² is a non-native square for this
  480×720 model and we did not vendor-repro it). The closure does **not** rest on the
  512² frame — it rests on native-resolution coherence below.

- **The closure rests on:** native-resolution **480×720 4-mode coherence** (clean
  foxes, max_row_jump 6.5, no banding) + the **over-tiling root fix** above.

- **VAE is symbolic, not frozen:** it decoded native 480×720 (≈ 4× the 112×176 trace
  extent) at `single_gpu` with zero tiling → this directly rebuts the reopening worry
  that "le VAE spatial gelé à la petite taille de trace". No forge re-symbolization was
  needed; the open item was a placement estimate, now fixed born-at-source in Prism.

## Reproduce

```bash
# coherent matrix (per mode), native config, batch=2
for M in --compiled --sequential --triton --triton-sequential; do
  python3 -m neurobrix run --model CogVideoX-2b \
    --prompt "a red fox walking in a snowy forest, cinematic" \
    $M --cfg 6 --height 480 --width 720 --num-frames 13 --steps 25 --seed 42 \
    --mode t2v --output /tmp/cogx2b_$M.mp4
done
# batch=1 exercise (cfg=1.0 -> batch=1 != trace batch=2)
python3 -m neurobrix run --model CogVideoX-2b --prompt "a red fox walking in a snowy forest, cinematic" \
  --compiled --cfg 1.0 --height 480 --width 720 --num-frames 13 --steps 50 --seed 42 --mode t2v
```

## Hocine validation: TODO
Visual: `../video_cfg2/CogVideoX-2b/m_{compiled,sequential,triton,triton_seq}_f6.png`
(four coherent foxes) + the seam A/B pair (`seam_BEFORE_512_pre-f42917f.png` hard
seam vs `seam_AFTER_512_single_gpu.png` no tile boundary).
