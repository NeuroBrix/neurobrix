# S4 video — whole-row re-run annex (2026-09-01)

Campaign: 16:48–20:54 UTC (main) + corrected-cell re-runs, flightrec
`20260901_164810`, clocks locked 1290 on all four GPUs, sequential
cells, n=3 per cell, dispersion ≤ 6% everywhere (mostly ≤ 2.5%).
Supersedes every 2026-08-30 video cell (that campaign's JSONs stand as
findings evidence only — no error cell enters a table). Media + shas:
`validation_outputs/bench_reference_2026_09_01/<row>/`.

## Why the re-run exists (the 08-30 diagnosis, in one paragraph)

Six of the eight failed diffusers arms were OUR harness, not vendor
fragmentation: one generic recipe (`vae=fp32` on every video pipeline,
no tiling, no offload, i2v image never passed) ran pipelines whose own
model cards mandate per-model recipes at 32 GB. Two were sourced
version facts (SanaVideoPipeline needs diffusers ≥ 0.36 — and THIS
checkpoint 0.38.0; Allegro-TI2V was never upstreamed — vendor-only
class, declarative DNR). The 08-30 "the vendor stack cannot start 8 of
these pipelines" capacity sentence was therefore OVERSTATED and is
retired. Rows now carry data-driven `diffusers_recipe` blocks recorded
in each cell's pins (commits `51bf0cd`, `bde795a`, `43c0181`,
`56452b4`, `d9b3d52`).

## Results (median wall s/clip, n=3, short pinned configs)

| row (steps×frames) | diffusers (vendor recipe) | nbx compiled | nbx triton | R29 |
|---|---|---|---|---|
| cog2b t2v (10×9) | 8.9 (±1.1%) | 45.4 (±0.7%) | 231.2 (±0.1%) | content PASS both |
| allegro t2v (6×9) | 4.6 (±0.2%) | 36.7 (±1.9%) | 154.1 (±0.3%) | UNCONVERGED both arms¹ |
| allegro ti2v | DNR-VENDOR-ONLY (sourced) | FAILS-AT-HEAD² | same² | — |
| sana-video t2v (8×9) | DNR-VERSION-CASCADE⁵ | FAILS-AT-HEAD² | 39.5 (±2.3%) | fox PASS (triton) |
| cog5b i2v (10×9) | 121.4 (±0.5%) offloaded | 120.2 (±0.6%) | 502.3 (±0.2%) | apple PASS |
| mochi t2v (8×13) | 72.8 (±2.4%) offloaded | 124.8 (±5.9%) | 566.4 (±2.4%) | animal PASS |
| opensora t2v (8×9) | no pipeline (prereg DNR) | 150.6 (±1.0%) | FAILS-AT-HEAD² | fox-mass PASS (compiled) |
| vace r2v (10×13) | 35.8 (±0.1%) ref-image task³ | D-VACE-IMGCOND-DIV62³ | same³ | vendor PASS |
| wan13b t2v | anchored 08-26 | anchored | anchored | anchored |
| wan14b i2v (6×9) | 179.4 (±0.9%) sequential-offload⁶ | D1 co-location⁴ | **557.5 (±0.5%) SHARDED cuda:2+3** | apple PASS (vendor AND sharded triton) |
| wan22 a14b i2v (6×9) | 153.2 (±1.4%) offloaded | D1 co-location⁴ | D7 cpu-pointer⁴ | apple+fox PASS (vendor) |

¹ Allegro at the pinned 6 steps converges on NEITHER arm (vendor frame
uniform gray, nbx frame colored noise) — the row is timing-valid (same
task, same steps) and its media must not be read as showcases. A
vendor-default-steps convergence cell is optional follow-up work.
² Engine findings, artifacts DEPRECATED with user-language reasons
(2026-09-01): D-ALLEGRO-TI2V-ADD5 (both engines),
D-SANAVIDEO-COMPILED-ADD11 (triton renders), D-OPENSORA-TRITON-SDPA
(compiled renders). Each returns to public with its validated fix.
³ The VACE build is traced WITH image conditioning; the parity config
(reference_images on the vendor arm) resolved the 08-30 FAILS-CONFIG
and exposed the real defect: aten.div::62, (1,768,60,104) vs
(1,384,60,104) on BOTH engines — D-VACE-IMGCOND-DIV62, artifact
deprecated (8th), 4-mode diagnosis in the maintenance lot. The 08-30
27.9 s prompt-only vendor cell measured a DIFFERENT task and is
retired.
⁴ Over-card rows on this rig class: compiled machine-config fails at
the D1 op-input co-location gap (aten.add cuda:3-vs-cuda:2, two data
points); wan22 triton hits the D7/zero3 class (CPU-offloaded pointer
reaching a Triton kernel at addmm::405). Loud refusals, artifacts
correct on big-enough single cards → NO deprecation; DETTE D1/D7 get
both cases. wan14b triton machine-config is the positive twin: the
31 GB model RENDERS sharded across cuda:2+3 — the 1d00037 proof is now
a campaign cell.
⁵ Five dated cells, five distinct incompatibility layers (0.35.2 no
SanaVideoPipeline; 0.36.0 no AutoencoderKLLTX2Video; 0.37.0 wrong VAE
config keys; 0.38.0 kwarg name, then its video post-processor reshape
fails at the row's 480×832×9f — the 720p checkpoint's decode emits a
mismatched spatial shape at this size). The vendor stack does not run
this checkpoint at the row config on any tested stable release; a
vendor timing cell, if ever needed, is a dedicated chantier at native
720p. The row's engine cell (nbx triton, 39.5 s, R29 fox) stands.
⁶ Four recipe rungs, each measured (image kwarg → CLIP image encoder
fp32 → model-level offload OOM at 31.3 GB → enable_sequential_cpu_offload):
the vendor weapon at the 27 GB-transformer size is LAYER-level offload
— 9.4 GB peak, 179.4 s/clip, slow by construction and recorded in
pins. Notable inversion: the vendor's per-layer offload on ONE card
runs ×3.1 faster than our sharded triton (557.5 s) — the sharded cell's
value is capacity (compiled cannot place it at all on this rig = D1),
and the triton sharding tax is measured, not assumed.

## Capacity reading (the axis, restated honestly)

**9 of 11 hub video models RENDER on at least one nbx engine at HEAD**
(cog2b, allegro-t2v, sana-video, cog5b, mochi, opensora, wan13b,
wan14b-sharded, wan22 excluded — its nbx arms need D1/D7; allegro-ti2v
and vace are the two deprecated regressions). The vendor stack, given
its OWN per-model recipes (which the 08-30 campaign wrongly withheld),
starts 7 of its 9 attempted pipelines; the two vendor DNRs are sourced
(Allegro-TI2V never upstreamed; SANA-Video a five-layer version
cascade) — the honest competitor statement is "per-model recipe work,
two library generations, and two checkpoints its stack cannot serve at
the row config", not "cannot start eight". The
conjunction still stands where it always did: ONE engine, one
container, one CLI serves the family; the vendor side needs per-model
recipes, two library generations, and offload for anything ≥ 14B on
32 GB cards.

## Speed reading (honest)

- nbx compiled runs ×5.1–×8.0 the vendor wall on the light t2v rows
  (cog2b 45.4 vs 8.9; allegro 36.7 vs 4.6) — the sm_70 structural
  conv/mm gap named since S3, priced per row.
- **cog5b i2v: nbx compiled MATCHES the offloaded vendor arm (120.2
  vs 121.4 s)** — where the vendor recipe pays PCIe offload, resident
  nbx placement cancels the engine gap.
- mochi: vendor offloaded 72.8 vs nbx 124.8 resident (×1.7).
- triton carries its structural gap (×3.4–5.1 vs compiled on video
  rows) — the same named front as S3/prefill; sana-video triton 39.5 s
  is the family's fastest triton cell.
- wan14b: vendor sequential-offload 179.4 s vs nbx sharded triton
  557.5 s (×3.1, note ⁶) — the row's capacity point is that compiled
  cannot place it on this rig at all (D1); the sharding tax is now a
  measured number, not an assumption.

Findings ledger: D-VACE-IMGCOND-DIV62 (new), D1 ×2 compiled
co-location data points, D7 cpu-pointer case, D-ALLEGRO-TI2V-ADD5 /
D-SANAVIDEO-COMPILED-ADD11 / D-OPENSORA-TRITON-SDPA (deprecations),
allegro 6-step unconvergence note. Hocine validation: TODO.
