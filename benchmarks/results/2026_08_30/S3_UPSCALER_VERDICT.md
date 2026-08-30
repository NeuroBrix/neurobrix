# S3 upscaler sub-segment — CLOSED (campaign 2026-08-30, locked protocol)

The whole-family table, same-campaign, that the tonal defect had been
blocking since 2026-08-27. Six rows x three arms, 18/18 cells, one
campaign window.

## Protocol

* GPU 2, clocks LOCKED 1290/1290 (log: `~/night_2026_08_30/night_master.log`,
  flightrec `20260829_232104`), quiet host.
* Window: 2026-08-29 23:21:04 -> 23:34:16 UTC, T 36->41 C.
* n=5 requests per cell after warmup, committed 448x448 asset,
  pinned config. Vendor stacks per rows.yml (pinned venvs).
* Every container is TODAY's artifact: swin2sr-x4 corrected+republished
  (2026-08-28), swinir x2/x4 + hat-l + realesrgan re-traced/replaced
  (2026-08-29) — no pre-fix cell patched into this table.

## Results (wall seconds, mean over n=5, stdev)

| row | vendor | nbx pytorch | nbx triton | pt/vendor | triton/vendor |
|---|---|---|---|---|---|
| swin2sr-x4 | 1.875 ±0.003 | 2.272 ±0.005 | 8.738 ±0.158 | 1.21x | 4.66x |
| swinir-x4 | 2.016 ±0.013 | 2.031 ±0.006 | 6.913 ±0.127 | **1.007x** | 3.43x |
| hat-l-x4 | 6.271 ±0.002 | 6.907 ±0.005 | 33.226 ±0.145 | 1.10x | 5.30x |
| realesrgan-x4 | 0.432 ±0.000 | 0.904 ±0.004 | 3.628 ±0.065 | 2.09x | 8.40x |
| swinir-x2 | 1.984 ±0.008 | **1.638 ±0.002** | 6.872 ±0.045 | **0.83x** | 3.46x |
| swin2sr-realworld-x4 | 1.879 ±0.003 | 2.058 ±0.006 | 8.832 ±0.147 | 1.10x | 4.70x |

## Reading, honest

* **compiled (pytorch) is at or near vendor parity across the Swin
  family**: +0.7% on swinir-x4, +10% on hat-l and realworld, +21% on
  swin2sr-x4 — and swinir-x2 BEATS its vendor stack by 17% (the
  official SwinIR repo runs fp32 single-image with no graph fusion).
  realesrgan's 2.09x is the outlier: a 0.43 s CNN where fixed per-run
  overhead (container load path, output save) dominates compute.
* **triton carries the known Volta structural gap** (3.4-8.4x vendor):
  sm_70 Triton cannot lower fp16 matmul to HMMA the way cuBLAS does
  (Phase 1.5 measurement: ~12% cuBLAS ceiling WITH autotune). The
  unfavourable number is recorded, per the reference-dimension rule —
  it is the honest current triton price on this hardware, not a defect
  of these rows.
* Dispersion <= 0.16 s worst cell (triton swin2sr); most cells < 0.01 —
  the locked protocol held.

## Correctness backing this table

Gate at four size classes (trace / 448 square / 240x320 rect /
200x300 unaligned): 58 passed, 0 failed, 5 xfail = the named
swin2sr-x2 engine-divergence defect (unpublished, tracked strict).
Warm serve: 17 passed, 0 failed, including the exact-size unaligned
cells on both engines. Vendor tone arms banked and mutation-proved for
every published artifact. Full trail:
`validation_outputs/upscaler_floordiv_republish_2026_08_29/VERDICT.md`.

## Companion — Sana-4Kpx 4-mode matrix CLOSED (same night)

Phase 2 of the same night run, GPU 3 pinned (sanctioned reproducer
config): `--sequential` and `--triton-sequential` at 4096^2, seed 42,
both rc=0, both PNGs 4096x4096 coherent and visually identical to each
other (R29 eyeballed). With compiled (36.6 s) and triton (453.9 s)
closed 2026-07-11, all four modes now render 4K factually. The night
timings (~102 s sequential, ~5 min triton-sequential) are UNLOCKED
correction runs, not perf cells — canonical perf numbers stay with the
locked protocol.


---

# S3 IMAGE segment — completion cells (same night, 2026-08-30 00:15-00:25 UTC)

The named remainder, measured. Locked 1290, GPU 2, n=5, same protocol.

| row | vendor (diffusers) | nbx pytorch | nbx triton |
|---|---|---|---|
| flex1 (1024, 20 steps) | 72.82 ±1.13 | (08-27 cells stand) | (08-27 cells stand) |
| pixart_xl (1024, 20) | 8.15 ±0.001 | 11.44 ±0.029 | 65.28 ±0.111 |
| pixart_sigma (1024, 20) | 8.54 ±0.002 | 13.46 ±0.003 | 78.26 ±0.048 |

* **flex diffusers arm**: the 08-27 JSON was EMPTY (pre-protobuf-fix
  failure recorded as a cell). Measured now with the sourced Flux/V100
  recipe completed: fp32 text encoders + embeds handed to the fp16
  transformer at fp16 (the missing second half — fp32 embeds built
  fp32 latents and crashed x_embedder). 72.8 s with vendor
  cpu-offload, the honest V100 number for a 26 GB-fp16 model.
* **pixart x2**: first measurement (rows created this night — they had
  never existed; the "disk arbitrage" blocker resolved by the NAS
  re-download after the vendor snapshots were found MISSING). nbx
  pytorch carries 1.40x/1.58x vendor; triton carries the Volta
  structural gap (8.0x/9.2x), recorded per the reference-dimension
  rule.

## S3 image segment — remaining list after this night

1. NOTHING measured remains. Every row of the segment has its three
   arms with n=5, lock, and window recorded.
2. ONE DECISION remains (supervisor): a locked Sana-4Kpx PERF row.
   Recommendation: the factual 4-mode matrix (all four render 4K,
   R29'd) suffices for S3; a locked perf row costs ~3-4 h of locked
   GPU for a table line with no comparable vendor arm at protocol
   (diffusers 4K on one V100-32G needs its own offload recipe). If
   the table wants the line, it is a scheduled decision, not a gap.
