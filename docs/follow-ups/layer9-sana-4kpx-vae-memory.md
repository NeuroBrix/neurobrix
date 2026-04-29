# Layer 9 — Sana 4Kpx VAE memory blocker (post-Layer 8)

**Status:** open — blocks `Sana_1600M_4Kpx_BF16` in `--triton`,
`--triton-sequential`, AND `native` on V100 32 GB after Layer 8
unblocked the spatial-symbol promotion regression.

**Out of scope of:** Layer 8 (engine-side shape-rebind audit).

## Symptom

Sana 4Kpx now advances **5 cascading 2× upsamples** in both triton
paths and native — from the `64×64` build-time latent through
`128×128 → 256×256 → 512×512 → 1024×1024 → 2048×2048 → 4096×4096`.
The pipeline reaches the final 4096×4096 convolution stage of the
DC-AE decoder (op ~700/720) and crashes:

- **`--triton-sequential`** at `aten.convolution::62`:
  `GPU malloc failed (error 2) for 4294967296 bytes` (4 GiB output
  tensor allocation).
- **`--triton` (compiled, with arena drain)** at the same conv: same
  4 GiB allocation failure.
- **`native`** at `aten.convolution::55`:
  `CUDA out of memory. Tried to allocate 36.00 GiB. GPU 2 has a total
  capacity of 31.74 GiB` (PyTorch picks a non-optimal cudnn algorithm
  with a 36 GiB workspace at full 4Kpx scale).

Native and both triton paths fail. The model is not producible on
V100 32 GB without a memory-engineering change.

## Why Layer 8 didn't fix this

Layer 8 closed the engine-side spatial-rebind regression: literal
build-time H/W/HxW values baked into `aten::view` / `aten::expand` /
`aten::reshape` shape args are now promoted to `s_h * s_w` symbolic
expressions and rebind correctly to runtime values. Sana 4Kpx
advances through every cascade stage (each 2× wider than the last)
without shape mismatches.

What it didn't address: at the FINAL stage the activations are
genuinely too large for V100 32 GB. The 4096×4096 conv needs:

- **Output tensor**: `1 * 128 * 4096 * 4096 * 2 (fp16) = 4 GiB` per
  intermediate.
- **Cudnn workspace** (native): up to 36 GiB depending on algorithm.
- **Live previous intermediates**: one or two prior cascade outputs
  still alive during the last conv (~10–20 GiB cumulative).

Even with aggressive arena draining and intermediate eviction the
peak working set exceeds V100 32 GB.

## Solution outline (Layer 9)

Three complementary directions, in priority order:

1. **VAE spatial tiling** (preferred). Tile the decoder spatial
   pipeline at the upsample boundaries (e.g., 4 × 2048×2048 tiles
   instead of one 4096×4096) with overlap, blend at seams. Layer 6.3
   correctly disabled the parasitic TilingEngine activation for
   symbolic graphs, but did NOT add the genuine memory-driven path —
   memory-bound graphs at runtime should have a fall-back tiling
   strategy. The contract becomes: *symbolic shapes handle
   resolution-adaptation* (Layer 8 done), *memory-driven tiling
   handles resolution-when-it-doesn't-fit* (Layer 9).

2. **Workspace-aware conv kernel selection**. The cudnn workspace
   bloat (36 GiB for the failing 4096×4096 conv) is algorithm-
   dependent. A `cudnn.benchmark`-style per-shape selection that
   prefers low-workspace algorithms when free memory is tight would
   bring the native path into budget without tiling.

3. **Hardware**. Run on A100 40 GB / 80 GB or H100. Not a fix on
   V100 32 GB; documented as the trivial workaround.

## Validation criteria for Layer 9

- Sana 4Kpx in `--triton`, `--triton-sequential`, and `native`
  produces a coherent 4096×4096 image on V100 32 GB (cosine vae.out
  vs reference ≥ 0.95).
- No regression on Sana 1024 (3 modes) — its cascade peaks at
  1024×1024 (1.5 GB intermediate), already fits without tiling, so
  Layer 9 must NOT activate when memory is sufficient.
- No regression on PixArt × 4 modes — they cap at 1024×1024 too.
- No regression on LLM harness 14/14 — LLM has no spatial cascade.

## Cross-references

- Layer 8 commit (parent): adds spatial-symbol promotion to
  `triton/promotion.py` and native `compiled_sequence.py`. Closes the
  engine regression that masked the memory issue.
- Layer 6.3 (`tiling_engine.py:from_component_config`) — disables
  the parasitic TilingEngine for symbolic-spatial graphs. Was the
  right call for the masking-bug case; needs a separate
  memory-driven re-entry for genuine memory-limited cases.
- `docs/architecture/symbolic-shapes-contract.md` — defines the
  symbolic-shape contract Layer 8 enforces. Layer 9 extends with the
  memory-driven tiling fallback contract.
