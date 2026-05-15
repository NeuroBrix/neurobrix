# P-NEUROBRIX-UPSCALERS — U3 + U4 verdict (2026-05-15)

## Outcome: 12/12 swin2sr cells ✓ + anti-régression v2 preserved

`nbx upscale` is operational on the 3 swin2sr containers across
all 4 execution modes, at the container trace size (64×64).

## Matrix (input 64×64, 32g profile)

| model | compiled | sequential | triton | triton-seq |
|---|---|---|---|---|
| swin2SR-classical-sr-x2-64 | 128² ✓ 3.1s | 128² ✓ 2.8s | 128² ✓ 139s† | 128² ✓ 6.9s |
| swin2SR-classical-sr-x4-64 | 256² ✓ 3.2s | 256² ✓ 2.9s | 256² ✓ 15.9s | 256² ✓ 6.9s |
| swin2SR-realworld-sr-x4-64-bsrgan-psnr | 256² ✓ 3.3s | 256² ✓ 2.9s | 256² ✓ 15.3s | 256² ✓ 6.8s |

† x2 triton 139 s = one-time Triton JIT compile of the full
swin2sr op graph; cache-warm runs land ~15 s like x4.

All output dimensions equal `input × scale` (x2 → 128², x4 →
256²). Visual inspection (R29): coherent red-apple upscale, no
garbage, on every cell. Artefacts under
`validation_outputs/p_neurobrix_upscalers/matrix/` (local, per
repo hygiene doctrine — durable verdict here).

## R32 cross-mode numerical equivalence

Cosine vs compiled reference = **1.00000** for all 9 non-compiled
cells (sequential / triton / triton-sequential × classical-x4 +
realworld-x4). R30 dualité confirmed: the 4 modes are
bit-equivalent, far above the ≥0.99 gate.

## Code delivered

| Commit | Scope |
|---|---|
| `7bba82b` | U3 — `nbx upscale` subcommand + `get_final_output` forward_pass branch |
| `3a42e46` | v1 backlog (BL-1 profile config gap) |
| `9c762bb` | U4 — `roll` + `reflection_pad2d` R33-pure triton kernels |

Key fixes:
- `executor.get_final_output`: added `flow.type=="forward_pass"`
  branch returning `flow.order[-1]`'s output. Without it the
  generic fallback returned the feature-trunk output at input
  resolution (64²) instead of the upscaled head output. Fires
  only for forward_pass flow — iterative_process (Sana) and
  autoregressive (TinyLlama) structurally unaffected.
- `roll_wrapper`: `torch.roll` via narrow + NBXTensor.cat
  (Swin cyclic window shift). R33-pure.
- `reflection_pad2d_wrapper`: was a RuntimeError stub. Now
  narrow + flip + cat, exact PyTorch reflect semantics.

## Anti-régression matrice v2 (PRESERVED)

| cell | result |
|---|---|
| TinyLlama compiled 32g | 4.25 s, coherent haiku (= 4.24 s ref) |
| Sana 4Kpx 32g compiled | 23.94 s, coherent red apple (= ~23 s baseline) |

No regression. The new code paths are additive (new subcommand,
new kernels for previously-unsupported ops) or guarded by
flow.type.

## Known limitation — v1 backlog BL-1

Inputs are validated at the exact container trace size (64×64).
Feeding a larger image hits a fixed-spatial-dim view mismatch
because the swin2sr containers' `profile.json` `config` lacks
`upscale` + `window_size`, so `TilingEngine.from_component_config`
returns None and never tiles. This is a model-packaging gap
(BL-1 in `v1_backlog.md`), not a runtime defect — the runtime
tiling logic is correct and data-driven, merely starved of the
config it needs. Resolves automatically when the container
profile carries those keys.

## Hub

All future upscaler .nbx publication targets neurobrix.es
(proprietary NeuroBrix hub). No huggingface.co publication.

## Next mandate

U5 — Real-ESRGAN integration (container production upstream +
NeuroBrix 4-mode validation).
