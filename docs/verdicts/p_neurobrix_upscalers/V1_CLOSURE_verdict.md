# P-NEUROBRIX-UPSCALERS-V1 — Consolidated closure verdict (2026-05-15)

## Outcome: CLOSED — 4 architecture families, 10 containers, 2 engine extensions

NeuroBrix gains a dedicated pure-super-resolution upscaler tier
via the `neurobrix upscale` subcommand, validated end-to-end
across execution modes with numerical equivalence to the
PyTorch fp32 reference.

## Models integrated (10 containers, all path-leak clean)

| family | containers | modes ✓ | cos vs fp32 ref |
|---|---|---|---|
| Swin2SR (U3/U4) | classical-x2, classical-x4, realworld-x4 | 4/4 | 1.00000 |
| Real-ESRGAN (U5) | x2, x4, x8 (RRDBNet CNN) | 4/4 | 1.00000 |
| SwinIR (U6) | classical-x2, classical-x4 | 4/4 | 0.999998 |
| HAT (U7) | HAT-S x4, HAT-L x4 | 2/4 (compiled+sequential) | 0.999998 |

All outputs `input × scale`, R29 visual: coherent super-
resolution on every validated cell. Per-model verdicts:
`U3_U4_verdict.md`, `U5_verdict.md`, `U6_verdict.md`,
`U7_verdict.md`.

## Engine extensions delivered (general, not upscaler-specific)

1. **Per-component `requires_fp32_compute` opt-in** (Prism dtype
   resolution, NeuroBrix `05a39ee`). Architectures whose
   activation range structurally exceeds fp16 (deep
   transformer-SR with no inter-block normalisation — SwinIR,
   HAT) declare this; Prism pins them to fp32 regardless of the
   hardware preferred dtype. Strictly opt-in, byte-identical for
   every existing model. Roots: SwinIR/HAT NaN-in-fp16 proven by
   `model.half().cuda()` repro + monkeypatch gate (cos 1.000000
   forced-fp32).

2. **Reachable constant-slot pre-population** (CompiledSequence,
   NeuroBrix `c0a1445`). The compiled path's constant pre-pop
   was dead code (orphaned after `rebind_partial`'s return);
   any graph with an in-forward orphan scalar constant
   (`mask[slice]=cnt`) crashed compiled mode. Moved into
   `bind_weights` with a 0-dim default. General fix; found via
   HAT (first model to exercise `aten::fill`).

Both verified anti-régression-inert on TinyLlama compiled
(KV-cache, coherent) and Sana 4Kpx 32g compiled (coherent red
apple), re-run after each engine change.

## Container-side robustness fixes

- `params`/`params_ema` BasicSR checkpoint-wrapper unwrap in
  the build path — universal for the SwinIR/HAT/DRCT family.
- Vendored-arch RGB-mean / computed buffers registered as
  buffers (not plain attributes) so they embed as graph
  constants instead of being mis-classified as missing weights.
- `nbx upscale` subcommand + `forward_pass` final-output
  extraction + R33-pure `roll`/`reflection_pad2d` kernels
  (U3/U4) — consumed Real-ESRGAN with ZERO runtime change
  (R34 model-agnostic design validated).

## Scope decisions (honest)

- **HAT triton / triton-sequential**: blocked by `aten::im2col`
  (OCAB `nn.Unfold`), no Triton kernel. Named follow-up
  **P-TRITON-IM2COL-KERNEL** (Level-1 primitive per the
  Triton-pure 2-level doctrine — closed at the highest level
  reached, remaining work named, not an orphan TODO).
- **U8 DRCT: deliberately skipped** (mandate-optional). No
  clean official DRCT-L on the source hub (Google-Drive-gated;
  only community fine-tunes). Architecturally redundant with
  HAT (deepest transformer-SR → same `requires_fp32_compute` +
  same `aten::im2col` triton block → would add a 4th model at
  2/4 modes with no new capability or engine coverage). The
  mandate's victory criterion (U5+U6+U7 closed + honest
  verdict) was already met with 3 new families + 2 engine
  extensions; remaining budget directed to closure.

## Named follow-up chantiers (tracked, not orphaned)

- **P-TRITON-IM2COL-KERNEL** — Triton-pure `aten::im2col` for
  OCAB-style unfold; unblocks HAT (and any unfold arch) on
  triton + triton-seq.
- **P-CONTAINER-EMBED-ORPHAN-SCALARS** — embed in-forward-loop
  Python-scalar constants as real container constant data
  (currently a 0-dim `torch.empty`; numerically negligible at
  container trace size, not guaranteed at larger tiled sizes —
  relates to BL-1 arbitrary-size tiling).
- **BL-1** (carried from U3/U4) — container `profile.json`
  lacks `upscale`/`window_size`, so arbitrary-input-size
  tiling stays gated; inputs validated at container trace size.

## Hub

All upscaler .nbx target neurobrix.es (proprietary NeuroBrix
hub). NO huggingface.co publication — HF is the raw-model
source only, never a destination.

## Anti-régression baselines (unchanged)

| baseline | result |
|---|---|
| TinyLlama compiled | coherent haiku (KV-cache exercised) |
| Sana 4Kpx 32g compiled | coherent red apple on white plate |
