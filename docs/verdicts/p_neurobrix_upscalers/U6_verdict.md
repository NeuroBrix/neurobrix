# P-NEUROBRIX-UPSCALERS — U6 verdict (SwinIR)

## Outcome: 8/8 swinir cells ✓ — fp16-unsafe architecture, fp32 opt-in seam added

SwinIR classical-SR x2 / x4 (JingyunLiang/SwinIR release v0.0
DF2K-M, Apache-2.0) operational across all 4 execution modes.
Required one container-packaging fix and one runtime dtype-engine
extension (per-component `requires_fp32_compute` opt-in) — both
strictly additive / opt-in.

## Matrix (input 64×64, test_64.png — downscaled apple)

| model | compiled | sequential | triton | triton-seq |
|---|---|---|---|---|
| swinir-classical-x2 | 128² ✓ | 128² ✓ | 128² ✓ | 128² ✓ |
| swinir-classical-x4 | 256² ✓ | 256² ✓ | 256² ✓ | 256² ✓ |

All output dims = `input × scale`. Visual inspection (R29):
coherent red-apple super-resolution, no garbage / no gray / no
black, on every cell. Sample: `swinir_x4_compiled_sample.png`.
Local artefacts under
`validation_outputs/p_neurobrix_upscalers/matrix_swinir/`.

## R32 cross-mode numerical equivalence

Cosine vs the PyTorch reference (canonical arch + checkpoint,
fp32) = **0.999999** (x2) / **0.999998** (x4) for ALL four
modes. The four modes are mutually bit-equivalent
(mean 157.7, std 93.3 identical across modes). Far above the
≥0.99 R32 gate. R30 dualité confirmed.

## Root cause + fix (factual, methodology-compliant)

**Symptom**: first run produced all-black (compiled/sequential,
NaN→0) and uniform gray ~120 (triton, the +mean tail surviving
alone).

**Diagnosis chain** (>5 iterations, factual instrumentation):
1. Canonical arch + checkpoint correct — fp32 reference
   coherent (uint8 std 93).
2. `NBX_NAN_GUARD_VERBOSE` → first NaN creator =
   `conv_after_body` (`aten.convolution::6`), input range
   ±2285.
3. Reference `conv_after_body` input is [-0.10, 0.24] —
   NeuroBrix fed it ±2285 (~10000× inflated) → root cause is
   upstream of the NaN op.
4. **Decisive repro**: the canonical arch in *true fp16 on GPU*
   (`model.half().cuda()`) also NaNs at `conv_after_body`.
   → SwinIR genuinely overflows fp16; NOT a NeuroBrix bug.
   SwinIR's deep RSTB cascade has no inter-block normalisation;
   activations grow monotonically past the fp16 range. (RRDBNet
   / Real-ESRGAN survives fp16 because its RRDB residual is
   scaled ×0.2 — magnitude bounded. Swin2SR survives because
   its activation range stays in fp16.)
5. Prism `_resolve_dtype` let `profile.preferred_dtype`
   (V100 → fp16) override the model's fp32.
6. Gate test (monkeypatch resolver → fp32): **cos = 1.000000**
   vs the fp32 reference. Fix definition proven before any
   engine code.

**Container-packaging fix**: the SR head's RGB-mean constant was
previously mis-packaged as a missing trained weight, leaving the
leading `aten::sub` with an unresolved input (invalid-input
crash). The container now carries it as a properly embedded
graph constant alongside the other registered buffers. The
BasicSR-family `params` / `params_ema` checkpoint wrapper is now
unwrapped during packaging (universal for SwinIR / HAT / DRCT).

**Runtime fix** (`solver.py`, NeuroBrix `05a39ee`): per-component
`requires_fp32_compute` opt-in read via the existing
`registry_flags` infra (no .nbx field, R18 preserved; env
override `NBX_FORCE_FP32_COMPUTE`). Checked before the fp16
preference in `_resolve_dtype` / `_resolve_component_dtypes`.
Dtype-engine doctrine: dtype is a property of the MODEL as well
as the hardware — this is the data-driven model-side seam.

## Anti-régression (PRESERVED)

The Prism seam is strictly opt-in: default-absent ⇒ empty set ⇒
byte-identical dtype resolution for every existing model. Flag
read verified True only for swinir-classical-x2/x4, False for
TinyLlama / Real-ESRGAN. Empirically confirmed inert:

| cell | result |
|---|---|
| TinyLlama compiled | coherent ocean verse, no error |
| Sana 4Kpx 32g compiled | coherent red apple on white plate (cuda:2) |

## Universality note (for U7/U8)

HAT and DRCT are also deep transformer-SR architectures and are
*expected* to need `requires_fp32_compute`, but this is inferred
from architectural similarity, NOT yet proven. U7 must first run
the `model.half().cuda()` fp16 probe on the canonical HAT arch
before claiming the flag applies.

## Hub

Future publication target: neurobrix.es (proprietary). No
huggingface.co publication.

## Next mandate

U7 — HAT integration (vendor arch, registry, build, 4-mode
validation; fp16 probe first).
