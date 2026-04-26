# Layer 7 — Prism per-component dtype override (PixArt VAE conv overflow)

**Status**: open — blocks PixArt-Alpha and PixArt-Sigma in `--triton`
mode after Layer 6 unblocked their earlier crash points.
**Affected models**: PixArt-XL-2-1024-MS, PixArt-Sigma-XL-2-1024-MS in
`--triton` and `--triton-sequential` on V100 32 GB.
**Out of scope of**: Layer 6 (architectural Prism dtype resolver work,
~200+ lines + plumbing).

## Symptom

PixArt VAE produces `min=nan, max=nan, mean=nan` final output. Image
saved is uniform color (3129 bytes PNG instead of ~1.4 MB).

## Diagnosis (op-by-op trace)

Wrapping every kernel wrapper with a NaN/Inf detector, the first
inf-producing op with clean inputs is:

```
FIRST inf/nan from conv2d_wrapper:
  out_shape=[1, 512, 512, 512] dtype=fp16
  inf=306, nan=0, finite_abs_max=6.55e+04
```

`6.55e+04` is fp16 max (`65504`). The conv math is correct; the math
output exceeds fp16's storage range and the cast `accum_fp32 → output_fp16`
saturates to ±inf. Subsequent ops propagate the inf into NaN.

## Root cause

Two independent facts collide:

1. **PixArt VAE graph dtype**: `graph.json::torch_dtype = "float32"`
   (the build-time graph capture records VAE in fp32, the diffusers
   default).
2. **Prism allocation on V100**: V100 has no native bf16 → Prism's
   default lowers fp32 weights to fp16 at load time
   (`PrismSolver._resolve_dtype()` returns fp16 for any V100 component
   with no explicit override).

The result: PixArt VAE runs in fp16 on V100. The model's intermediate
activations at the 512-channel × 512×512 spatial conv legitimately
exceed fp16 range:

```
worst-case mag estimate = in_max × weight_max × in_channels × kernel_h × kernel_w
                        ≈ 38      × 0.2        × 512         × 3        × 3
                        ≈ 3.6e7   ≫ 65504 (fp16 max)
```

Native PyTorch handles this same conv on V100 because:
- AMP autocast (per `core/dtype/engine.py`) forces fp32 for `mm`, `bmm`,
  `div`, `addmm` on V100.
- Conv operates fp16 inputs with **fp32 accumulator AND fp32 output**
  (cuDNN behavior on V100), then the next op's autocast handles dtype.

In triton, our conv kernel already uses an fp32 accumulator (`accum =
tl.zeros(..., dtype=tl.float32)`), but the **output** dtype matches the
input dtype (fp16), so the cast at `tl.store` saturates.

## Why Layer 6 doesn't fix this

Sub-fix 6.2 (SDPA SMEM) operates on the **kernel internals**. It
guarantees the SDPA math is computed in fp32. But the conv2d **output
buffer** is allocated by `conv2d_wrapper` as `NBXTensor.empty((..., out_c,
...), dtype=x.dtype)` — fp16 when the input arrived fp16. The dtype of
the output buffer is a **wrapper-level** decision, not a kernel-level one.

A candidate sub-fix 6.5 (`out_dtype=tl.float32` on `tl.dot` in the conv
kernel) was tested empirically on the PixArt scenario and produced
**identical results** (15526 inf, finite_abs_max=65504) with and without
the change — Cas Y in the Layer 6 validation matrix. The reason: the
existing `accum += tl.dot(...)` line already forces fp32 accumulation
because `accum` is fp32-allocated; the truncation happens at the fp16
output buffer store. Sub-fix 6.5 was retired from the Layer 6 commit.

## Solution outline (Layer 7)

The cleanest universal fix is **Prism per-component dtype override** that
respects the build-time recorded dtype when overflow risk is detected:

1. **Detect**: in `PrismSolver._resolve_dtype()`, when a component's
   `graph.json::torch_dtype = float32` AND the hardware lacks native bf16
   AND the model family is "image"/"video" (where VAE intermediates can
   exceed fp16 range), pin the component's allocation dtype to fp32
   instead of downcasting to fp16.
2. **Plumb**: ensure `WeightLoader` honors the per-component dtype (it
   already does — see `core/runtime/weight_loader.py`).
3. **Verify**: triton dispatcher receives fp32 inputs → conv2d output
   buffer is fp32 → no overflow at store time.

A more granular alternative is **per-op dtype upgrade** (mirror of native
AMP) inside the triton path, but that's a deeper architectural change
(~500+ lines, equivalent of building DtypeEngine for triton).

The cheap interim is to add a `target_dtype: "float32"` override in
PixArt VAE's `profile.json` and have `PrismSolver` honor it as the
highest-priority dtype source. ~50 lines but couples Prism to per-model
overrides.

## Validation criteria for Layer 7

- PixArt-Alpha + PixArt-Sigma in `--triton` produce coherent images
  (cosine vae.output_0 vs native ≥ 0.95).
- Sana 1024 (3 modes), LLM harness 14/14 — zero regression.
- VRAM use on V100: PixArt VAE in fp32 doubles its weight footprint
  (~280 MB → 560 MB) — acceptable, well under V100 32 GB total.

## Cross-references

- Layer 6 commit (this fix's parent): see
  `docs/architecture/symbolic-shapes-contract.md` for the master contract
  that Layer 6.3 implements.
- Layer 8 (`docs/follow-ups/layer8-computable-buffers-extension.md`) is
  the parallel architectural blocker for Sana 4Kpx.
