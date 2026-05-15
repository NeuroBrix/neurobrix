# NBX_FORCE_FP32_ACCUM experiment on Sana 4Kpx VAE-isolation (2026-05-09)

## Hypothesis

After:
- Phase 3b "first ULP at add::0" was REFUTED (cross-variant report shows
  add::0 is baseline noise, not shape-specific).
- Per-position chronological bisection identified mm::1 (op_idx=13)
  as first non-noise divergence (134 ULPs at one fingerprint position,
  consistent with K=1024 fp16 reduction amplification of upstream
  ~2.5 ULP noise from permute::0).
- conv::36 microtest (proper depthwise groups=4096) BIT-EXACT vs
  torch.F.conv2d on captured triton-mode silu::12 input.

The remaining "distributed numerical drift" hypothesis predicts that
matmul fp16 accumulator amplification is the dominant noise source.
The wrapper exposes `NBX_FORCE_FP32_ACCUM=1` which upcasts BOTH inputs
of mm/bmm/addmm to fp32 in memory (instead of the default Volta path
which upcasts only the activation, leaving weights fp16 with PROMOTE_B
inline cast).

## Setup

```bash
NBX_FORCE_FP32_ACCUM=1 NBX_DISABLE_AUTOTUNE=1 \
    python3 tools/vae_isolation_probe.py \
    --vae-only-decode --mode triton_sequential
```

Saved latent: `vae_isolation_input.pt` (1, 32, 128, 128) fp32, captured
from sequential 4Kpx decode (oracle).

Output PNG: `vae_isolation_tri_decode.png`
Baseline (without FORCE_FP32): `vae_isolation_tri_decode_baseline_garbage.png`

## Result — REFUTED

PNG still produces **green texture garbage** identical visual class to
baseline. **Forcing matmul fp32 accumulators did NOT fix the bug.**

Sequential oracle (`vae_isolation_seq_decode.png`): coherent red apple.
Triton + FORCE_FP32: green textured noise — same garbage as baseline.

## Conclusion

**Matmul fp16 accumulator chain noise is NOT the cause** of the Sana
4Kpx VAE-isolation green texture. The mm/bmm/addmm path is correct
even at the noisy K=1024 reduction scale (or PROMOTE_B path is bit-exact).

Eliminates:
- "First ULP at add::0 dtype-storage timing" (refuted by cross-variant)
- "Distributed matmul fp16 accumulator chain noise" (refuted by
  FORCE_FP32 experiment, this commit)

## Next investigation candidates

Per CLAUDE.md doctrine and the bisection findings, remaining
hypotheses:

1. **Conv2d (non-depthwise) fp16 accumulator** at K=channels reduction
   (e.g., conv::35 with K=512). Wrappers do NOT have a conv-equivalent
   FORCE_FP32 knob today. Need to either implement one diagnostically
   or read the kernel accumulator dtype.
2. **Runtime integration** — kill_slots, deferred-free, op_uid_interceptor
   side effects (CLAUDE.md "live-set audit" suspect list).
3. **Specific untested kernel bug** in one of the 117 cross-variant
   ops not yet microtested.

The cross-variant report's TOP-shape-specific list within the VAE
includes: `conv::2`, `relu::0`, `bmm::1`, `add::1`, `conv::7`,
`add::11`, `conv::11`, `silu::5`, `add::20`, `mm::12`, `mm::14`,
`cat::6`, plus deeper ops. Conv-class amplification at fp16
accumulator is the most-actionable next test.
