# Wan2.1-T2V-1.3B — R29 verdict (compiled mode)

**Verdict: COMPILED = COHERENT (fp16, shippable default).**

`compiled_coherent.mp4` (= fp16_negfix.mp4): 33 frames, 832x480, a SHARP red fox
in snow — orange coat, white face/chest, snowy foreground, blurred snowy forest
background. Visually verified (not stats-only): matches the prompt and the vendor
diffusers reference (`vendor_fixed_native_frame.png`, a sharp fox) in quality.

- frame std = 0.241 (vendor reference 0.242); inter-frame |Δ| = 3.2 (smooth, not noise)
- denoising latent traces the vendor U-curve: 0.99 -> dip 0.73 -> recover 1.05
  (vendor 1.02). fp16 trajectory == fp32 (DtypeEngine upcasts matmuls), so fp16
  is coherent; no requires_fp32_compute needed.

## Fixes that made it coherent (committed)
1. Symbolic nearest-exact upsample (4 runtime sites + Forge rule) — VAE spatial
   no longer frozen at trace resolution.
2. Collision-free VAE stimulus (latent W=11) — channel(96) vs 8*s3(96) value-match
   collision removed.
3. Text-embedding padding zeroing (`zero_pad_embeddings`) on BOTH the positive and
   the CFG negative/uncond branches — the DiT no longer cross-attends to UMT5's
   non-zero pad embeddings (vendor parity). The neg branch was the residual that
   produced the blurry two-band output.

## Status of the other modes (NOT yet coherent/run)
- sequential: FAILS (modulation `view[1,6,dim]` batch-freeze; the sequential
  dispatcher lacks compiled's symbol promotion for view targets).
- triton / triton-sequential: not yet run (video triton path untested).

Hocine validation: TODO (please eyeball compiled_coherent_frame.png).
Relaunch: see prompt.txt.
