# Wan2.1-I2V-14B-480P — 4-mode validation (R29 + drift gate)

## VERDICT 2026-06-27: CFG batch=2 re-validation — OPEN (NOT counted), absorption DISSOLVED

Re-traced + rebuilt from the post-batch-fix forge graphs (fresh 90.66 GB `.nbx`,
2026-06-26). Re-validated at **cfg=5.0 → CFG BATCHED → transformer runs batch=2**
(the absorption-triggering path; cfg engine `mode=BATCHED` when `do_cfg`). Drift
gate at steps=1, seed=42, the same captured input across modes.

**Two distinct findings, kept separate:**

**1. The batch=1-trace-absorption is DISSOLVED (the reopening's actual question).**
All three modes that ran produced the correct batch=2 shapes with no absorption
shape-crash — the batch symbol s0 now runs at 2, born-at-source in the re-trace.

| mode | batch=2 run | velocity std | vs compiled (velocity) | inputs vs compiled |
|------|-------------|--------------|------------------------|--------------------|
| compiled (oracle)      | ✅ 77.7s, pipeline_parallel (transformer→cuda:2, text_encoder→cuda:3) | 1.1776 | — | — |
| sequential (torch mirror) | ✅ 60.3s, pipeline_parallel | 1.1776 | corr **0.999997**, relL2 0.002 | corr 1.000000 |
| triton                 | ✅ 464.7s, pipeline_parallel (multi-GPU works, NOT D1) | 1.2177 | corr 0.9709, relL2 0.24 | corr 1.000000 (shared, pre-CFG) |
| triton_sequential      | ❌ cannot run batch=2 — see DETTE D1 | — | — | — |

compiled + sequential at CFG batch=2 = **PROVEN** (torch-mirror velocity corr
0.999997, all inputs corr 1.0, healthy std 1.1776 = the known-good I2V baseline).

**2. Two residuals keep it OPEN (not 4/4-at-batch=2 → not counted):**

- **triton_sequential — DETTE D1 (multi-device).** `triton/sequential.py` takes a
  single `device_idx` (everything → `cuda:{device_idx}`) and has none of the
  cross-device machinery (`compute_op_devices` / `_run_multi_device` /
  `device_transfer.transfer_tensor`) that `triton/sequence.py` has. The 31.25 GB
  transformer at batch=2 cannot fit one 32 GB GPU → it needs the multi-GPU
  placement triton_seq structurally can't do. This is a **placement** gap, NOT
  absorption. Filed under DETTE D1.
- **triton (compiled) — guided-velocity divergence, deferred.** triton RAN at
  batch=2 on multi-GPU (pipeline_parallel, transformer single-device cuda:2 +
  text_encoder cuda:3 with the inter-component D2D handoff — so it is NOT
  D1-blocked). The **guided** velocity `−4·uncond + 5·cond` is corr 0.9709 /
  relL2 0.24 vs the compiled oracle, vs corr 0.9997 / relL2 2.4% at batch=1
  (cfg=1.0, un-guided). The guidance scale 5× amplifies the per-branch fp16 noise
  (~6× → ~15% predicted; measured 24% sits slightly above the uncorrelated
  ceiling, so a small genuine per-branch component is not excluded). **Code-read
  discriminator (no GPU):** the triton CFG BATCHED path applies the i2v condition
  **symmetrically** to both branches — `i2v_conditioning.apply()` repeats the
  condition to the state batch (`condition.repeat(state.shape[0]//cond.shape[0], …)`,
  cfg/engine.py:294), batches the CLIP image embed `[v, v]` (cfg/engine.py:310-334),
  and the 46b8932 extent-guard is a no-op for Wan. So this is **NOT** the
  CogVideoX-I2V asymmetric per-branch bug; it is consistent with guidance-amplified
  fp16, **cause not isolated**. Deferred (the triton multi-GPU axis is DETTE-class).

Caveat (per the drift gate's own design): the `inputs corr 1.0` row is the
shared **pre-CFG** state + encoder outputs (batch=1-shape dump) — it confirms the
encoders/condition build match, not the per-branch state inside the batch=2
forward. The code-read above covers the per-branch question instead.

**Net:** Wan-I2V-14B is a 14B multi-GPU model; its full 4-mode CFG-batch=2 closure
is gated on DETTE D1 (triton_seq multi-device) + the triton guided-velocity
question. Stays **OPEN / not counted** until D1 is addressed. The absorption fix —
the substantive reason it was reopened — is **confirmed dissolved**.

---

**Historical (REOPENED 2026-06-16):** the 4/4 below was proven at cfg=1.0 /
batch=1 (drift gate steps=1, cfg=1.0) — the batch symbol s0 never ran != 1. The
batch=2 re-validation above supersedes the "expected to re-green" note.

---
**Date:** 2026-06-16
**VERDICT (at batch=1 / cfg=1.0 — degenerate path): 4/4 PROVEN CORRECT.**
- compiled: ✅ COHERENT (R29 fox, multi-GPU).
- sequential: ✅ COHERENT (R29 fox, velocity byte-identical to compiled).
- triton-compiled: ✅ PROVEN by cross-engine drift gate (velocity vs compiled
  oracle corr=0.999718, rel_L2=2.4%, inputs corr=1.0). Coherent frame DEFERRED
  (V100 throughput ~70 min/run — redundant, mirrors coherent compiled).
- triton_sequential: ✅ PROVEN by drift gate (corr=0.999724, rel_L2=2.36%,
  inputs corr=1.0). Coherent frame DEFERRED (V100 throughput).
The two triton coherent frames are deferred to an assumed long run or A100/H100;
drift is the correctness proof (not "runs", not the slow frame). Policy applies
to the whole 14B+ class.

---
(historical detail below)

**Date:** 2026-06-15
**Verdict:**
- compiled (multi-GPU, f9): **PASS** — coherent fox (compiled_FIXED_frame4.png).
- sequential / pytorch op-by-op (multi-GPU, f9): **PASS** — coherent fox
  (sequential_FIXED_frame4.png); step-0 velocity byte-identical to compiled (std 1.1718).
- triton / triton_sequential: **OPEN** — 3 named runtime gaps (see "Triton path" below),
  a separate chantier. The multi-GPU NBX infrastructure is present; the blockers are
  upstream (triton text_encoder output extraction) and triton_sequential multi-device.

Both PyTorch modes (the oracle pair) are coherent and the degenerate-output root cause is
fixed; this is the substantive closure of the compiled bring-up.

## Triton-compiled: PROVEN CORRECT by cross-engine drift gate (2026-06-16)
The correctness proof of a mode is the velocity-diff vs the compiled oracle on an
IDENTICAL captured input — NOT the (V100-slow) coherent frame, and NOT "runs
without crash". Drift gate (steps=1, cfg=1.0, seed=42, pipeline_parallel):
- INPUTS to the transformer match the compiled oracle bit-for-bit: hidden_states
  corr=1.000000 max|Δ|=0.0000; text corr=1.000000 max|Δ|=0.0007; CLIP-image
  corr=1.000000 max|Δ|=0.0087. The triton encoders + i2v conditioning are exact.
- VELOCITY (transformer output) triton-compiled vs compiled oracle:
  **corr=0.999718, rel_L2=2.4%, stdΔ=0.0278** — expected fp16 accumulation over
  40 blocks (only legitimate mode diff = DtypeEngine); tighter than the
  compiled-vs-vendor 0.988. => triton-compiled Wan-I2V-14B is CORRECT.
- Coherent R29 frame: **DEFERRED** — V100 sm_70 throughput (~70 min/run, triton
  static ~10-12% cuBLAS, structural). Render on an assumed long run or A100/H100.
  Redundant for correctness: the triton pipeline mirrors the coherent compiled
  run and the latents/velocity match (proven above).

## Triton path (partial — gap 1 FIXED, gaps 2-4 = a focused triton chantier)
1. **triton text_encoder weight binding — FIXED (commit 9ff8ec6).** Same root cause as the
   compiled (94ff5f8) and sequential (a96dced) weight fixes: `TritonSequence.bind_weights`
   matched by exact name only, so `encoder.token_embed.weight` (graph) never found
   `token_embed.weight` (.nbx) -> embed unbound -> whole encoder propagated empty -> triton
   gather returned [] (diag NBX_DIAG_TRITON_PRELOOP=1). Added the trailing-suffix fallback
   (3rd R30 mirror). Validated: all 3 pre_loop components now produce output.
2. **triton i2v conditioning — OPEN.** The triton flow (triton/flow/iterative_process.py)
   does NOT apply the i2v latent conditioning at all — only the compiled flow + CFG do
   (the conditioning brick was written core/torch-only). The triton transformer's first op
   `aten.convolution::0` (patch_embedding) fails with "Pointer argument cannot be accessed
   from Triton (cpu tensor?)". Needs the i2v conditioning ported to the triton flow as
   NBXTensor (R33-pure), mirroring core/runtime/resolution/i2v_conditioning.py.
3. **triton multi-GPU at scale — pending.** At steps=1/batch=1 the 31.5GB transformer fit
   cuda:2 alone (single_gpu). CFG batch=2 / more frames will need the shard_map multi-device
   path (triton/sequence.py compute_op_devices + _run_multi_device + per-device arenas + D2D
   device_transfer.memcpy(kind=3) — all present per code audit, untested for this model).
4. **triton_sequential multi-device — named.** triton/sequential.py takes a single
   `device_idx` with no cross-device transfer; port the multi-device path from sequence.py.

## Inputs
- prompt: "a red fox walking in snow"
- input image: validation_outputs/video/fox_i2v_input_720x480.png (a red fox in snow)
- height 480, width 832, num-frames 9 (= trace T), steps 20, seed 42, CFG (guidance default)
- placement: weight_sharding across cuda:2 + cuda:3 (transformer 31.25 GB fp16 > one 32 GB GPU)

## Output
- compiled_FIXED_f9.mp4 (frames 0/4/8 in compiled_FIXED_frame4.png)
- All frames: coherent red fox walking in snow, natural temporal motion (stride changes
  f0->f4->f8), matches the input first frame. NOT the previous checkerboard.

## Root cause fixed (degenerate checkerboard -> coherent)
The Wan I2V DiT cross-attends to the text embeds UNMASKED over max_sequence_length (512).
diffusers `_get_t5_prompt_embeds` pads UMT5 text to 512 with zeros; the COUNT of the 504
trailing zero-pad tokens is part of the trained conditioning. NeuroBrix fed only 226 tokens.
Bit-equal microtest (vendor WanTransformer3DModel on NeuroBrix's exact captured inputs):
text=226 -> velocity std 0.46 (corr 0.988 with NeuroBrix: forward + multi-GPU CORRECT);
text=512 zero-padded -> velocity std 1.19 (healthy, T2V coherent baseline was 1.45).

## Fix (both pure-runtime, no rebuild)
1. forge/config/model_registry.yml: Wan2.1-I2V-14B text_encoder `zero_pad_embeddings: true`.
2. core/components/handlers/text_encoder_handler.py finalize_embeddings: after the zero-pad
   masking, extend the sequence to max_sequence_length with zeros (`_cat` + `.new_zeros`,
   torch + NBXTensor -> R30 compiled & triton). Data-driven, gated by the flag (R23 inert).
Verified: encoder_hidden_states now [1,512,4096] = 8 real + 504 zeros; velocity std 0.56->1.17.

## Also validated this chantier (the path to "runs")
- Multi-GPU compute_op_devices scan-all-weight-slots (compiled_sequence.py) — corr 0.988 vs vendor.
- I2V latent conditioning + normalization (i2v_conditioning.py) — vendor-matched.
- CFG image-batch (cfg/engine.py) — encoder_hidden_states_image batched for CFG.

## Relaunch
neurobrix run --model Wan2.1-I2V-14B-480P-Diffusers --compiled \
  --prompt "a red fox walking in snow" \
  --input-image validation_outputs/video/fox_i2v_input_720x480.png \
  --height 480 --width 832 --num-frames 9 --steps 20 --seed 42 --output <out>.mp4

## Hocine validation: TODO
## Remaining: triton 4-mode (after compiled closure), anti-reg T2V/Sana, then Wan2.2-I2V-A14B.
