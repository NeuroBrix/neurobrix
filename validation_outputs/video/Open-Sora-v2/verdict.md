# Open-Sora-v2 — VERDICT: OPEN (forge custom-pipeline DONE; runtime FLUX-video flow structurally validated)

**Date:** 2026-06-27 · **Family:** video T2V · **Arch:** MMDiT (FLUX-style packed-latent video, non-diffusers) + HunyuanVAE + T5 + CLIP

This is a **strong-OPEN**, not a closure. The forge side (the directive's gaps
①②③) is **done and committed**; the runtime is a FLUX-video packed-latent flow
bring-up that is **structurally validated through the MMDiT entry** but not yet
coherent. Recorded honestly per "closure = verdict, never prose".

## DONE + committed (both remotes)

**Forge — custom-pipeline build, fully traceable 4/4:**
- `e950d90` custom-pipeline Phase A (`pipeline: custom` → synthetic topology from
  model_index + registry + phase_a_flow; CLIPTextModel **text-tower** load, not
  AutoModel→vision; MMDiT stimulus). Re-trace **GREEN 4/4** (text_encoder 1597,
  text_encoder_2/CLIP 491, transformer/MMDiT 5633, vae 11017 ops).
- `8b51ef7` tokenizer + de-bloat: custom pipelines symlink text_encoder→vendor
  dir, so no tokenizer/ was found and the vendor parents were embedded as +27 GB
  of duplicate weights. Fix: exclude symlink-parent dirs + derive tokenizer/
  tokenizer_2 from the `global.input_ids*` connections (allowlist only). **.nbx
  85.95 GB → 45.6 GB.**
- `a0b39ba` scheduler: registry `scheduler` block (FlowMatchEulerDiscreteScheduler)
  + builder emits modules/scheduler/ when none in snapshot.
- `27b5949` defaults + de-collision: `temporal_compression_ratio` from HunyuanVAE
  `time_compression_ratio` alias (→ latent_frames); `dtype` default for custom
  pipelines (denoiser bf16→fp16 compute dtype); **MMDiT stimulus de-collision**
  (T·H·W = 4·4·4 = 64 collided with in_channels 64 → froze the img_in reshape;
  distinct 3·4·5 = 60 → num_tokens symbolic; addmm::0 validated).

**Runtime — FLUX-video packed-latent flow (compiled), structurally validated:**
- `core/runtime/resolution/flux_video_conditioning.py` (new brick) + flow edits
  in `core/flow/iterative_process.py`: 5D pack/unpack
  (`[B,C,T,H,W]↔[B,T·(H/2)·(W/2),C·4]`, matching the vendor `pack` rearrange),
  state-variable alias (global.img→global.latents), and img_ids (3-axis grid) /
  txt_ids / cond (T2V zeros) synthesis. All **gated on the denoiser declaring an
  `img_ids` input** (FLUX-family only) → inert for every other model.
- Grounding runs walked the cascade: state ✓ → 5D pack ✓ → latent_frames ✓ →
  dtype ✓ → **MMDiT compute reached** (addmm::0 passes after the de-collision).

## REMAINING (named sub-chantier — resume here)

1. **MMDiT RoPE/pe shape residual** — at `aten.mul::11` the positional-embedding
   multiply mismatches (312 vs 64 at dim 3): the T5 seq-alignment × the FLUX
   3-axis `pe` (likely another trace dim to de-collide, or a txt-seq/pe interaction).
2. **Rest of the 40-block MMDiT + VAE decode** — numerical correctness + a
   coherent frame (scheduler shift: the *√num_frames video factor is deferred).
3. **CFG-engine `txt` naming** — at cfg>1 the CFG engine can't determine the
   `encoder_hidden_states` variable (MMDiT names it `txt`); needs data-driven
   handling for batch=2 closure.
4. **triton mirror** — port the brick + 5D packing to `triton/` (R30) after
   compiled is coherent.

## Reproduce (current state — fails at mul::11)

```bash
python3 -m neurobrix run --model Open-Sora-v2 --compiled --mode t2v \
  --prompt "a red fox walking in snow" \
  --height 256 --width 256 --num-frames 13 --steps 4 --seed 42 --cfg 1.0 \
  --output /tmp/os_t2v.mp4
```

## Hocine validation: N/A (OPEN — no coherent frame yet)
