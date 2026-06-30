# Open-Sora-v2 — VERDICT: OPEN (transformer executes end-to-end; VAE SDPA-mask product-freeze is the residual)

**Date:** 2026-06-27 (transformer-execution update 2026-06-30) · **Family:** video T2V · **Arch:** MMDiT (FLUX-style packed-latent video, non-diffusers) + HunyuanVAE + T5 + CLIP

This is an **OPEN with the whole transformer now executing**. As of 2026-06-30
the complete MMDiT denoiser (all 4164 ops, double- + single-blocks) runs in
compiled mode; the failure point has moved out of the transformer into the VAE
decoder. Recorded honestly per "closure = verdict, never prose".

## UPDATE 2026-06-30 — MMDiT transformer executes end-to-end

The 2026-06-28 diagnosis of `mul::11` as a **head_dim symbolization collision**
was **WRONG** and is retracted. The true root was a **cluster of frozen
sequence lengths** in the FLUX concat-attention: the model reads `tensor.shape[i]`
into Python arithmetic (`txt_len + img_len`, `seq * heads`, `full - txt`) which
the trace dispatcher never observes symbolically, so the text/image seq lengths
(31 / 60) baked as concrete literals into downstream op shape-args. Five textures,
all recovered born-at-source in the trace (build commit `b36da6d`):
`(A+B)-A → B` algebra · concat-split suffix-slice start · cat re-propagation ·
deep seq-leaf recovery (single trace + sum). After the fix the complete static
invariant — **zero frozen seq leaf in any shape-arg or dim** — is 0, and the run
advances `mul::11 → mul::238 → all single-blocks → VAE`. The engine needs **no
scalar-promotion crutch** (the graph is fully symbolic).

Regression: Wan2.1-T2V trace bit-inert; CogVideoX-2b coherent at 50 steps (only
its own latent frozen-seq dims symbolize, trace-preserving).

**VAE SDPA-mask product freeze — RESOLVED (`7b65216`).** The mid-attention mask
key length froze at the trace `T*H*W` product (2772 = 9·14·22). Fixed by extending
seq-leaf recovery to DISTINCT-symbol products (2- and 3-way) + a slice-end reinject
from the now-symbolic output dim. The run now advances **past the VAE
mid-attention**. Regression: Wan inert, CogVideoX product-inert.

**Residual blocker (the VAE temporal cluster — active).** Past the mid-attention
the run hits `aten.split_with_sizes::0` in `decoder.up.0.up_sample.0`:
`split_sizes=[1, 8]` where `8 = s1−1` (temporal `T−1`, the causal-3D "first-frame +
rest" split) froze; runtime `T=4` ⇒ `[1,8]` overshoots. ~308 temporal slices in the
decoder carry the same `s1`/`s1−1` class. This is a NEW freeze texture
(`symbol − small_const`, split-sizes-summing-to-a-dim) — the causal-3D-VAE temporal
handling, its own multi-freeze cluster (cf. CogVideoX/Mochi/Allegro VAE temporal
work). Next chantier.

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

1. **MMDiT RoPE/pe shape residual — RESOLVED 2026-06-30 (`b36da6d`).** The
   "head_dim symbolized" reading was WRONG; the real root was the frozen-seq
   cluster (FLUX concat-attention reading `shape[i]` into Python arithmetic). See
   the "UPDATE 2026-06-30" section above. The entire transformer now executes.
2. **VAE SDPA-mask product freeze — RESOLVED (`7b65216`).** Seq-leaf recovery
   extended to products + slice-end reinject; run advances past the mid-attention.
   New active blocker = the **VAE temporal cluster** (`split[1,T−1]` +
   ~308 temporal slices in the causal-3D decoder, `symbol − small_const` texture).
3. **CFG-engine `txt` naming** — at cfg>1 the CFG engine can't determine the
   `encoder_hidden_states` variable (MMDiT names it `txt`); needs data-driven
   handling for batch=2 closure.
4. **triton mirror** — port the brick + 5D packing to `triton/` (R30) after
   compiled is coherent.

## Reproduce (current state — transformer passes, fails in VAE at sdpa::0)

```bash
python3 -m neurobrix run --model Open-Sora-v2 --compiled --mode t2v \
  --prompt "a red fox walking in snow" \
  --height 256 --width 256 --num-frames 13 --steps 4 --seed 42 --cfg 1.0 \
  --output /tmp/os_t2v.mp4
```

## Hocine validation: N/A (OPEN — no coherent frame yet)
