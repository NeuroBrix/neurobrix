# Wan2.2-I2V-A14B — VERDICT: OPEN (dual-denoiser infrastructure DONE; i2v vae_encoder residual)

**Date:** 2026-06-27 · **Family:** video I2V · **Arch:** dual-denoiser (MoE-of-experts boundary switch) — 2× WanTransformer3DModel (14B each, 28B total) + UMT5 + AutoencoderKLWan

This is an **OPEN with a delivered core**, recorded honestly. The directive's main
Wan2.2 ask — the **dual-denoiser boundary-switch infrastructure** — is built and
validated through trace + build. The runtime close is blocked on the orthogonal i2v
`vae_encoder` image-encode component (a forge Phase-A mechanism the dual-denoiser
trace broke), not on the dual-denoiser itself.

## DONE + validated (the dual-denoiser core — directive's main ask)

- **Registry** (`forge/config/model_registry.yml`): dual denoiser (`transformer`
  role=denoiser, `transformer_2` role=denoiser_low_noise), `boundary_ratio: 0.9`
  (from model_index.json), i2v_latent_conditioning on both experts, UMT5
  `zero_pad_embeddings`, dropped the bogus `image_encoder` (Wan2.2-I2V has
  `image_dim=None` — VAE-latent conditioning only, no CLIP, unlike Wan2.1-I2V).
- **Runtime dual-denoiser boundary switch** (`core/flow/iterative_process.py`
  `_setup_dual_denoiser`): one expert per step — high-noise `transformer` for
  t ≥ boundary_timestep (= boundary_ratio × num_train_timesteps), low-noise
  `transformer_2` for t < boundary. Role-preferred ordering, loop-order fallback.
  GATED on boundary_ratio + ≥2 loop denoisers → inert for every single-denoiser model.
- **Builder**: `boundary_ratio` propagated to defaults.json; `_apply_registry_flow_overlay`
  extended to overlay `loop`/`pre_loop`/`post_loop` (Phase-A traced the dual-denoiser
  as a **static_graph** with both transformers chained — the registry `flow:` block
  overrides it with the correct iterative_process loop carrying BOTH experts).
- **Rope cos/sin fix** (`forge/tracer/video_vae_patches.py`): diffusers ≥ 0.35
  changed WanRotaryPosEmbed from a complex `freqs` buffer to real `freqs_cos`/
  `freqs_sin` (use_real=True). The symbolic-temporal patch now branches on the
  buffer layout (reuses the proven SanaVideo cos/sin symbolic grid, split sizes
  inline from attention_head_dim). **Both 14B transformers trace** (4494 ops each).
- **Analyzer second-denoiser detection** (`forge/importer/analyzer.py`): added
  `transformer_2` to COMPONENT_DIRS — previously it fell through to `_detect_modules`
  and was embedded as a generic *module* (raw weights), not a neural component.
  **Validated**: manifest neural = [transformer, transformer_2, vae, text_encoder],
  4/4 graphs, .nbx 64.28 GB (28B bf16).

Build verified: `flow.type: iterative_process`, `loop.components: [transformer,
transformer_2]`, `boundary_ratio: 0.9` in defaults.

## REMAINING (named sub-chantier — resume here)

1. **i2v `vae_encoder` component** — Wan2.2-I2V conditions via a 36ch VAE-latent
   concat (16 noise + 4 mask + 16 vae_latent); the encode pass needs a `vae_encoder`
   component (= AutoencoderKLWan in encode mode, VAE config IDENTICAL to Wan2.1-I2V).
   The forge creates it during the i2v Phase-A trace, but the dual-denoiser's
   static_graph Phase-A broke that detection → no vae_encoder graph. Fix: forge
   Phase-A i2v detection for dual-denoiser pipelines (so it emits the encode pass),
   then registry `source_module: vae` alias. The runtime pre_loop already lists it.
2. **compiled + sequential close** — run at 13f-class (boundary_ratio=0.9 exercises
   both experts), drift-gate vs the oracle, batch/CFG exercised, coherent frame.
3. **triton / triton_seq** → DETTE D1 (28B multi-GPU, same as Wan-I2V-14B).

## Reproduce (current state — fails at vae_encoder in pre_loop)

```bash
python3 -m neurobrix run --model Wan2.2-I2V-A14B-Diffusers --compiled --mode i2v \
  --input-image validation_outputs/video/fox_i2v_input_720x480.png \
  --prompt "a red fox walking in a snowy forest" \
  --height 480 --width 720 --num-frames 13 --steps 8 --seed 42 --cfg 5.0 \
  --output /tmp/wan22_i2v.mp4
```

## Hocine validation: N/A (OPEN — no coherent frame yet)
