# Wan2.2-I2V-A14B — VERDICT: OPEN, FORGE-BLOCKED (dual-denoiser + all runtime infra DONE; sole blocker = forge `vae_encoder` trace gap; D1 REFUTED 2026-07-01)

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

## UPDATE 2026-07-01 — runtime probe (batch=2 triton_sequential): D1 REFUTED; the ONLY blocker is the forge `vae_encoder` gap (both modes)

A batch=2 triton_sequential probe (cfg=5.0, seed 42) settled the two open runtime
questions in ONE run:

- **Placement — D1 REFUTED (item 3 was wrong).** Prism places the two 14B experts
  at the COMPONENT level on separate GPUs — `transformer → cuda:2`,
  `transformer_2 → cuda:0` (strategy `pipeline_parallel`), `vae → cuda:2`,
  `text_encoder → cuda:1`. The experts do NOT intra-split; each fits one card and
  crosses devices only at the component boundary (handled) — exactly like the now-
  closed Wan-I2V-14B. So the triton axes are NOT DETTE D1; that characterization is
  rescinded. (D1 = latent debt for Qwen3-class only, per DETTE.md.)
- **The vae_encoder gap is confirmed on BOTH modes.** triton_sequential fails at the
  identical point as compiled: `RuntimeError: ZERO FALLBACK: Component 'vae_encoder'
  not in executors. Available: [text_encoder, vae, transformer, transformer_2]`. The
  .nbx genuinely LACKS the `vae_encoder` component (verified: Wan-I2V-14B's topology
  has it; Wan2.2's does not — the dual-denoiser static_graph Phase-A dropped the i2v
  encode detection). This is upstream of the transformer, so it blocks BOTH branches
  identically — it is NOT a triton numerical issue.
- **Triton numerical axis (once unblocked) is a KNOWN quantity.** Wan2.2 shares the
  same TritonDtypeEngine as the now-closed Wan-I2V-14B and is also a CFG model, so a
  batch=2 guided-velocity divergence would be the SAME false-alarm shape (CFG
  amplification of clean per-branch fp16 noise). Validate via the per-branch gate
  (NBX_DUMP_CFG_BATCH) — do NOT re-run the whole class-it/read-CFG-engines arc.
  [[feedback_drift_gate_is_correctness_proof]]

**ESCALATION (build-side, separate system):** Wan2.2-I2V is blocked on a forge
trace-completeness gap — the dual-denoiser trace must emit the `vae_encoder` encode
pass (like Wan-I2V-14B's trace does). The runtime is ready (pre_loop lists it,
i2v_conditioning resolves `condition_component=vae_encoder`, dual-denoiser boundary
switch validated); it cannot hand-roll a missing component (R34/forge doctrine).
Runtime path to close, once the re-trace lands: compiled+sequential drift-gate →
triton per-branch gate → R29. All runtime infra is in place.

## REMAINING (named sub-chantier — resume here)

1. **i2v `vae_encoder` component (FORGE, separate system — the sole blocker).**
   Wan2.2-I2V conditions via a 36ch VAE-latent concat (16 noise + 4 mask + 16
   vae_latent); the encode pass needs a `vae_encoder` component (= AutoencoderKLWan
   in encode mode, VAE config IDENTICAL to Wan2.1-I2V). The forge creates it during
   the i2v Phase-A trace, but the dual-denoiser's static_graph Phase-A broke that
   detection → no vae_encoder graph. Fix: forge Phase-A i2v detection for
   dual-denoiser pipelines (so it emits the encode pass), then registry
   `source_module: vae` alias. The runtime pre_loop already lists it.
2. **compiled + sequential close** — run at 13f-class (boundary_ratio=0.9 exercises
   both experts), drift-gate vs the oracle, batch/CFG exercised, coherent frame.
3. **triton / triton_seq** — NOT D1 (refuted, component-level placement). Same
   TritonDtypeEngine as closed Wan-I2V-14B → per-branch drift gate + R29.

## Reproduce (current state — fails at vae_encoder in pre_loop)

```bash
python3 -m neurobrix run --model Wan2.2-I2V-A14B-Diffusers --compiled --mode i2v \
  --input-image validation_outputs/video/fox_i2v_input_720x480.png \
  --prompt "a red fox walking in a snowy forest" \
  --height 480 --width 720 --num-frames 13 --steps 8 --seed 42 --cfg 5.0 \
  --output /tmp/wan22_i2v.mp4
```

## Hocine validation: N/A (OPEN — no coherent frame yet)
