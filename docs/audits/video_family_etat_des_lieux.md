# VIDEO family — factual inventory (2026-05-26)

**Scope**: a factual, inspection-only inventory of the video-generation model
checkpoints present on the Dell but never built into a runtime container
(listed as un-opened technical debt in the test harness). This is the base
for the future video mandate — **no fixes here**, only the map. Method:
static inspection of the local HF model checkpoints, the vendor
pipeline/modeling source in `/home/mlops/ml/venv` (diffusers `0.38.0.dev0`),
and cross-reference against the NeuroBrix runtime engines. No GPU, no model
load. Checkpoints are referred to below by model-directory name.

## NeuroBrix-side capability baseline (what already exists)

| Component type | Runtime-proven today? | Evidence |
|---|---|---|
| T5 text encoder | **Yes** | PixArt-Sigma/XL run in the image family. |
| Gemma2 text encoder | **Yes** | Sana 1024 / 4Kpx run in the image family. |
| UMT5 text encoder | Likely (T5 variant) | Same arch family as T5; untested as such. |
| CLIP text/vision encoder | **Yes** | broad image-family use (CLIP vision in upscaler/vlm paths). |
| 2D DiT denoiser | **Yes** | Sana (linear-attn DiT), PixArt (cross-attn DiT). |
| **3D temporal DiT** (temporal attention + 3D patchify) | **No — new** | none exists; the core video unknown. |
| 2D VAE (incl. DC-AE) | **Yes** | Sana DC-AE, image VAEs. |
| **3D causal-temporal VAE** | **No — new** | AutoencoderKLWan/CogVideoX/Mochi/Allegro/LTX2 all absent. |
| Diffusion flow (`iterative_process`) | **Partially prepared** | `core/flow/iterative_process.py:458` accepts `5D [B,C,T,H,W]` at the pre-VAE gate — anticipation only; 5D latent alloc / noise sampling / scheduler step / `variable_resolver` plumbing unverified. |
| Schedulers UniPC / DDIM-v-pred / FlowMatchEuler / DPM++ | **Yes** (image lineage) | scheduler engine `core/module/scheduler/*`; flow-matching used by Sana. |

**Key reuse finding (corrects the opening hypothesis):** the VibeVoice
**next-token-diffusion** flow (`core/flow/next_token_diffusion.py`) is
**not** the reuse vector for these video models. It is audio/speech-specific
(control tokens `speech_start/speech_end/speech_diffusion/eos`, per-token
acoustic-latent diffusion head, acoustic/semantic tokenizers → 24 kHz
waveform). All 11 video models below are **latent diffusion** (a DiT denoiser
iterated over scheduler timesteps on a full 5D latent), not autoregressive
token generation. The reusable runtime is the existing **`iterative_process`
diffusion flow** (already 5D-latent-aware) + the **scheduler engine**, plus a
new **3D VAE** post-loop and a new **3D DiT** per-step component.

## Per-model inventory

"Oracle" = matching diffusers pipeline class importable in 0.38.0.dev0 (for
§5.8 reference runs).

### Wan family — `WanTransformer3DModel` + `AutoencoderKLWan` (z16) + UMT5, UniPC flow-matching
1. **Wan2.1-T2V-1.3B-Diffusers** (27G) — T2V. `WanTransformer3DModel`
   (30L/12H×128, RoPE), patch `[1,2,2]`. Oracle **yes** (`pipeline_wan.py`).
   Smallest Wan transformer (5.3G). **Best first target.**
2. **Wan2.1-I2V-14B-480P-Diffusers** (65G) — I2V. Wan transformer scaled
   (40L/40H, in_channels 36, image_dim 1280) **+ CLIP image_encoder** (1.2G).
   Oracle **yes** (`pipeline_wan_i2v.py`).
3. **Wan2.1-VACE-1.3B-diffusers** (18G) — T2V + control/editing.
   `WanVACETransformer3DModel` = Wan 1.3B + VACE control branch
   (`vace_in_channels 96`). Oracle **yes**.
4. **Wan2.2-I2V-A14B-Diffusers** (118G, heaviest) — I2V. **Dual-expert**:
   `transformer` + `transformer_2` switched by `boundary_ratio 0.9`
   (high/low-noise). Oracle **yes** (0.38 supports `transformer_2`).

### CogVideoX family — `CogVideoXTransformer3DModel` + `AutoencoderKLCogVideoX` + T5, CogVideoXDDIM v-pred
5. **CogVideoX-2b** (13G, smallest total) — T2V. 30L/30H×64, **no RoPE**,
   v-prediction zero-SNR, latent 16ch, temporal_compression 4. Oracle
   **yes**. Architecturally the cleanest video DiT → fallback first target.
6. **CogVideoX-5b-I2V** (21G) — I2V. Same transformer class, 42L/48H, RoPE
   ON, in_channels 32 (16 latent + 16 VAE-encoded image cond, no separate
   image encoder). Oracle **yes** (`pipeline_cogvideox_image2video.py`).

### Allegro family — `AllegroTransformer3DModel` + `AutoencoderKLAllegro` + T5, EulerAncestral
7. **Allegro** (31G) — T2V. DiT 32L/24H×96, ada_norm_single. Oracle **yes**.
8. **Allegro-TI2V** (19G) — Text+Image-to-Video. Oracle **partial** —
   pipeline + VAE present but `AllegroTransformerTI2V3DModel` **missing** in
   diffusers 0.38; a locally vendored copy of the original Allegro repo (its
   `single_inference_ti2v.py`) can serve as the TI2V oracle.

### Singletons
9. **mochi-1-preview** (112G) — T2V flow-matching (`FlowMatchEulerDiscrete`).
   `MochiTransformer3DModel` 48L/24H×128 (swiglu, qk rms_norm), transformer
   **52G** (largest single denoiser), `AutoencoderKLMochi`. Oracle **yes**.
10. **Open-Sora-v2** (72G) — T2V, Flux-derived **MMDiTModel** (19 dual + 38
    single blocks, hidden 3072). Dual text enc (T5-v1.1-xxl + CLIP),
    `AutoencoderKLCausal3D`. Flat top-level safetensors (not component
    subfolders). Oracle **NO** — `OpenSoraV2Pipeline`, `MMDiTModel`,
    `AutoencoderKLCausal3D` all **missing** in diffusers 0.38 (its index
    pins 0.32.1). Needs a vendored pipeline → highest tooling cost.
11. **SANA-Video_2B_720p_diffusers** (16G) — **the partially-worked one,
    registered family `video`**. T2V, DPM++ flow (`flow_shift 8.0`).
    `SanaVideoTransformer3DModel` (20L/20H×112, **SANA linear-attention
    lineage** — related to our working Sana image DiT), patch `[1,1,1]`,
    **Gemma2 text encoder** (already runtime-proven), VAE
    `AutoencoderKLLTX2Video` (latent **128ch**, spatial_compression 32,
    temporal_compression 8). Oracle **yes**. **Carries the standing blocker
    P-PRISM-VIDEO-5D-UNPACK.**

## Shared-architecture grouping (ROI)

| Family | Models | Denoiser | VAE | Text enc | Scheduler |
|---|---|---|---|---|---|
| **Wan** (best ROI: 4 models, 1 VAE, 1 base transformer) | #1,#2,#3,#4 | WanTransformer3DModel (+VACE; 2.2 ×2) | AutoencoderKLWan | UMT5 | UniPC flow |
| CogVideoX | #5,#6 | CogVideoXTransformer3DModel | AutoencoderKLCogVideoX | T5 | CogVideoXDDIM v-pred |
| Allegro | #7,#8 | AllegroTransformer3DModel(/TI2V) | AutoencoderKLAllegro | T5 | EulerAncestral |
| Mochi | #9 | MochiTransformer3DModel | AutoencoderKLMochi | T5 | FlowMatchEuler |
| Open-Sora v2 | #10 | MMDiTModel (Flux) | AutoencoderKLCausal3D | T5-xxl + CLIP | in-pipeline |
| SANA-Video | #11 | SanaVideoTransformer3DModel | AutoencoderKLLTX2Video | Gemma2 | DPM++ flow |

## Capability-gap analysis

1. **Generation flow**: reuse the existing `iterative_process` diffusion flow
   (already 5D-aware) + scheduler engine. **No** next-token-diffusion reuse
   (those video models are not autoregressive). New work is the per-step 3D
   DiT component + post-loop 3D VAE, not a new flow type.
2. **5D symbolic shapes**: partially covered. The symbolic-shapes contract
   (`docs/architecture/symbolic-shapes-contract.md:66`) already specifies
   5D `[B,C,T,H,W]` spatial-adaptive behaviour, and the diffusion flow
   validates 5D latents. **The gap is Prism placement** — see blocker.
3. **DtypeEngine video ops**: `conv3d`/`conv_transpose3d` already in
   `AMP_FP16_OPS` (`core/dtype/engine.py`); a `conv3d` Triton reference
   exists under `kernels/triton_kernels_ref/`. Dtype-wise the 3D conv path is
   anticipated; the Triton-pure 3D-conv **kernel/wrapper** is unbuilt
   (triton-mode concern, not compiled-mode).
4. **Matured symbolic (ceil-pad, audio_token_grid)**: ceil-pad
   (P-CEIL-PAD-WINDOW) may resurface for temporal windowing; audio_token_grid
   is audio-specific. Neither is the dominant video unknown.
5. **3D causal-temporal VAE**: genuinely new (causal temporal convs, temporal
   compression 4–8). No 3D VAE has ever been run. This is the second core
   unknown after the 3D DiT.

## Known blockers

- **P-PRISM-VIDEO-5D-UNPACK** (P1): the Prism placement/allocator hardcodes a
  4-tuple shape unpack; 5D `[B,C,T,H,W]` fails with `too many values to
  unpack (expected 4)`. Site: `src/neurobrix/core/prism/solver.py` (or
  callee). Reproduces on SANA-Video both modes; **not** dtype-related
  (reproduces with `NBX_DISABLE_AUTO_FP32=1`). This blocks **every** video
  model, not just SANA-Video — it is the first thing the video mandate must
  fix. Tracked in `docs/follow-ups/INDEX.md`.

## Suggested priority

1. **Wan2.1-T2V-1.3B** — first target. Smallest Wan transformer (5.3G), full
   importable oracle, T2V-only (no image-encoder complication), and unlocks
   the highest-leverage family (4 Wan models share VAE + base transformer).
2. **CogVideoX-2b** — architecturally-simplest fallback (smallest total 13G,
   no-RoPE DiT, full oracle).
3. **SANA-Video** — natural third: SANA lineage reuses our proven Sana image
   DiT patterns + Gemma2 encoder; but it is the 128-ch LTX2-VAE / patch[1,1,1]
   stress case and carries P-PRISM-VIDEO-5D-UNPACK directly.
4. Defer **Open-Sora-v2** (no oracle, needs vendored pipeline) and
   **Allegro-TI2V** (missing TI2V transformer class) until their oracles are
   vendored. Defer the 14B+ / dual-expert / 112G models (Wan I2V-14B,
   Wan2.2-A14B, Mochi) until the 1.3B path is proven.

**Sequencing note**: P-PRISM-VIDEO-5D-UNPACK must close before any video
model can run end-to-end. The first-target work splits naturally into:
(a) fix Prism 5D placement, (b) bring up Wan2.1-T2V-1.3B text-enc (UMT5) +
3D DiT + 3D VAE through the existing diffusion flow, validating each
component against the `pipeline_wan.py` oracle §5.8.
