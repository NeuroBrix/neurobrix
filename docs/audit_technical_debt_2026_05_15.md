# Technical-debt audit — 2026-05-15

Purpose: clear the ground before the next thrust —
**completing NeuroBrix model coverage** (audio / STT / TTS /
multimodal / video / VLM). Every item below is read from the
current code, the live hub, and the local artefact tree. No
assumptions.

Companion: `handover_2026_05_15_post_upscalers_v1.md` (state).
Packaging-side specifics live in the private companion only.

---

## A. Model-coverage matrix (registry universe vs done vs hub)

Registry = 46 models / 9 families. "Conv." = `.nbx`
conversion artefacts present locally. "Hub" = published count
on `neurobrix.es` (queried live).

| Family | Registry models | Conv. done | Published on hub | Coverage gap |
|---|---|---|---|---|
| **llm** (3) | TinyLlama, Qwen3-30B-A3B-Thinking, deepseek-moe-16b | all 3 | 4 (those + Qwen3-30B-A3B base) | published; revalidate runtime |
| **image** (5) | Flex.1-alpha, PixArt-Sigma-XL, PixArt-XL, Sana-1024, Sana-4Kpx | all 5 | 5 | published; `--triton` blocked (see C) |
| **multimodal** (1) | Janus-Pro-7B | yes | 1 | published |
| **stt** (3) | parakeet-tdt-1.1b, whisper-large, whisper-large-v3-turbo | all 3 | 1 (v3-turbo) | **publish parakeet + whisper-large** |
| **tts** (5) | Kokoro-82M, VibeVoice-1.5B, chatterbox, openaudio-s1-mini, orpheus-3b | all 5 | 2 (chatterbox, orpheus) | **publish Kokoro, VibeVoice, openaudio** + Kokoro native bug (C) |
| **audio_llm** (3) | Voxtral-Mini-3B, canary-qwen-2.5b, granite-speech-3.3-8b | Voxtral, canary-qwen (granite **not** converted) | 1 (Voxtral) | **convert granite-speech; publish canary-qwen + granite** |
| **video** (11) | Allegro, Allegro-TI2V, CogVideoX-2b, CogVideoX-5b-I2V, Open-Sora-v2, SANA-Video, Wan2.1-T2V/I2V/VACE, Wan2.2-I2V, mochi-1 | **only SANA-Video** | 1 (SANA-Video) | **biggest gap: 10/11 not converted** |
| **vlm** (3) | Qwen3-VL-30B, gemma-4-26B, gemma-4-E4B | only gemma-4-E4B | 0 | **convert Qwen3-VL + gemma-4-26B; publish all 3; family unproven on hub** |
| **upscaler** (12) | swin2SR ×3, real-esrgan ×3, swinir ×2, hat ×2, drct-x4, swinir-realworld-x4 | 10 (`.nbx` built; drct + swinir-realworld pending) | **0** | **publish the 10 built upscalers** (routine packaging op; hub infra exists) |

**Headline gaps for the continuation:**
1. **video** — 10 of 11 models not yet converted (Allegro
   ×2, CogVideoX ×2, Wan ×4, Open-Sora-v2, mochi-1). Largest
   single body of work.
2. **vlm** — family essentially unproven: 2/3 not converted,
   0 on hub. Qwen3-VL-30B + gemma-4-26B.
3. **audio_llm** — granite-speech-3.3-8b not converted.
4. **publish backlog** — many converted models are NOT on the
   hub: all 10 upscalers, parakeet, whisper-large, Kokoro,
   VibeVoice, openaudio, canary-qwen. The hub is live; this is
   a routine `publish` per model, not infra.
5. **runtime revalidation** — `nbx run` claims 9 families but
   per-family CLI→runtime paths have not been freshly smoke-
   tested this cycle, especially non-LLM in `--triton`.

---

## B. Open follow-ups (`docs/follow-ups/`) — status read per file

| File | Status | Blocks | Severity | Est. scope |
|---|---|---|---|---|
| `janus_triton_anticorrelation.md` | **CLOSED** | — | — | — |
| `pixart_triton_arena_inter_run_bug.md` | **ARCHIVED** (superseded post Layer 6) | — | — | — |
| `layer7-prism-dtype-override.md` | **OPEN** | PixArt-Alpha + PixArt-Sigma in `--triton` | med | medium |
| `layer8-computable-buffers-extension.md` | **OPEN** | Sana 4Kpx in `--triton` | med | medium-large |
| `layer9-sana-4kpx-vae-memory.md` | **OPEN** | Sana 4Kpx `--triton` (VAE memory watermark) | high (Sana 4Kpx) | large |
| `kokoro_cudnn_batch_norm_regression.md` | **OPEN** | Kokoro-82M `::native` (`aten::cudnn_batch_norm` undefined tensor) | **high** (a TTS model we must complete) | medium |
| `qwen3_vram_investigation.md` | **OPEN** | Qwen3-30B `v100-16g --triton` ~10.8 GB peak vs target | med | medium |
| `p-triton-im2col-kernel.md` | **OPEN** (opened today) | HAT `--triton`/`--triton-seq` | med | small-medium (1 kernel) |
| `p-container-embed-orphan-scalars.md` | **OPEN** (opened today) | correctness of shifted-window mask at arbitrary tiled size (BL-1 related) | low now / med later | medium |

## B2. Upscaler v1 backlog (`docs/verdicts/p_neurobrix_upscalers/v1_backlog.md`)

- **BL-1** — packaged container `profile.json` lacks `upscale`
  + `window_size` → `tiling_engine.from_component_config()`
  returns None → arbitrary-input-size upscaling gated; inputs
  validated only at the container's native tile size. Root =
  model-packaging (config not propagated). **OPEN**, medium.
- **BL-2** — `manifest.model_type` is `None` for upscaler
  containers. Cosmetic. **OPEN**, low.

## B3. Cross-referenced backlogs (memory / prior verdicts)

| Item | Status | Note |
|---|---|---|
| Sana 4Kpx `addmm::7` `cudaErrorIllegalAddress` | **OPEN**, isolated | pre-existing intrinsic crash in `block.0.cross_attn.out.0`; high for Sana 4Kpx triton |
| P-TRITON-CPU-UPSTREAM-WHEEL-FOLLOWUP | **OPEN / external** | triton-cpu no PyPI wheel; documented + escalated, NOT internally forked (R25); low actionability our side |
| P-OP-LEVEL-CROSS-DEVICE-SPLIT (Gap B) | **OPEN**, large | multi-GPU op-level split for NBXTensor; lower priority than live-watermark per Sana doctrine |
| P-AUTOTUNE-OFFLINE | **DEFERRED post-dev** | Volta sm_70 mm/bmm/addmm vs cuBLAS structural gap; do NOT re-test measured static-config patterns |
| Qwen3-30B-A3B zero3 latent mat2-on-CPU | **OPEN**, latent | only if Prism budget inflates past HW total → zero3 mishandles weight streaming |
| Fused MoE multi-device (3 remapping bugs) | **OPEN** | kernel correct + TOPK_DIVIDE fixed; multi-device remap + Python-dispatch/CPU-sort speed bottleneck |
| Triton compiled UAF (cudaFree during async) | **OPEN**, arch-decision | deferred-free fixes it; needs architecture decision |

---

## C. Debt that directly blocks the continuation (prioritised)

The continuation is "complete audio/STT/TTS/multimodal/video/
VLM". Debt that stands in the way, ranked:

1. **Kokoro `::native` `cudnn_batch_norm` regression** —
   blocks a TTS model we must finish in its DEFAULT mode (not
   even triton). Highest-priority correctness debt for the
   continuation. Scope: medium (native handler).
2. **Video family conversion (10 models)** — not debt per se
   but the largest *new* work; several use custom code paths
   (Allegro-TI2V component-by-component, Open-Sora-v2 custom
   MMDiT). Expect per-model packaging issues; budget for it.
3. **VLM family proof (Qwen3-VL, gemma-4-26B)** — family has
   never been on the hub; runtime path likely needs validation
   beyond gemma-4-E4B.
4. **`dispatch.py:700-701` `index_put`/`index_put_` = identity
   lambdas** (`P-DISPATCH-INDEX-PUT-CORRECTNESS`) — silent
   no-op on the triton path. If any audio/video/multimodal
   graph uses scatter/index_put in `--triton`, output is
   silently wrong (not a crash). Cheap to audit, must do
   before claiming triton coverage for new families.
5. **layer7/8/9 + Sana addmm::7** — block IMAGE `--triton`
   (already published in `--compiled`); relevant only when the
   continuation includes triton parity for image/video.
6. **publish backlog** — purely operational: `publish` the
   converted-but-unpublished models. No code. Do as models
   are validated.

---

## D. Code TODO/FIXME (NeuroBrix-authored, non-vendored)

14 real items (203/217 are in the vendored reference kernel
library, not debt). Only one is correctness-relevant:

- `kernels/dispatch.py:700-701` — `index_put`/`index_put_`
  identity lambdas (see C-4). **Investigate.**
- `kernels/dispatch.py:343` — linspace via fallback (cosmetic).
- `kernels/nbx_tensor.py:1945`, `utils/shape_utils.py:400`,
  `utils/random_utils.py:17,43`, `utils/tensor_wrapper.py:60`
  — cosmetic / obsolete vendored-origin markers.

---

## E. Recommended sequencing for the continuation

1. **Clear blocking correctness debt first** (Hocine's
   "prepare the ground"):
   - `P-DISPATCH-INDEX-PUT-CORRECTNESS` — 0.5–1 day; gates
     trust in triton for ALL new families.
   - Kokoro `::native` regression — unblocks a TTS model in
     its default mode.
2. **Publish the already-converted backlog** (operational, no
   code): 10 upscalers, parakeet, whisper-large, Kokoro (after
   #1), VibeVoice, openaudio, canary-qwen. One smoke
   round-trip first (publish → `nbx hub` → `nbx import`).
3. **Convert + validate the missing models**, by ascending
   risk: audio_llm `granite-speech` → vlm `gemma-4-26B` /
   `Qwen3-VL` → video (Wan family → CogVideoX → Allegro →
   Open-Sora-v2 → mochi). Per model: convert → `nbx run`
   per-family smoke (`--compiled` first, then `--triton`) →
   R29 artefact → publish.
4. **Triton parity** (layer7/8/9, im2col, Sana addmm::7) only
   where the continuation demands `--triton` for a given
   family; otherwise track as named follow-ups.
5. **P-AUTOTUNE-OFFLINE** stays deferred post-dev.

Closure of the continuation = every registry family has ≥1
model running in `nbx run` (compiled minimum) with an R29
artefact and a hub publish, and every open correctness debt
above is either fixed or a named tracked follow-up.

---

## FIN

This audit is the planning input for the next session. The
hub is LIVE (15 models) — coverage work is convert + validate
+ publish, not infrastructure. HF is upstream source only.
