# NeuroBrix CLI — Usage by Family

NeuroBrix runtime supports 9 model families with a uniform CLI. The
required and optional flags per family are declared in
`src/neurobrix/config/families/<family>.yml` and validated at runtime —
the CLI itself stays small.

This page lists canonical invocations per family with reference
open-source models. Run `neurobrix run --help` for the full flag list.

## Common flags

| Flag | Purpose |
|------|---------|
| `--model <name>` | Model directory name in `~/.neurobrix/cache/` |
| `--hardware <id>` | Hardware profile (auto-detect if omitted) |
| `--output <path>` | Save destination. Auto-extension if extension omitted; strict mismatch error otherwise |
| `--seed <int>` | Deterministic seed |
| `--triton` / `--triton-sequential` / `--sequential` | Execution backend (default: compiled native) |

## Execution modes — two branches, four modes

Every family runs through two independent compute branches, each in a
sequential (op-by-op) and a compiled (fused hot-loop) variant. No flag =
`--compiled`.

| Flag | Branch | Substrate | Role |
|------|--------|-----------|------|
| `--compiled` *(default)* | PyTorch | `torch` + cuDNN/cuBLAS | production path |
| `--sequential` | PyTorch | `torch` ATen, op-by-op | reference oracle (proves graph/trace) |
| `--triton` | Triton | NeuroBrix `@triton.jit` + `NBXTensor`, no `torch.*`/cuDNN | vendor-agnostic production |
| `--triton-sequential` | Triton | NeuroBrix kernels, op-by-op | kernel oracle |

Use the default to just run a model, `--triton` for a vendor-agnostic path,
and the two `*-sequential` oracles op-by-op to debug a numerical discrepancy.
The flags are also accepted by `neurobrix serve` and `neurobrix upscale`.

## llm — Text Generation

References: TinyLlama, DeepSeek-MoE, Qwen3 (Alibaba), Llama, Mistral.

```bash
# Basic completion
neurobrix run --model TinyLlama-1.1B-Chat-v1.0 \
    --prompt "What is 2+2?" --max-tokens 30

# Chat with system prompt
neurobrix run --model Qwen3-30B-A3B-Thinking-2507 \
    --system "You are a careful math tutor." \
    --prompt "Explain 7×8 to a child" \
    --max-tokens 200 --chat
```

| Flag | Required | Notes |
|------|----------|-------|
| `--prompt` | yes | Input text |
| `--system` | no | System message (chat mode) |
| `--chat` / `--no-chat` | no | Force chat template on/off |
| `--temperature`, `--max-tokens`, `--repetition-penalty` | no | Sampling params |

Output: `.txt` (or `.json` if `--output result.json`).

## vlm — Vision-Language Models

Reference: Qwen3-VL (Alibaba), LLaVA, InternVL.

```bash
neurobrix run --model Qwen3-VL-7B \
    --input-image cat.jpg \
    --prompt "What animal is in this image?" \
    --max-tokens 100
```

| Flag | Required | Notes |
|------|----------|-------|
| `--prompt` | yes | Question / instruction |
| `--input-image` | yes | Image file path |
| `--system`, `--chat` | no | Chat-style framing |

Output: `.txt`.

## multimodal — Unified Models (text OR image output)

Reference: DeepSeek Janus-Pro (`multimodal_understanding` ↔ `generate`).
**`--mode` is mandatory** because the .nbx is traced for one head.

```bash
# Image generation (requires Janus build traced for image AR)
neurobrix run --model Janus-Pro-7B \
    --mode image --prompt "a red cat sitting on a couch"

# Text understanding (requires Janus build traced for text AR)
neurobrix run --model Janus-Pro-7B \
    --mode text --input-image cat.jpg --prompt "describe this image"
```

A `.nbx` traced for one mode rejects the other with a clear error:

```
ERROR: This 'Janus-Pro-7B' build supports only --mode image
       (its trace generation_type is 'autoregressive_image').
```

| Flag | Required | Notes |
|------|----------|-------|
| `--prompt` | yes | Always |
| `--mode {text,image}` | yes | Multimodal-strict — no auto-deduction |
| `--input-image` | only `--mode text` | Image to describe |

Output: `.txt` (text mode) or `.png` (image mode).

## tts — Text-to-Speech

References: Chatterbox, Orpheus, OpenAudio-S1, Kokoro, VibeVoice.

```bash
# Basic
neurobrix run --model chatterbox --prompt "Hello world"

# With voice clone reference
neurobrix run --model chatterbox \
    --reference-audio my_voice.wav \
    --prompt "Hello world" \
    --output greeting.wav
```

| Flag | Required | Notes |
|------|----------|-------|
| `--prompt` | yes | Text to speak |
| `--reference-audio` | no | Voice clone source |
| `--speaker` | no | Speaker preset id (model-specific) |

Output: `.wav` (24 kHz mono unless model overrides).

## stt — Speech-to-Text

References: Whisper (OpenAI), Canary-Qwen (NVIDIA), Parakeet RNN-T.

```bash
neurobrix run --model whisper-large-v3-turbo \
    --audio interview.wav \
    --output transcript.txt
```

| Flag | Required | Notes |
|------|----------|-------|
| `--audio` | yes | Input wav |
| `--prompt` | no | Optional context hint |

Output: `.txt`.

## audio_llm — Audio-Conditioned Language Models

References: Voxtral (Mistral), Qwen2-Audio, Qwen3-Omni (Alibaba) audio mode.

```bash
neurobrix run --model Voxtral-Mini-3B-2507 \
    --audio meeting.wav \
    --prompt "summarize the discussion" \
    --max-tokens 200
```

| Flag | Required | Notes |
|------|----------|-------|
| `--audio` | yes | Input wav |
| `--prompt` | yes | Question / instruction |
| `--system`, `--chat`, `--max-tokens` | no | Same as llm |

Output: `.txt`.

## image — Image Generation

References: PixArt-XL, PixArt-Sigma, Sana (NVlabs), Flux, SDXL.
Supports `t2i` (default), `img2img`, `inpainting` — auto-deduced from
inputs.

```bash
# Text-to-image (default)
neurobrix run --model PixArt-XL-2-1024-MS \
    --prompt "a red apple on a white plate" --steps 20

# Image-to-image (auto: --input-image present)
neurobrix run --model Sana_1600M_1024px_MultiLing \
    --input-image base.png \
    --prompt "make it stylized cyberpunk"

# Inpainting (auto: --input-image + --mask-image present)
neurobrix run --model PixArt-XL-2-1024-MS \
    --input-image scene.png --mask-image mask.png \
    --prompt "replace the sky with aurora"
```

| Flag | Required | Notes |
|------|----------|-------|
| `--prompt` | yes | Text prompt |
| `--input-image` | no | Triggers `img2img` mode |
| `--mask-image` | no | Requires `--input-image`, triggers `inpainting` |
| `--reference-image` | no | Style reference (model-specific) |
| `--steps`, `--cfg`, `--height`, `--width` | no | Diffusion params |

Output: `.png`.

## upscaler — Image Super-Resolution

References: Swin2SR (classical = clean images, realworld = degraded
photos), Real-ESRGAN, BSRGAN.

Dedicated subcommand:

```bash
neurobrix upscale \
    --model swin2SR-realworld-sr-x4-64-bsrgan-psnr \
    --input low_res.png \
    --output upscaled.png \
    --mode compiled
```

| Flag | Required | Notes |
|------|----------|-------|
| `--model` | yes | Upscaler model name |
| `--input` | yes | Source image (PNG/JPEG) |
| `--output` | yes | Destination image path (PNG) |
| `--mode` | no | `compiled` (default) / `sequential` / `triton` / `triton-sequential` |
| `--hardware` | no | Hardware profile id (auto-detected if omitted) |

Output: `.png` at `input × upscale_factor` (the factor is
intrinsic to each model, e.g. x2 / x4). All four execution modes
produce numerically equivalent results.

Model guide:

| Model | Scale | Best for |
|-------|-------|----------|
| swin2SR-classical-sr-x2-64 | 2× | clean source images |
| swin2SR-classical-sr-x4-64 | 4× | clean source images |
| swin2SR-realworld-sr-x4-64-bsrgan-psnr | 4× | degraded / real-world photos |

## video — Video Generation

References: Wan2.x (Alibaba), CogVideoX, HunyuanVideo, SANA-Video.
Supports `t2v` (default), `i2v`, `v2v` — auto-deduced.

```bash
# Text-to-video
neurobrix run --model SANA-Video_2B_720p_diffusers \
    --prompt "ocean waves at sunset" --num-frames 24

# Image-to-video (auto: --input-image present)
neurobrix run --model Wan2-1B-I2V \
    --input-image first_frame.png \
    --prompt "camera pans left" --num-frames 32

# Video-to-video (auto: --video present)
neurobrix run --model CogVideoX-V2V \
    --video source.mp4 --prompt "same scene at night"
```

| Flag | Required | Notes |
|------|----------|-------|
| `--prompt` | yes | Text prompt |
| `--input-image` | no | Triggers `i2v` mode |
| `--video` | no | Triggers `v2v` mode |
| `--num-frames`, `--fps` | no | Length / playback rate |
| `--steps`, `--cfg`, `--height`, `--width` | no | Diffusion params |

Output: `.mp4` (H.264, yuv420p).

## Mode resolution rules

For each invocation:

1. If `--mode <X>` is passed: validate against `modes.supported`, use it.
2. Otherwise auto-deduce from optional flags' `triggers_mode` (e.g.
   `--input-image` triggers `img2img` for image family).
3. Otherwise fall back to `modes.default`.
4. If the family is `multimodal_strict` (currently only `multimodal`)
   and no mode is set, raise `--mode is required`.

## Output extension policy

- If `--output` is omitted: auto-name `output_<model>.<ext>` where
  `<ext>` comes from `output.default_extension_per_mode[mode]`.
- If `--output result.<ext>` is passed with a matching extension: use as-is.
- If the extension does not match the resolved mode: error clearly,
  never silently write wrong content.
- If `--output result` (no extension): append the auto-extension.

## Adding a new family

1. Create `src/neurobrix/config/families/<family>.yml` with the
   uniform schema (`output_processing`, `execution.has_kv_cache`,
   `inputs.required` / `inputs.optional`, `modes.supported` /
   `modes.default` / `modes.multimodal_strict`,
   `output.default_extension_per_mode`).
2. No CLI / runtime code changes required if the writer is one of
   `txt` / `wav` / `png` / `mp4`. For new formats, extend
   `core/runtime/output_dispatch.save_output()`.
