# Single-Shot Inference

Single-shot mode runs one inference request and exits. Useful for scripting and batch processing. The same command serves all 9 model families — image, video, LLM, VLM, multimodal, TTS, STT, audio_llm, and upscalers.

## Basic Usage

```bash
neurobrix run --model <name> --prompt "<text>"
```

NeuroBrix auto-detects your hardware. To target a specific profile, add `--hardware <profile>`.

## Image Generation

```bash
neurobrix run \
  --model PixArt-Sigma-XL-2-1024-MS \
  --prompt "A cyberpunk city at night" \
  --steps 20
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | required | Text prompt |
| `--steps` | model default | Number of inference steps |
| `--height` | model default | Output height in pixels |
| `--width` | model default | Output width in pixels |
| `--seed` | random | Reproducibility seed |
| `--output` | `output.png` | Output file path |

## Video Generation

Text-to-video by default; passing `--input-image` switches to image-to-video automatically.

```bash
# Text-to-video
neurobrix run \
  --model SANA-Video_2B_720p \
  --prompt "ocean waves at sunset" \
  --num-frames 81 --seed 42 \
  --output waves.mp4

# Image-to-video
neurobrix run \
  --model Wan2.2-I2V-A14B \
  --input-image first_frame.png \
  --prompt "camera pans left" \
  --num-frames 49 --output clip.mp4
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | required | Text prompt |
| `--input-image` | — | First frame (switches to image-to-video) |
| `--num-frames` | model default | Number of frames to generate |
| `--fps` | model default | Output frame rate |
| `--steps` | model default | Number of diffusion steps |
| `--cfg` | model default | Guidance scale |
| `--seed` | random | Same seed reproduces the same video |
| `--output` | auto | Output `.mp4` path |

Video models run at their native resolution and clip length; large spatio-temporal activations are tiled automatically, and multi-GPU machines are used automatically for the largest models.

## Language Models

```bash
neurobrix run \
  --model deepseek-moe-16b-chat \
  --prompt "Explain quantum computing in simple terms"
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | required | Input text |
| `--max-tokens` | model default | Maximum output tokens |

## Advanced Options

```bash
# Override any model default
neurobrix run --model ... --prompt "..." \
  --set global.guidance_scale=7.5 \
  --set global.num_inference_steps=50

# Target specific hardware (optional)
neurobrix run --model ... --hardware v100-32g --prompt "..."

# Execution modes (default: --compiled)
neurobrix run ... --sequential          # PyTorch op-by-op (debug)
neurobrix run ... --triton              # NeuroBrix Triton kernels (vendor-agnostic)
neurobrix run ... --triton-sequential   # Triton kernels op-by-op (debug)
```

!!! tip
    For repeated inference, use [serve mode](serving.md) instead. Single-shot loads and unloads weights each time (~30-60s overhead per run).
