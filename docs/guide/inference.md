# Single-Shot Inference

Single-shot mode runs one inference request and exits. Useful for scripting and batch processing. The same command serves all 9 model families — image, video, LLM, VLM, multimodal, TTS, STT, audio_llm, and upscalers.

## Basic Usage

```bash
neurobrix run --model <name> --prompt "<text>"
```

NeuroBrix uses the default hardware profile of your machine. To target a specific profile, add `--hardware <profile>`.

## Image Generation

```bash
neurobrix run \
  --model PixArt-alpha/PixArt-Sigma-XL-2-1024-MS \
  --prompt "A cyberpunk city at night" \
  --steps 20
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | required | Text prompt |
| `--steps` | model default | Number of inference steps |
| `--cfg` | model default | Guidance scale |
| `--height` | model default | Output height in pixels |
| `--width` | model default | Output width in pixels |
| `--seed` | random | Seeds sampling and random initialisation |
| `--output` | `output_<model>.png` | Output file path; a missing extension is added, a wrong one is refused by name |

## Video Generation

Text-to-video by default; passing `--input-image` switches to image-to-video, and `--video` to video-to-video.

```bash
# Text-to-video
neurobrix run \
  --model Efficient-Large-Model/SANA-Video_2B_720p_diffusers \
  --prompt "ocean waves at sunset" \
  --num-frames 81 --seed 42 \
  --output waves.mp4

# Image-to-video
neurobrix run \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --input-image first_frame.png \
  --prompt "camera pans left" \
  --num-frames 49 --output clip.mp4
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | required | Text prompt |
| `--input-image` | — | First frame (switches to image-to-video) |
| `--video` | — | Input video (switches to video-to-video) |
| `--num-frames` | the container's value | Number of frames to generate |
| `--fps` | the container's value | Output frame rate |
| `--steps` | the container's value | Number of diffusion steps |
| `--cfg` | the container's value | Guidance scale |
| `--seed` | random | Same seed reproduces the same video |
| `--output` | `output_<model>.mp4` | Output `.mp4` path |

A video model's frame count, steps, guidance scale and frame rate come from its container. When neither the request
nor the container gives one of them, the run is refused by name.

Video models run at their native resolution and clip length; large spatio-temporal activations are tiled automatically, and multi-GPU machines are used automatically for the largest models.

## Language Models

```bash
neurobrix run \
  --model deepseek-ai/deepseek-moe-16b-chat \
  --prompt "Explain quantum computing in simple terms"
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | required | Input text |
| `--max-tokens` | model default | Maximum output tokens |
| `--temperature` | model default | Sampling temperature (0 = greedy) |
| `--top-k` / `--top-p` | model default | Top-k and nucleus sampling |
| `--repetition-penalty` | model default | Repetition penalty (1.0 = none) |
| `--system` | — | System prompt for chat-style models |
| `--chat` / `--no-chat` | — | Force the chat template, or raw text completion |
| `--turns FILE` | — | A JSON list of user messages played as one chat session |

## Advanced Options

```bash
# Override any model default
neurobrix run --model ... --prompt "..." \
  --set global.guidance_scale=7.5 \
  --set global.num_inference_steps=50

# Target specific hardware (optional)
neurobrix run --model ... --hardware v100-32g --prompt "..."

# Print the placement plan and exit, without loading anything
neurobrix run --model ... --explain-plan

# Execution modes (default: --compiled where PyTorch is installed, --triton where it is not)
neurobrix run ... --compiled            # PyTorch fused graph
neurobrix run ... --sequential          # PyTorch op-by-op (debug)
neurobrix run ... --triton              # NeuroBrix Triton kernels (nbxtensor), no PyTorch needed
neurobrix run ... --triton-sequential   # Triton kernels op-by-op (debug)
```

!!! tip
    For repeated inference, use [serve mode](serving.md) instead. Single-shot loads and unloads weights each time.
