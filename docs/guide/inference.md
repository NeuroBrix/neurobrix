# Single-Shot Inference

Single-shot mode runs one inference request and exits. Useful for scripting and batch processing.

## Basic Usage

```bash
neurobrix run --model <name> --hardware <profile> --prompt "<text>"
```

## Image Generation

```bash
neurobrix run \
  --model PixArt-Sigma-XL-2-1024-MS \
  --hardware v100-32g \
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

## Language Models

```bash
neurobrix run \
  --model deepseek-moe-16b-chat \
  --hardware v100-32g \
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
neurobrix run --model ... --hardware ... --prompt "..." \
  --set global.guidance_scale=7.5 \
  --set global.num_inference_steps=50

# Execution modes
neurobrix run ... --sequential   # Debug mode (native ATen)
neurobrix run ... --triton       # Custom Triton kernels
```

!!! tip
    For repeated inference, use [serve mode](serving.md) instead. Single-shot loads and unloads weights each time (~30-60s overhead per run).
