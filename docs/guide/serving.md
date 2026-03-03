# Serving Models

Serve mode is the **recommended way** to use NeuroBrix. It loads weights into VRAM once and keeps them resident, eliminating the ~30-60 second load time on each inference.

## Start a Server

```bash
neurobrix serve --model <model-name>
```

NeuroBrix auto-detects your hardware. To target a specific profile, add `--hardware <profile>`.

Example:

```bash
neurobrix serve --model PixArt-Sigma-XL-2-1024-MS
```

The daemon starts in the background. Weights are loaded into GPU memory and stay resident.

## Run Inference

Once the server is running:

```bash
# Image generation
neurobrix run --prompt "A sunset over the ocean" --steps 20

# With custom parameters
neurobrix run --prompt "A portrait" --steps 30 --height 1024 --width 1024
```

## Interactive Chat (LLM)

For language models:

```bash
neurobrix serve --model deepseek-moe-16b-chat
neurobrix chat --temperature 0.7
```

Chat commands:

| Command | Action |
|---------|--------|
| `/quit` | Exit chat |
| `/clear` | Clear conversation history |

## Stop the Server

```bash
neurobrix stop
```

This unloads weights and frees GPU memory.

## Override Parameters

Use `--set` to override model defaults at serve time:

```bash
neurobrix serve --model 1600m-1024 \
  --set global.num_inference_steps=30 \
  --set global.guidance_scale=5.0
```

## Targeting Specific Hardware

By default, serve uses your full machine. To use a specific GPU profile:

```bash
# Single GPU only (reserve others for training)
neurobrix serve --model 1600m-1024 --hardware v100-32g

# Multi-GPU profile for large models
neurobrix serve --model deepseek-moe-16b-chat --hardware c4140-4xv100-custom-nvlink
```

See [Hardware Profiles](hardware.md) for available profiles and [Writing Hardware Profiles](hardware-profiles.md) for creating custom ones.

## How It Works

```
neurobrix serve
  → Detect hardware (or load --hardware profile)
  → Load .nbx from ~/.neurobrix/cache/
  → Prism solver allocates components to GPUs
  → Weights loaded to VRAM (eager mode)
  → Unix socket created for IPC
  → Daemon waits for inference requests

neurobrix run --prompt "..."
  → Client connects to daemon socket
  → Sends inference parameters
  → Runtime executes graph (weights already in VRAM)
  → Returns output (saved to disk for images)
```
