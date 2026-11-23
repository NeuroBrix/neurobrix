# Serving Models

Serve mode is the **recommended way** to use NeuroBrix for repeated requests. It loads the model once and, on a warm
plan, keeps its weights on the device between requests, so a request does not pay the load again.

## Start a Server

```bash
neurobrix serve --model <model-name>
```

NeuroBrix uses the default hardware profile of your machine. To target a specific profile, add `--hardware <profile>`.

Example:

```bash
neurobrix serve --model PixArt-alpha/PixArt-Sigma-XL-2-1024-MS
```

The daemon starts in the background. `serve` returns once the model is loaded and says whether the daemon serves
warm (weights stay on the device) or cold (weights reload per request).

## Run Inference

Once the server is running, `--model` can be omitted:

```bash
# Image generation
neurobrix run --prompt "A sunset over the ocean" --steps 20

# With custom parameters
neurobrix run --prompt "A portrait" --steps 30 --height 1024 --width 1024
```

## Interactive Chat (LLM)

For language models:

```bash
neurobrix serve --model deepseek-ai/deepseek-moe-16b-chat
neurobrix chat --temperature 0.7
```

Chat commands:

| Command | Action |
|---------|--------|
| `/new` | Start a new conversation |
| `/context` | Show token usage |
| `/status` | Show engine status |
| `/quit` | Exit chat |

## OpenAI-compatible API

```bash
neurobrix serve --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --http      # http://127.0.0.1:8765/v1
```

Endpoints, parameters and settings: [Serving and chat](../serving.md#openai-compatible-http-api).

## Stop the Server

```bash
neurobrix stop
```

This unloads weights and frees GPU memory.

## Override Parameters

`serve` takes no `--set`: a served model plans with its container's own values. Change a value per request with the
`run` flags the daemon reads (`--steps`, `--cfg`, `--height`, `--width`, `--seed`, ...):

```bash
neurobrix serve --model Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers
neurobrix run --prompt "a lighthouse at dusk" --steps 30 --cfg 5.0
```

## Targeting Specific Hardware

By default, serve uses your full machine. To use a specific GPU profile:

```bash
# Single GPU only (reserve others for training)
neurobrix serve --model Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers --hardware v100-32g

# Multi-GPU profile for large models
neurobrix serve --model deepseek-ai/deepseek-moe-16b-chat --hardware c4140-4xv100-custom-nvlink
```

See [Hardware Profiles](hardware.md) for available profiles and [Writing Hardware Profiles](hardware-profiles.md) for creating custom ones.

## How It Works

```
neurobrix serve
  → Read the hardware profile (the machine's default, or --hardware)
  → Load the model from ~/.neurobrix/cache/
  → Prism plans where each component runs
  → Warm plan: weights loaded to the device and kept there
    Cold plan: weights loaded per request
  → Local socket created (a Unix socket on Linux and macOS, loopback TCP on Windows)
  → Daemon waits for requests

neurobrix run --prompt "..."
  → Client connects to the daemon socket
  → Sends the request parameters
  → Runtime executes the graph on the loaded model
  → Returns the output (text on stdout; images, audio and video saved to disk)
```
