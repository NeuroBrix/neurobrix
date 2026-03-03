# NeuroBrix

**Universal Deep Learning Inference Engine**

Execute any AI model — image generation, language models, audio, video — without model-specific code. One engine, any model.

---

## What is NeuroBrix?

NeuroBrix is an inference runtime that loads any deep learning model from a single `.nbx` container and executes it on any supported GPU. No framework code, no model-specific adapters, no integration work.

```bash
pip install neurobrix
```

```bash
# Import a model from the hub
neurobrix import sana/1600m-1024

# Start serving (auto-detects your hardware)
neurobrix serve --model 1600m-1024

# Generate
neurobrix run --prompt "A photograph of a mountain lake at sunset"
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Universal Runtime** | Same engine for diffusion, LLM, audio, video models |
| **NBX Format** | Self-contained model archive with graph, weights, and topology |
| **Prism Solver** | Automatic hardware allocation across single/multi-GPU setups |
| **Serve Mode** | Persistent daemon keeps weights in VRAM for instant inference |
| **ZERO Philosophy** | No hardcoded values, no silent defaults, no domain assumptions |

## Supported Model Families

| Family | Models | Status |
|--------|--------|--------|
| **Image** | PixArt-Sigma, Sana, FLUX, Janus-Pro | Production |
| **LLM** | DeepSeek-MoE, Qwen3 | Production |
| **Audio** | Whisper | In progress |
| **Video** | CogVideoX | In progress |

## Quick Links

- [Getting Started](getting-started/index.md) — Install and run your first model
- [CLI Reference](reference/cli.md) — Full command documentation
- [Architecture](architecture/index.md) — How the runtime works
- [NBX Format](architecture/nbx-format.md) — The universal model container
