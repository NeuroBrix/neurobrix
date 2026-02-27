<p align="center">
  <img src="https://raw.githubusercontent.com/NeuroBrix/neurobrix/main/assets/logo.svg" alt="NeuroBrix Logo" width="300"/>
</p>

<h1 align="center">NeuroBrix</h1>

<p align="center">
  <strong>Universal Deep Learning Inference Engine</strong><br/>
  One engine. Any model. Any modality. Zero model-specific code.
</p>

<p align="center">
  <a href="https://pypi.org/project/neurobrix/"><img src="https://img.shields.io/pypi/v/neurobrix?include_prereleases&color=blue" alt="PyPI"/></a>
  <a href="https://pypi.org/project/neurobrix/"><img src="https://img.shields.io/pypi/pyversions/neurobrix?include_prereleases" alt="Python 3.10 | 3.11 | 3.12"/></a>
  <a href="https://github.com/NeuroBrix/neurobrix/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"/></a>
  <a href="https://github.com/NeuroBrix/neurobrix/stargazers"><img src="https://img.shields.io/github/stars/NeuroBrix/neurobrix?style=social" alt="GitHub Stars"/></a>
  <a href="https://neurobrix.es/models"><img src="https://img.shields.io/badge/hub-neurobrix.es-orange" alt="NeuroBrix Hub"/></a>
</p>

<p align="center">
  <a href="https://neurobrix.es/models">Hub</a> &nbsp;|&nbsp;
  <a href="https://neurobrix.es/docs">Docs</a> &nbsp;|&nbsp;
  <a href="https://pypi.org/project/neurobrix/">PyPI</a> &nbsp;|&nbsp;
  <a href="https://github.com/NeuroBrix/neurobrix/blob/main/ROADMAP.md">Roadmap</a> &nbsp;|&nbsp;
  <a href="https://github.com/NeuroBrix/neurobrix/blob/main/CONTRIBUTING.md">Contributing</a>
</p>

---

## The Problem

The AI inference landscape is fragmented. Every model family requires its own stack, its own pipeline code, its own deployment tooling. Want to run a diffusion model? Learn ComfyUI or write custom diffusers pipelines. Need an LLM? Pick between Ollama, vLLM, llama.cpp — each with its own limitations. Multimodal? Start from scratch.

**NeuroBrix eliminates this fragmentation entirely.**

One engine. One CLI. One container format. Import a model, run it. The runtime doesn't know or care whether it's executing a diffusion transformer, a mixture-of-experts LLM, or a multimodal generator. It sees tensors, graphs, and execution plans — nothing else.

---

## Why NeuroBrix?

| Capability | Ollama | llama.cpp | vLLM | ComfyUI | **NeuroBrix** |
|:-----------|:------:|:---------:|:----:|:-------:|:-------------:|
| LLMs | Yes | Yes | Yes | -- | **Yes** |
| Image generation | -- | -- | -- | Yes | **Yes** |
| Multimodal (understand + generate) | -- | -- | -- | -- | **Yes** |
| Mixture-of-Experts | -- | -- | Yes | -- | **Yes** |
| Multi-GPU (pipeline parallel) | -- | -- | Yes | -- | **Yes** |
| Multi-GPU (fine-grained / tensor parallel) | -- | -- | Yes | -- | **Yes** |
| Automatic hardware allocation | -- | -- | -- | -- | **Yes** |
| Single unified runtime | -- | -- | -- | -- | **Yes** |
| Universal model format | -- | GGUF (LLM only) | -- | -- | **NBX (any model)** |
| Framework-independent | -- | Yes | -- | -- | **Yes** |
| No model-specific code | -- | -- | -- | -- | **Yes** |

Other tools solve one piece of the puzzle. NeuroBrix solves the whole puzzle.

---

## Installation

```bash
pip install neurobrix
```

With Triton kernel acceleration:
```bash
pip install neurobrix[cuda]
```

**Requirements:** Python 3.10+ / PyTorch 2.0+ with CUDA / NVIDIA GPU

---

## Quick Start

### 1. Import a Model

```bash
# Browse the hub
neurobrix hub

# Filter by category
neurobrix hub --category IMAGE
neurobrix hub --category LLM

# Import a model
neurobrix import sana/1600m-1024

# Save disk space — delete .nbx archive after extraction
neurobrix import pixart/sigma-xl-1024 --no-keep
```

### 2. Serve (Recommended)

The **serve** workflow is how NeuroBrix is meant to be used. It loads weights into VRAM once and keeps the model warm — every subsequent request executes with zero startup overhead.

```bash
# Start the serving daemon
neurobrix serve --model 1600m-1024 --hardware v100-32g

# For LLMs — interactive chat
neurobrix chat
neurobrix chat --temperature 0.7 --max-tokens 512

# Stop the daemon and free VRAM
neurobrix stop
```

**Why serve mode?**
- Weights loaded **once**, kept warm in GPU memory
- Subsequent requests execute instantly
- Background daemon — your terminal stays free
- Automatic idle timeout (default: 30 min)

**Chat commands** (inside the session):

| Command | Description |
|---------|-------------|
| `/new` | Start a new conversation (clears KV cache) |
| `/context` | Show token usage and cache state |
| `/status` | Engine status and memory |
| `/quit` | Exit |

### 3. Single-Shot Run

For quick one-off experiments:

```bash
# Image generation
neurobrix run --model PixArt-Sigma-XL-2-1024-MS --hardware v100-32g \
    --prompt "a sunset over mountains" --steps 20

# Text generation
neurobrix run --model Qwen3-30B-A3B-Thinking-2507 --hardware c4140-4xv100-custom-nvlink \
    --prompt "Explain quantum computing" --temperature 0.7
```

---

## Available Models

All models are hosted on the [NeuroBrix Hub](https://neurobrix.es/models) and can be imported with a single command.

### Image Generation

| Model | Size | Highlights |
|-------|-----:|------------|
| [**Flex.1-alpha**](https://neurobrix.es/models) | 24.5 GB | 8B parameter rectified flow transformer |
| [**FLUX.2-dev**](https://neurobrix.es/models) | 105.1 GB | 32B parameter text-to-image |
| [**Sana_1600M_4Kpx_BF16**](https://neurobrix.es/models) | 12.1 GB | Efficient 4K image synthesis |
| [**Sana_1600M_1024px_MultiLing**](https://neurobrix.es/models) | 12.1 GB | Multilingual 1024px, runs on laptop GPUs |
| [**PixArt-Sigma-XL-2-1024-MS**](https://neurobrix.es/models) | 20.3 GB | Diffusion Transformer, 4K capable |
| [**PixArt-XL-2-1024-MS**](https://neurobrix.es/models) | 20.4 GB | Diffusion Transformer, 4K capable |
| [**Janus-Pro-7B**](https://neurobrix.es/models) | 13.8 GB | Multimodal understanding + VQ image generation |

### Large Language Models

| Model | Size | Highlights |
|-------|-----:|------------|
| [**deepseek-moe-16b-chat**](https://neurobrix.es/models) | 30.6 GB | 16.4B parameter Mixture-of-Experts |
| [**Qwen3-30B-A3B-Thinking-2507**](https://neurobrix.es/models) | 57.1 GB | 30.5B total / 3.3B active, 262K context, reasoning |

Browse and search the full catalog: **[neurobrix.es/models](https://neurobrix.es/models)**

---

## The NBX Format

NeuroBrix introduces `.nbx` — a **universal container format for AI models**. Where GGUF is limited to LLMs and ONNX struggles with dynamic architectures, NBX captures any computation graph with full fidelity.

```
model.nbx (self-contained archive)
  ├── graph.json         Complete computation graph (TensorDAG)
  ├── topology.json      Execution flow (iterative / autoregressive / forward)
  ├── manifest.json      Component relationships and metadata
  ├── defaults.json      Runtime parameters and shapes
  ├── variables.json     Configurable inputs
  └── weights/           Parameters in safetensors format
```

**What makes NBX different:**

- **Framework-independent** — no dependency on PyTorch, TensorFlow, or any framework at runtime interpretation level
- **Self-describing** — the container carries everything needed to execute: graph, weights, topology, and metadata
- **Modality-agnostic** — the same format works for diffusion models, LLMs, MoE, multimodal, and any future architecture
- **Deterministic** — the execution graph is fully resolved at build time; the runtime follows it mechanically

The runtime never interprets the model. It reads the graph and executes it. The semantic meaning of the computation is entirely encoded in the container.

---

## Prism: Automatic Hardware Allocation

You describe your hardware. NeuroBrix figures out the rest.

The **Prism solver** analyzes the model's memory footprint against your hardware profile and automatically selects the optimal execution strategy. No manual sharding, no configuration, no guesswork.

| Strategy | Description |
|----------|-------------|
| `single_gpu` | Model fits entirely in one GPU |
| `single_gpu_lifecycle` | Components loaded/unloaded sequentially on one GPU |
| `component_placement` | Whole components on different GPUs |
| `pipeline_parallel` | Per-layer sequential fill across GPUs |
| `block_scatter` | Block-level distribution across GPUs |
| `weight_sharding` | Weight-file distribution across GPUs |
| `lazy_sequential` | Stream components through limited VRAM |
| `zero3` | CPU offload with GPU compute |

### Hardware Profiles

NeuroBrix ships with built-in hardware profiles and supports custom definitions:

| Profile | GPUs | VRAM | Interconnect |
|---------|------|-----:|--------------|
| `v100-16g` | 1x V100 | 16 GB | -- |
| `v100-32g` | 1x V100 | 32 GB | -- |
| `v100-32g-2` | 2x V100 | 64 GB | PCIe |
| `v100-32g-x2-nvlink` | 2x V100 | 64 GB | NVLink 300 GB/s |
| `c4140-4xv100-16GB-nvlink` | 4x V100 | 64 GB | NVLink 300 GB/s |
| `c4140-4xv100-custom-nvlink` | 4x V100 | 96 GB | NVLink 300 GB/s |

Add your own profiles as YAML files in `~/.neurobrix/config/hardware/`.

---

## Architecture

### Execution Flow

```
.nbx Container ──> Prism Solver ──> Execution Plan ──> CompiledSequence ──> Output
                   (hardware)       (strategy)         (graph engine)
```

The runtime compiles the entire execution graph at load time into a **CompiledSequence** — a zero-overhead execution path with pre-resolved tensor slots, automatic mixed precision (AMP), direct SDPA calls, and integer-indexed memory arena. No dict lookups per step. No interpretation overhead.

### ZERO Principles

These are not aspirations. They are invariants enforced throughout the codebase.

| Principle | What It Means |
|-----------|---------------|
| **ZERO HARDCODE** | All values are derived from the NBX container. Nothing is hardcoded in the engine. |
| **ZERO FALLBACK** | The system crashes explicitly if data is missing. No silent defaults, no best-effort guesses. |
| **ZERO SEMANTIC** | The runtime has no domain knowledge. It doesn't know what "image", "token", or "latent" means. Only tensors and axes. |

---

## Roadmap

NeuroBrix is in active development. The engine is real, the models run, and the scope is expanding.

**Coming next:**

- **Video generation** — CogVideoX and beyond
- **Audio / TTS / STT** — Whisper, voice synthesis
- **Vision-Language Models** — multimodal understanding at scale
- **Upscalers** — super-resolution models
- **3D generation** — mesh and NeRF models
- **Embeddings** — text and image embedding models
- **Quantization** — INT8/INT4 with NBX-native quantization support
- **Apple Silicon** — Metal/MPS backend
- **NeuroBrix Studio** — desktop GUI for model management and inference

See the full **[Roadmap](https://github.com/NeuroBrix/neurobrix/blob/main/ROADMAP.md)** for details and timelines.

---

## CLI Reference

```bash
# Serving (recommended)
neurobrix serve --model <name> --hardware <profile>
neurobrix chat [--temperature T] [--max-tokens N]
neurobrix stop

# Single-shot
neurobrix run --model <name> --hardware <profile> --prompt <text> [options]

# Model management
neurobrix hub [--category IMAGE|LLM] [--search <query>]
neurobrix import <org/name> [--no-keep] [--force]
neurobrix list [--store]
neurobrix remove <name> [--store|--all]
neurobrix clean [--store|--cache|--all] [-y]

# Inspection
neurobrix info [--models] [--hardware] [--system]
neurobrix inspect <model.nbx> [--topology] [--weights]
neurobrix validate <model.nbx> [--level deep] [--strict]
```

---

## Contributing

NeuroBrix is open source under the Apache 2.0 license. Contributions are welcome.

See **[CONTRIBUTING.md](https://github.com/NeuroBrix/neurobrix/blob/main/CONTRIBUTING.md)** for guidelines.

---

## License

Apache License 2.0 — Copyright 2025 Hocine Benkelaya [Neural Networks Holding LTD](https://neurobrix.es)

See [LICENSE](LICENSE) for the full text.
