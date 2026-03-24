<p align="center">
  <img src="https://gitlab.com/neurobrix/neurobrix/-/raw/main/assets/logo.svg" alt="NeuroBrix Logo" width="120"/>
</p>

<h1 align="center">NeuroBrix</h1>

<p align="center">
  <strong>Universal Deep Learning Inference Engine</strong><br/>
  One engine. Any model. Any modality. Zero model-specific code.
</p>

<p align="center">
  <a href="https://pypi.org/project/neurobrix/"><img src="https://img.shields.io/pypi/v/neurobrix?include_prereleases&color=blue" alt="PyPI"/></a>
  <a href="https://pypi.org/project/neurobrix/"><img src="https://img.shields.io/pypi/pyversions/neurobrix?include_prereleases" alt="Python 3.10 | 3.11 | 3.12"/></a>
  <a href="https://gitlab.com/neurobrix/neurobrix/-/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"/></a>
  <a href="https://gitlab.com/neurobrix/neurobrix"><img src="https://img.shields.io/badge/GitLab-neurobrix-FC6D26?logo=gitlab&logoColor=white" alt="GitLab"/></a>
  <a href="https://neurobrix.es/models"><img src="https://img.shields.io/badge/hub-neurobrix.es-orange" alt="NeuroBrix Hub"/></a>
</p>

<p align="center">
  <a href="https://neurobrix.es/models">Hub</a> &nbsp;|&nbsp;
  <a href="https://neurobrix.es/docs">Docs</a> &nbsp;|&nbsp;
  <a href="https://pypi.org/project/neurobrix/">PyPI</a> &nbsp;|&nbsp;
  <a href="https://gitlab.com/neurobrix/neurobrix">GitLab</a> &nbsp;|&nbsp;
  <a href="#roadmap">Roadmap</a> &nbsp;|&nbsp;
  <a href="https://gitlab.com/neurobrix/neurobrix/-/blob/main/CONTRIBUTING.md">Contributing</a>
</p>

---

## The Problem

The AI inference landscape is fragmented. Every model family requires its own stack, its own pipeline code, its own deployment tooling. Want to run a diffusion model? Learn ComfyUI or write custom diffusers pipelines. Need an LLM? Pick between Ollama, vLLM, llama.cpp — each with its own limitations. Audio? Video? Start from scratch.

**NeuroBrix eliminates this fragmentation entirely.**

One engine. One CLI. One container format. Import a model, run it. The runtime doesn't know or care whether it's executing a diffusion transformer, a mixture-of-experts LLM, a speech recognizer, or a video generator. It sees tensors, graphs, and execution plans — nothing else.

---

## Why NeuroBrix?

| Capability | Ollama | llama.cpp | vLLM | ComfyUI | **NeuroBrix** |
|:-----------|:------:|:---------:|:----:|:-------:|:-------------:|
| LLMs | Yes | Yes | Yes | -- | **Yes** |
| Image generation | -- | -- | -- | Yes | **Yes** |
| Video generation | -- | -- | -- | -- | **Yes** |
| Audio (STT + TTS) | -- | -- | -- | -- | **Yes** |
| Multimodal (understand + generate) | -- | -- | -- | -- | **Yes** |
| Mixture-of-Experts | -- | -- | Yes | -- | **Yes** |
| Multi-GPU auto-allocation | -- | -- | Yes | -- | **Yes** |
| Cross-platform (Linux, Windows, macOS) | Yes | Yes | -- | -- | **Yes** |
| Universal model format | -- | GGUF (LLM only) | -- | -- | **NBX (any model)** |
| No model-specific code | -- | -- | -- | -- | **Yes** |

Other tools solve one piece of the puzzle. NeuroBrix solves the whole puzzle.

---

## Installation

### Step 1: Install PyTorch with CUDA

```bash
# For CUDA 12.4 (RTX 30xx, 40xx, A100, H100)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older GPUs like V100)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"  # Should print: True
```

### Step 2: Install NeuroBrix

```bash
pip install neurobrix
```

### Platform Support

| Platform | GPU Support | Notes |
|----------|-------------|-------|
| **Linux** | CUDA, Triton kernels | Full support, recommended for production |
| **Windows** | CUDA | Fully supported. Triton not available on Windows |
| **macOS** | CPU only | MPS/Metal support planned |

**Requirements:** Python 3.10+ / PyTorch 2.1+ with CUDA / NVIDIA GPU

---

## Quick Start

```bash
# Import a model from the hub
neurobrix import Vendor/Model_Name --no-keep

# Generate an image (hardware auto-detected)
neurobrix run --model Model_Name \
    --prompt "A sunset over mountains" --steps 20

# Or serve for instant repeat inference
neurobrix serve --model Model_Name
neurobrix run --prompt "A robot painting on canvas" --output robot.png
neurobrix stop
```

### Serve Mode (Hot Run Mode , Recommended)

Loads weights into VRAM once and keeps the model warm. Every subsequent request runs with zero startup overhead.

```bash
neurobrix serve --model Model_Name

# Image generation (instant — model already loaded)
neurobrix run --prompt "A cat in a hat" --output cat.png

# LLM interactive chat
neurobrix chat --temperature 0.7

# Stop and free VRAM
neurobrix stop
```

---

## NeuroBrix Hub & Model Management

Models are hosted on the **[NeuroBrix Hub](https://neurobrix.es/models)** and managed locally through a two-tier storage system:

- **Store** (`~/.neurobrix/store/`) — downloaded `.nbx` archives (compressed)
- **Cache** (`~/.neurobrix/cache/`) — extracted models ready for inference

### Browse & Import

```bash
# Browse the full hub catalog
neurobrix hub

# Filter by family
neurobrix hub --category IMAGE
neurobrix hub --category LLM
neurobrix hub --category AUDIO
neurobrix hub --category VIDEO

# Search by name
neurobrix hub --search sana

# Import a model (downloads .nbx → extracts to cache)
neurobrix import vendor/model_name

# Import and delete the .nbx archive to save disk space
neurobrix import pixart/sigma-xl-1024 --no-keep

# Force re-import (overwrites existing)
neurobrix import Vendor/Model_Name --force
```

### List & Manage

```bash
# List installed models in cache (ready to run)
neurobrix list

# List downloaded .nbx archives in store
neurobrix list --store

# Show system info: installed models, hardware, disk usage
neurobrix info --models

# Remove a model from cache
neurobrix remove Model_Name

# Remove from both store and cache
neurobrix remove Model_Name --all

# Clean everything — free all disk space
neurobrix clean --all -y
```

### How It Works

```
neurobrix import Vendor/Model_Name --no-keep
  │
  ├─ 1. Download .nbx from neurobrix.es → ~/.neurobrix/store/
  ├─ 2. Extract to ~/.neurobrix/cache/Model_Name/
  ├─ 3. Validate manifest, components, weights
  └─ 4. Delete .nbx from store (--no-keep)

neurobrix run --model Model_Name --prompt "..."
  │
  └─ Reads directly from cache — zero extraction overhead
```

---

## Supported Models

NeuroBrix is a **runtime engine** — it executes models but does **not train or create** them. All models listed below are the work of their respective authors and are subject to their original licenses. **Users must review and accept each model's license before use.**

### Image Generation

| Model | Author | License | Size |
|-------|--------|---------|-----:|
| Sana 1600M 4K | NVIDIA / MIT | Apache 2.0 | 12 GB |
| PixArt-Sigma-XL-2-1024-MS | PixArt | OpenRAIL++ | 20 GB |
| PixArt-XL-2-1024-MS | PixArt | OpenRAIL++ | 20 GB |
| Flex.1-alpha | Ostris | Apache 2.0 | 24 GB |
| Janus-Pro-7B | DeepSeek | MIT | 14 GB |

### Video Generation

| Model | Author | License | Size |
|-------|--------|---------|-----:|
| SANA-Video 2B 720p | NVIDIA / MIT | Apache 2.0 | 17 GB |

### Audio (Speech-to-Text + Text-to-Speech)

| Model | Author | License | Size | Type |
|-------|--------|---------|-----:|------|
| Whisper Large | OpenAI | MIT | 6 GB | STT |
| Whisper Large V3 Turbo | OpenAI | MIT | 3 GB | STT |
| Parakeet TDT 1.1B | NVIDIA | CC-BY-4.0 | 4 GB | STT |
| Canary-Qwen 2.5B | NVIDIA | CC-BY-4.0 | 10 GB | STT |
| Voxtral Mini 3B | Mistral AI | Apache 2.0 | 7 GB | STT |
| Orpheus 3B | Canopy Labs | Apache 2.0 | 7 GB | TTS |
| Kokoro 82M | Hexgrad | Apache 2.0 | 0.3 GB | TTS |
| VibeVoice 1.5B | Will Held | Apache 2.0 | 6 GB | TTS |
| OpenAudio S1 Mini | Fish Audio | CC-BY-NC-SA-4.0 | 2 GB | TTS |
| Chatterbox | Resemble AI | MIT | 1 GB | TTS |

### Large Language Models

| Model | Author | License | Size |
|-------|--------|---------|-----:|
| DeepSeek-MoE-16B | DeepSeek | MIT | 31 GB |
| Qwen3-30B-A3B-Thinking | Alibaba / Qwen | Apache 2.0 | 57 GB |
| TinyLlama 1.1B | TinyLlama | Apache 2.0 | 4 GB |

> **Non-commercial:** OpenAudio S1 Mini uses CC-BY-NC-SA-4.0 — non-commercial use only. Check each model's license on the [NeuroBrix Hub](https://neurobrix.es/models) before commercial deployment.

Browse the full catalog and license details: **[neurobrix.es/models](https://neurobrix.es/models)**

---

## The NBX Format

NeuroBrix introduces `.nbx` — a **universal container format for AI models**. Where GGUF is limited to LLMs and ONNX struggles with dynamic architectures, NBX captures any computation graph with full fidelity.

```
model.nbx (self-contained archive)
  ├── manifest.json              Model metadata and component list
  ├── topology.json              Execution flow and component connections
  ├── runtime/
  │   ├── defaults.json          Generation parameters, model config
  │   └── variables.json         Runtime tensor allocation rules
  ├── components/
  │   ├── text_encoder/          Text conditioning (CLIP, T5, etc.)
  │   │   ├── graph.json         Computation graph (TensorDAG)
  │   │   ├── profile.json       Component config
  │   │   └── weights/           Safetensors shards
  │   ├── transformer/           Core model (DiT, UNet, decoder, etc.)
  │   │   ├── graph.json
  │   │   ├── profile.json
  │   │   └── weights/
  │   ├── vae/                   Image/video decoder (diffusion models)
  │   │   ├── graph.json
  │   │   ├── profile.json
  │   │   └── weights/
  │   └── ...                    Any number of components per model
  └── modules/
      └── tokenizer/             Tokenizer files
```

The component structure adapts to each model: diffusion models have text_encoder + transformer + vae, LLMs have model + lm_head, audio models have encoder + decoder, etc.

**What makes NBX different:**

- **Framework-independent** — no dependency on PyTorch, TensorFlow, or any framework at runtime interpretation level
- **Self-describing** — the container carries everything needed to execute
- **Modality-agnostic** — the same format works for diffusion, LLMs, MoE, audio, video, and any future architecture
- **Deterministic** — the execution graph is fully resolved at build time

---

## Prism: Automatic Hardware Allocation

You describe your hardware. NeuroBrix figures out the rest. Hardware is auto-detected — the `--hardware` flag is optional.

| Strategy | Description |
|----------|-------------|
| `single_gpu` | Model fits entirely in one GPU |
| `single_gpu_lifecycle` | Components loaded/unloaded sequentially |
| `pipeline_parallel` | Per-layer sequential fill across GPUs |
| `block_scatter` | Block-level distribution across GPUs |
| `weight_sharding` | Weight-file distribution across GPUs |
| `lazy_sequential` | Stream components through limited VRAM |
| `zero3` | CPU offload with GPU compute |

**GPU support:** NVIDIA, AMD, Intel, Apple (planned), plus Tenstorrent, Moore Threads, Biren, Iluvatar, Hygon DCU, Cambricon detection.

---

## Architecture

```
.nbx Container ──> Prism Solver ──> Execution Plan ──> CompiledSequence ──> Output
                   (hardware)       (strategy)         (zero-overhead)
```

The runtime compiles the entire execution graph at load time into a **CompiledSequence** — a zero-overhead execution path with pre-resolved tensor slots, automatic mixed precision, direct SDPA calls, and integer-indexed memory arena. No dict lookups per step. No interpretation overhead.

---

## Roadmap

### Done

- [x] **CompiledSequence** — zero-overhead graph execution engine
- [x] **Prism solver** — automatic multi-GPU hardware allocation (7 strategies)
- [x] **Image family** — 6 diffusion models (PixArt, Sana, Flex, Janus)
- [x] **LLM family** — MoE (DeepSeek), dense (TinyLlama, Qwen3)
- [x] **Audio family** — 11 models, 5 flow handlers (STT + TTS)
- [x] **Video family** — SANA-Video 720p (first of 10 planned)
- [x] **Cross-platform** — Linux, Windows, macOS support
- [x] **Hardware auto-detection** — 10 GPU vendors, CPU-only fallback
- [x] **Persistent serving** — warm daemon with chat interface
- [x] **DtypeEngine** — automatic mixed precision (AMP)
- [x] **TilingEngine** — universal spatial tiling for large inputs
- [x] **NBX Hub** — model registry at neurobrix.es

### Next

- [ ] **Video family expansion** — remaining 9 models (Wan2.1, CogVideoX, Allegro, Mochi, Open-Sora)
- [ ] **Vision-Language Models** — multimodal understanding at scale
- [ ] **Quantization** — INT8/INT4 with NBX-native support
- [ ] **Apple Silicon** — Metal/MPS backend
- [ ] **Upscalers** — super-resolution models
- [ ] **3D generation** — mesh and NeRF models
- [ ] **Embeddings** — text and image embedding models
- [ ] **NeuroBrix Studio** — desktop GUI for model management

---

## CLI Reference

```bash
# Serving (recommended) — hardware auto-detected
neurobrix serve --model <name>
neurobrix chat [--temperature T] [--max-tokens N]
neurobrix run --prompt <text> [--output file] [--steps N] [--cfg F] [--seed N]
neurobrix stop

# Single-shot — hardware auto-detected
neurobrix run --model <name> --prompt <text> [options]

# Model management
neurobrix hub [--category IMAGE|LLM|AUDIO|VIDEO]
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

See **[CONTRIBUTING.md](https://gitlab.com/neurobrix/neurobrix/-/blob/main/CONTRIBUTING.md)** for guidelines.

---

## Model Licenses & Responsible Use

**NeuroBrix is an inference engine — it does not create, train, or own any AI model.**

All models listed in this repository are the intellectual property of their respective authors. NeuroBrix converts published model weights into the `.nbx` container format for efficient execution. The original model licenses remain in full effect.

**User responsibilities:**

- **Review the license** of each model before downloading or using it
- **Non-commercial models** (e.g., CC-BY-NC-SA-4.0) may not be used for commercial purposes
- **Gated models** on Hugging Face require explicit license acceptance before access
- **Redistribution** of model weights is governed by each model's license, not by NeuroBrix's license
- **You are solely responsible** for ensuring your use complies with the applicable model license

**NeuroBrix Hub (neurobrix.es):**

The NeuroBrix Hub hosts pre-built `.nbx` packages for convenience. These packages contain model weights in their original precision, repackaged in the NBX container format. All models on the hub are sourced from publicly available releases with permissive or open licenses. If you are a model author and believe your work is hosted in violation of your license terms, please contact us at legal@neurobrix.es for immediate removal.

---

## License

**NeuroBrix Engine** — Apache License 2.0

Copyright 2025-2026 Hocine Benkelaya

NeuroBrix is developed by [**WizWorks OÜ**](https://wizworks.io), a property of [**Neural Networks Holding LTD**](https://neuralnetworkholding.com).

The Apache 2.0 license covers the NeuroBrix engine, CLI, runtime, and NBX format tooling. **It does not cover the model weights** executed by the engine — those are governed by their respective licenses as listed in the [Supported Models](#supported-models) section.

See [LICENSE](LICENSE) for the full text.
