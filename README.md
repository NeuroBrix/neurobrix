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
  <a href="#roadmap">Roadmap</a> &nbsp;|&nbsp;
  <a href="https://github.com/NeuroBrix/neurobrix/blob/main/CONTRIBUTING.md">Contributing</a>
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
neurobrix import sana/1600m-1024 --no-keep

# Generate an image (hardware auto-detected)
neurobrix run --model Sana_1600M_1024px_MultiLing \
    --prompt "A sunset over mountains" --steps 20

# Or serve for instant repeat inference
neurobrix serve --model Sana_1600M_1024px_MultiLing
neurobrix run --prompt "A robot painting on canvas" --output robot.png
neurobrix stop
```

### Serve Mode (Recommended)

Loads weights into VRAM once and keeps the model warm. Every subsequent request runs with zero startup overhead.

```bash
neurobrix serve --model Sana_1600M_1024px_MultiLing

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
neurobrix import sana/1600m-1024

# Import and delete the .nbx archive to save disk space
neurobrix import pixart/sigma-xl-1024 --no-keep

# Force re-import (overwrites existing)
neurobrix import sana/1600m-1024 --force
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
neurobrix remove Sana_1600M_1024px_MultiLing

# Remove from both store and cache
neurobrix remove Sana_1600M_1024px_MultiLing --all

# Clean everything — free all disk space
neurobrix clean --all -y
```

### How It Works

```
neurobrix import sana/1600m-1024 --no-keep
  │
  ├─ 1. Download .nbx from neurobrix.es → ~/.neurobrix/store/
  ├─ 2. Extract to ~/.neurobrix/cache/Sana_1600M_1024px_MultiLing/
  ├─ 3. Validate manifest, components, weights
  └─ 4. Delete .nbx from store (--no-keep)

neurobrix run --model Sana_1600M_1024px_MultiLing --prompt "..."
  │
  └─ Reads directly from cache — zero extraction overhead
```

---

## Supported Models

All models are hosted on the [NeuroBrix Hub](https://neurobrix.es/models).

### Image Generation

| Model | Size | Resolution |
|-------|-----:|-----------|
| **Sana_1600M_1024px_MultiLing** | 12 GB | 1024px, multilingual |
| **Sana_1600M_4Kpx_BF16** | 12 GB | 4096px ultra-high resolution |
| **PixArt-Sigma-XL-2-1024-MS** | 20 GB | Diffusion Transformer |
| **PixArt-XL-2-1024-MS** | 20 GB | Diffusion Transformer |
| **Flex.1-alpha** | 24 GB | Rectified flow transformer |
| **Janus-Pro-7B** | 14 GB | Multimodal understand + VQ generation |

### Video Generation

| Model | Size | Resolution |
|-------|-----:|-----------|
| **SANA-Video_2B_720p** | 17 GB | 1280x704, 81 frames, 16fps |

### Audio (Speech-to-Text + Text-to-Speech)

| Model | Size | Type |
|-------|-----:|------|
| **Whisper Large** | 6 GB | STT — multilingual transcription |
| **Whisper Large V3 Turbo** | 3 GB | STT — fast multilingual |
| **Parakeet TDT 1.1B** | 4 GB | STT — NeMo transducer |
| **Canary-Qwen 2.5B** | 10 GB | STT — multilingual, multi-task |
| **Granite Speech 3.3-8B** | 17 GB | STT — audio-conditioned LLM |
| **Voxtral Mini 3B** | 7 GB | STT — multimodal audio LLM |
| **Orpheus 3B** | 7 GB | TTS — expressive speech synthesis |
| **Kokoro 82M** | 0.3 GB | TTS — lightweight, fast |
| **VibeVoice 1.5B** | 6 GB | TTS — diffusion-based acoustic model |
| **OpenAudio S1 Mini** | 2 GB | TTS — DualAR codec generation |
| **Chatterbox** | 1 GB | TTS — conversational speech |

### Large Language Models

| Model | Size | Highlights |
|-------|-----:|-----------|
| **DeepSeek-MoE-16B** | 31 GB | 64-expert Mixture-of-Experts |
| **Qwen3-30B-A3B-Thinking** | 57 GB | 30B total / 3B active, reasoning |
| **TinyLlama 1.1B** | 4 GB | Compact, fast inference |

Browse the full catalog: **[neurobrix.es/models](https://neurobrix.es/models)**

---

## The NBX Format

NeuroBrix introduces `.nbx` — a **universal container format for AI models**. Where GGUF is limited to LLMs and ONNX struggles with dynamic architectures, NBX captures any computation graph with full fidelity.

```
model.nbx (self-contained archive)
  ├── graph.json         Complete computation graph (TensorDAG)
  ├── topology.json      Execution flow and component connections
  ├── manifest.json      Component metadata
  ├── defaults.json      Runtime parameters
  └── weights/           Parameters in safetensors format
```

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

### ZERO Principles

| Principle | What It Means |
|-----------|---------------|
| **ZERO HARDCODE** | All values derived from the NBX container. Nothing hardcoded in the engine. |
| **ZERO FALLBACK** | System crashes explicitly if data is missing. No silent defaults. |
| **ZERO SEMANTIC** | Runtime has no domain knowledge. Only tensors and execution plans. |

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

See **[CONTRIBUTING.md](https://github.com/NeuroBrix/neurobrix/blob/main/CONTRIBUTING.md)** for guidelines.

---

## License

Apache License 2.0 — Copyright 2025-2026 Hocine Benkelaya

NeuroBrix is developed by [**WizWorks OÜ**](https://wizworks.io), a property of [**Neural Networks Holding LTD**](https://neuralnetworkholding.com).

See [LICENSE](LICENSE) for the full text.
