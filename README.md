<p align="center">
  <img src="https://raw.githubusercontent.com/NeuroBrix/neurobrix/main/assets/logo.svg" alt="NeuroBrix Logo" width="280"/>
</p>

<h1 align="center">NeuroBrix</h1>

<p align="center">
  <strong>Universal Deep Learning Inference Engine</strong><br/>
  One engine. Any model. No model-specific code.
</p>

<p align="center">
  <a href="https://pypi.org/project/neurobrix/"><img src="https://img.shields.io/pypi/v/neurobrix?include_prereleases&color=blue" alt="PyPI"/></a>
  <a href="https://pypi.org/project/neurobrix/"><img src="https://img.shields.io/pypi/pyversions/neurobrix?include_prereleases" alt="Python"/></a>
  <a href="https://github.com/NeuroBrix/neurobrix/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"/></a>
  <a href="https://neurobrix.es/models"><img src="https://img.shields.io/badge/models-NeuroBrix%20Hub-orange" alt="Hub"/></a>
</p>

---

## What is NeuroBrix?

NeuroBrix is a **universal inference engine** that runs any deep learning model from a single, unified runtime. Instead of writing model-specific pipelines for each architecture — diffusion, autoregressive LLMs, mixture-of-experts, multimodal — NeuroBrix reads a self-contained `.nbx` container and executes the model graph directly.

**The runtime has zero domain knowledge.** It doesn't know what an "image" or a "token" is. It only sees tensors, axes, and execution graphs. All model behavior is encoded in the `.nbx` container.

### Core Principles

| Principle | Meaning |
|-----------|---------|
| **ZERO HARDCODE** | All values derived from the container. Nothing is hardcoded in the engine. |
| **ZERO FALLBACK** | The system crashes explicitly if data is missing. No silent defaults. |
| **ZERO SEMANTIC** | The runtime has no domain knowledge. Only tensors and execution plans. |

---

## Available Models

All models are hosted on the [NeuroBrix Hub](https://neurobrix.es/models) and can be imported with a single command.

### Image Generation

| Model | Org | Size | Description |
|-------|-----|------|-------------|
| **Flex.1-alpha** | Ostris | 24.5 GB | 8B parameter rectified flow transformer |
| **FLUX.2-dev** | black-forest-labs | 105.1 GB | 32B parameter text-to-image model |
| **Sana_1600M_4Kpx_BF16** | Sana | 12.1 GB | Efficient 4K image synthesis |
| **Sana_1600M_1024px_MultiLing** | Sana | 12.1 GB | Multilingual 1024px, runs on laptop GPU |
| **PixArt-Sigma-XL-2-1024-MS** | PixArt | 20.3 GB | Diffusion Transformer for 4K generation |
| **PixArt-XL-2-1024-MS** | PixArt | 20.4 GB | Diffusion Transformer for 4K generation |
| **Janus-Pro-7B** | deepseek-ai | 13.8 GB | Multimodal understanding + VQ image generation |

### Large Language Models

| Model | Org | Size | Description |
|-------|-----|------|-------------|
| **deepseek-moe-16b-chat** | deepseek-ai | 30.6 GB | 16.4B parameter Mixture-of-Experts |
| **Qwen3-30B-A3B-Thinking-2507** | Qwen | 57.1 GB | 30.5B total / 3.3B active, 262K context |

Browse the full catalog: [neurobrix.es/models](https://neurobrix.es/models)

---

## Installation

```bash
pip install neurobrix
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+ (with CUDA support)
- NVIDIA GPU

---

## Quick Start

### 1. Import a Model

```bash
# Browse the hub
neurobrix hub

# Download a model
neurobrix import sana/1600m-1024

# Save disk space (delete .nbx archive after extraction)
neurobrix import pixart/sigma-xl-1024 --no-keep
```

### 2. Serve the Model (Recommended)

The **serve** mode is the primary way to use NeuroBrix. It loads model weights into VRAM once, keeps them warm, and serves requests instantly with zero startup overhead.

```bash
# Start the serving daemon (loads weights, returns terminal immediately)
neurobrix serve --model 1600m-1024 --hardware v100-32g

# Interactive chat session (for LLMs)
neurobrix chat
neurobrix chat --temperature 0.7 --max-tokens 512

# Stop the daemon and free VRAM
neurobrix stop
```

**Why serve mode?**
- Weights are loaded **once** and stay in GPU memory
- Subsequent requests execute instantly (no reload)
- Automatic idle timeout (default: 30 min)
- Background daemon — your terminal stays free

**Chat commands** (inside the chat session):

| Command | Description |
|---------|-------------|
| `/new` | Start a new conversation (clears KV cache) |
| `/context` | Show token usage and cache state |
| `/status` | Show engine status and memory |
| `/quit` | Exit chat |

### 3. Single-Shot Run (One-Off Execution)

For quick experiments without keeping the model loaded:

```bash
# Image generation
neurobrix run --model PixArt-Sigma-XL-2-1024-MS --hardware v100-32g \
    --prompt "a sunset over mountains" --steps 20

# Text generation
neurobrix run --model deepseek-moe-16b-chat --hardware c4140-4xv100-custom-nvlink \
    --prompt "Explain quantum computing" --temperature 0.7
```

> **Note:** `run` loads and unloads the model for each execution. For repeated use, prefer `serve` + `chat`.

---

## How It Works

```
.nbx Container ──> Prism Solver ──> Execution Plan ──> Runtime Executor ──> Output
                    (hardware)       (strategy)         (graph engine)
```

### The NBX Container

The `.nbx` file is a self-contained archive that holds everything needed to run a model:
- **Execution graph** — the complete computation graph as a TensorDAG
- **Weights** — model parameters in safetensors format
- **Topology** — execution flow (iterative, autoregressive, forward)
- **Metadata** — shapes, dtypes, default values, component relationships

The runtime reads the container and executes it mechanically. It never interprets the content — it just follows the graph.

### Prism: Automatic Hardware Allocation

The Prism solver analyzes the model's memory footprint against your hardware and selects the optimal execution strategy automatically. No manual configuration needed.

| Strategy | When Used |
|----------|-----------|
| `single_gpu` | Model fits entirely in one GPU |
| `single_gpu_lifecycle` | Components loaded/unloaded sequentially on one GPU |
| `pp_nvlink` / `pp_pcie` | Pipeline parallelism across multiple GPUs |
| `fgp_nvlink` / `fgp_pcie` | Fine-grained parallelism (MoE expert distribution) |
| `tp` | Tensor parallelism |
| `lazy_sequential` | Stream components through limited VRAM |
| `zero3` | CPU offload with GPU compute |

### Compiled Execution

NeuroBrix compiles the entire execution graph at load time into a **CompiledSequence** — a zero-overhead execution path with:
- Pre-resolved tensor slots (no dict lookups per step)
- Automatic mixed precision (AMP) following PyTorch rules
- Direct `F.scaled_dot_product_attention` calls
- Integer-indexed memory arena

---

## Hardware Profiles

NeuroBrix uses YAML hardware profiles to describe your GPU setup. The Prism solver reads these to determine the best strategy.

| Profile | GPUs | VRAM | Interconnect |
|---------|------|------|--------------|
| `v100-16g` | 1x V100 16GB | 16 GB | — |
| `v100-32g` | 1x V100 32GB | 32 GB | — |
| `v100-32g-2` | 2x V100 32GB | 64 GB | PCIe |
| `v100-32g-3` | 3x V100 32GB | 96 GB | PCIe |
| `v100-32g-x2-nvlink` | 2x V100 32GB | 64 GB | NVLink 300 GB/s |
| `c4140-4xv100-16GB-nvlink` | 4x V100 16GB | 64 GB | NVLink 300 GB/s |
| `c4140-4xv100-custom-nvlink` | 4x V100 (mixed) | 96 GB | NVLink 300 GB/s |

Custom profiles can be added as YAML files in `~/.neurobrix/config/hardware/`.

---

## CLI Reference

### Core Commands

```bash
# Model serving (recommended workflow)
neurobrix serve --model <name> --hardware <profile>   # Start daemon
neurobrix chat [--temperature T] [--max-tokens N]     # Interactive chat
neurobrix stop                                         # Stop daemon

# Single-shot execution
neurobrix run --model <name> --hardware <profile> --prompt <text> [options]

# Model management
neurobrix hub [--category IMAGE|LLM] [--search <query>]   # Browse registry
neurobrix import <org/name> [--no-keep] [--force]          # Download model
neurobrix list [--store]                                    # List installed
neurobrix remove <name> [--store|--all]                    # Remove model
neurobrix clean [--store|--cache|--all] [-y]               # Wipe all

# Inspection
neurobrix info [--models] [--hardware] [--system]          # System info
neurobrix inspect <path.nbx> [--topology] [--weights]      # Inspect container
neurobrix validate <path.nbx> [--level deep] [--strict]    # Validate integrity
```

### Run Options

| Flag | Description |
|------|-------------|
| `--model` | Model name (required) |
| `--hardware` | Hardware profile ID (required) |
| `--prompt` | Input text (required) |
| `--steps` | Inference steps (diffusion) |
| `--cfg` | Guidance scale |
| `--height` / `--width` | Output resolution |
| `--output` | Output file path |
| `--seed` | Random seed |
| `--temperature` | Sampling temperature (LLM) |
| `--repetition-penalty` | Repetition penalty (LLM) |
| `--chat` / `--no-chat` | Force chat template on/off |
| `--set KEY=VALUE` | Override any runtime variable |
| `--seq_aten` | Use native ATen dispatch (debug) |
| `--triton` | Use Triton kernels (experimental) |

---

## The NBX Format

NeuroBrix introduces the `.nbx` container format as a **universal standard for AI model packaging**. Rather than each framework defining its own way to store and load models, `.nbx` captures the complete execution blueprint — graph, weights, topology, and metadata — in a single portable archive.

The models available on the [NeuroBrix Hub](https://neurobrix.es/models) are built using our proprietary tracing technology. We capture the computation graph at a low level, preserving the exact execution semantics of the original model while making it framework-independent. This process produces a self-describing container that any compatible runtime can execute without needing access to the original model code.

Our goal is to establish a structured, vendor-neutral format that enables better organization and interoperability across the AI ecosystem — separating the model definition from the execution environment.

---

## Storage Architecture

```
~/.neurobrix/
  store/              Downloaded .nbx archives (transport)
    1600m-1024.nbx
  cache/              Extracted models (runtime reads from here)
    1600m-1024/
      manifest.json
      topology.json
      defaults.json
      runtime/
      components/
      weights/
```

The `.nbx` file is only used for transport. The runtime always reads from the extracted cache.

---

## Project Structure

```
neurobrix/
  cli/                  Command-line interface
    commands/           run, serve, chat, hub, import, list, ...
  core/                 Runtime engine
    runtime/            Executor, graph engine, CompiledSequence
    prism/              Hardware solver and execution planning
    flow/               Execution flows (iterative, autoregressive, forward)
    dtype/              Automatic mixed precision engine
    cfg/                Classifier-free guidance engine
    module/             Tokenizer, scheduler, KV cache, text processor
    io/                 Weight loading
  kernels/              Triton GPU kernels
  nbx/                  .nbx container format
  serving/              Daemon server, client, session management
  config/               Hardware profiles, family configs
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| [PyTorch](https://pytorch.org/) >= 2.0 | Tensor computation and GPU execution |
| [safetensors](https://github.com/huggingface/safetensors) >= 0.4 | Fast model weight loading |
| [NumPy](https://numpy.org/) >= 1.24 | Numerical operations |
| [PyYAML](https://pyyaml.org/) >= 6.0 | Configuration parsing |
| [requests](https://requests.readthedocs.io/) >= 2.28 | Registry HTTP client |
| [tqdm](https://tqdm.github.io/) >= 4.65 | Progress bars |
| [Triton](https://triton-lang.org/) >= 2.1 | Custom GPU kernels (optional: `pip install neurobrix[cuda]`) |

---

## License

Apache License 2.0 — Copyright 2025 [Neural Networks Holding LTD](https://neurobrix.es)

See [LICENSE](LICENSE) for the full text.
