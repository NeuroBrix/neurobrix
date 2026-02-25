# NeuroBrix CLI Reference

> **Version**: 0.1.0 | **Last Update**: February 2026

## Installation

```bash
pip install neurobrix
```

Development mode (no install needed):
```bash
PYTHONPATH=src python -m neurobrix <command> [options]
```

---

## Commands Overview

| Command | Description |
|---------|-------------|
| [`serve`](#serve) | Start persistent model serving daemon |
| [`chat`](#chat) | Interactive chat session with a running daemon |
| [`stop`](#stop) | Stop the serving daemon |
| [`run`](#run) | Run single-shot model inference |
| [`hub`](#hub) | Browse models on the NeuroBrix Hub |
| [`import`](#import) | Download a model from the hub |
| [`list`](#list) | Show installed models |
| [`remove`](#remove) | Remove a model |
| [`clean`](#clean) | Wipe all local models |
| [`info`](#info) | Display system information |
| [`inspect`](#inspect) | Inspect a .nbx container |
| [`validate`](#validate) | Validate .nbx file integrity |

---

## Model Serving (Recommended Workflow)

### serve

Start the serving daemon. Loads model weights into VRAM once and keeps them warm for instant inference.

```bash
neurobrix serve --model <name> --hardware <profile> [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model name (required) | — |
| `--hardware` | Hardware profile ID (required) | — |
| `--timeout` | Idle timeout in seconds | 1800 |
| `--foreground` | Run in foreground (for debugging) | false |
| `--seq_aten` | Use native ATen dispatch | false |
| `--triton` | Use Triton kernels | false |

Example:
```bash
neurobrix serve --model deepseek-moe-16b-chat --hardware c4140-4xv100-custom-nvlink
```

### chat

Connect to a running daemon for interactive text generation.

```bash
neurobrix chat [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--max-tokens` | Maximum tokens per response | model default |
| `--temperature` | Sampling temperature (0 = greedy) | model default |
| `--repetition-penalty` | Repetition penalty (1.0 = none) | 1.0 |

**In-session commands:**

| Command | Description |
|---------|-------------|
| `/new` | Start a new conversation (clears KV cache) |
| `/context` | Show token usage and cache state |
| `/status` | Show engine status and memory |
| `/quit` | Exit chat |

### stop

Stop the serving daemon and free GPU memory.

```bash
neurobrix stop
```

Uses a graceful shutdown sequence: socket request → SIGTERM → SIGKILL (escalation with timeouts).

---

## Single-Shot Inference

### run

Execute a model once. Loads weights, runs inference, saves output, then unloads.

```bash
neurobrix run --model <name> --hardware <profile> --prompt <text> [options]
```

| Flag | Description |
|------|-------------|
| `--model` | Model name (required) |
| `--hardware` | Hardware profile ID (required) |
| `--prompt` | Input text (required) |
| `--steps` | Number of inference steps (diffusion) |
| `--cfg` | Classifier-free guidance scale |
| `--height` / `--width` | Output resolution in pixels |
| `--output` | Output file path |
| `--seed` | Random seed for reproducibility |
| `--temperature` | Sampling temperature (LLM) |
| `--repetition-penalty` | Repetition penalty (LLM) |
| `--chat` / `--no-chat` | Force chat template on/off |
| `--set KEY=VALUE` | Override any runtime variable |
| `--seq_aten` | Use native ATen dispatch (debug) |
| `--triton` | Use Triton kernels (experimental) |

Examples:
```bash
# Image generation
neurobrix run --model PixArt-Sigma-XL-2-1024-MS --hardware v100-32g \
    --prompt "a sunset over mountains" --steps 20

# Text generation
neurobrix run --model deepseek-moe-16b-chat --hardware c4140-4xv100-custom-nvlink \
    --prompt "Explain quantum computing" --temperature 0.7

# Override runtime variables
neurobrix run --model 1600m-1024 --hardware v100-32g \
    --prompt "A cat" --set global.guidance_scale=7.5
```

> **Note:** For repeated use, prefer `serve` + `chat` to avoid reloading weights each time.

---

## Model Management

### hub

Browse models available on the [NeuroBrix Hub](https://neurobrix.es/models).

```bash
neurobrix hub [options]
```

| Flag | Description |
|------|-------------|
| `--category` | Filter by category: `IMAGE`, `LLM`, `VIDEO`, `AUDIO` |
| `--search` | Search by model name |

### import

Download a model from the hub.

```bash
neurobrix import <org/name> [options]
```

| Flag | Description |
|------|-------------|
| `--force` | Re-download even if already installed |
| `--no-keep` | Delete .nbx archive after extraction (saves disk space) |

Example:
```bash
neurobrix import sana/1600m-1024 --no-keep
```

### list

Show installed models.

```bash
neurobrix list [--store]
```

- Default: Shows models in cache (ready to run)
- `--store`: Shows downloaded .nbx archives

### remove

Remove a model from local storage.

```bash
neurobrix remove <model_name> [options]
```

| Flag | Description |
|------|-------------|
| (none) | Remove from cache only (keep .nbx in store) |
| `--store` | Remove .nbx only (keep extracted model) |
| `--all` | Remove from both cache and store |

### clean

Wipe all local models.

```bash
neurobrix clean [--store] [--cache] [--all] [-y]
```

Requires at least one of `--store`, `--cache`, or `--all`. Shows a summary (file count, total size) and asks for confirmation unless `-y` is passed.

---

## Inspection

### info

Display system information.

```bash
neurobrix info [options]
```

| Flag | Description |
|------|-------------|
| `--models` | Show installed models |
| `--hardware` | Show available hardware profiles |
| `--system` | Show GPU and system info |

### inspect

Inspect a .nbx container without extracting it.

```bash
neurobrix inspect <path.nbx> [options]
```

| Flag | Description |
|------|-------------|
| `--topology` | Show execution flow |
| `--weights` | Show weight information |

### validate

Validate .nbx file integrity.

```bash
neurobrix validate <path.nbx> [options]
```

| Flag | Description |
|------|-------------|
| `--level` | Validation depth: `structure`, `schema`, `coherence`, `deep` |
| `--strict` | Treat warnings as errors |

---

## Storage Layout

```
~/.neurobrix/
├── store/              Downloaded .nbx archives (transport format)
│   └── model-name.nbx
└── cache/              Extracted models (runtime reads from here)
    └── model-name/
        ├── manifest.json
        ├── topology.json
        ├── defaults.json
        └── ...
```

---

## Hardware Profiles

| Profile | GPUs | VRAM | Interconnect |
|---------|------|------|--------------|
| `v100-16g` | 1x V100 16GB | 16 GB | — |
| `v100-32g` | 1x V100 32GB | 32 GB | — |
| `v100-32g-2` | 2x V100 32GB | 64 GB | PCIe |
| `v100-32g-3` | 3x V100 32GB | 96 GB | PCIe |
| `v100-32g-x2-nvlink` | 2x V100 32GB | 64 GB | NVLink 300 GB/s |
| `c4140-4xv100-16GB-nvlink` | 4x V100 16GB | 64 GB | NVLink 300 GB/s |
| `c4140-4xv100-custom-nvlink` | 4x V100 (mixed) | 96 GB | NVLink 300 GB/s |

Custom profiles can be added as YAML files in the hardware configuration directory.

---

## Debug Environment Variables

| Variable | Effect |
|----------|--------|
| `NBX_DEBUG=1` | Verbose logging (forces GPU sync, ~2x slower) |
| `NBX_TRACE_ZEROS=1` | Find first op producing all-zeros from non-zero input |
| `NBX_TRACE_NAN=1` | Trace NaN/Inf in first few ops |
| `NBX_NAN_GUARD=1` | Replace NaN with 0 in every op (expensive) |
