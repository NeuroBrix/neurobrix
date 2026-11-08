# NeuroBrix CLI Reference

> **Version**: 0.3.0 | **Last Update**: July 2026

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
| [`upscale`](#upscale) | Upscale an image with a super-resolution model |
| [`hub`](#hub) | Browse models on the NeuroBrix Hub |
| [`import`](#import) | Download a model from the hub |
| [`list`](#list) | Show installed models |
| [`remove`](#remove) | Remove a model |
| [`clean`](#clean) | Wipe all local models |
| [`info`](#info) | Display system information |
| [`inspect`](#inspect) | Inspect a .nbx container |
| [`validate`](#validate) | Validate .nbx file integrity |
| [`doctor`](#doctor) | Diagnose installation issues (PATH, etc.) |

---

## Model Serving (Recommended Workflow)

A daemon loads one model once and answers requests over a local socket. The full description (lifecycle, warm and
cold serving per Prism strategy, operations, cancellation, the conversation and its KV cache) is
[Serving and chat](../serving.md); the wire format is [the daemon wire protocol](wire-protocol.md).

### serve

Start the serving daemon. It loads the model once and keeps it loaded; on a warm plan the weights stay on the device
between requests.

```bash
neurobrix serve --model <name> [--hardware <profile>] [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model name (required) | — |
| `--hardware` | Hardware profile ID | this machine's default profile |
| `--timeout` | Idle timeout in seconds: the daemon exits when no connection is open and none arrives for this long | 1800 |
| `--foreground` | Serve in this process (blocks the terminal, for debugging) | false |
| `--compiled` | PyTorch fused graph | default where PyTorch is installed |
| `--sequential` | PyTorch op-by-op (debug) | false |
| `--triton` | NeuroBrix Triton kernels (compiled) | default where PyTorch is not installed |
| `--triton-sequential` | Triton kernels op-by-op (debug) | false |
| `--ready-fd` | POSIX: inherited pipe fd that receives the ready record (one JSON line), then is closed | — |
| `--ready-file` | File the ready record is written to, whole, once the daemon accepts | — |

At most one engine flag. Without `--foreground`, `serve` returns once the model is loaded and prints
`[Serve] Model loaded — <model> (<family>, warm serving)` (or `cold serving`), or `[Serve] Failed: <error>` with exit
code 1. It refuses to start (exit code 1) when a daemon is already running. `NBX_SOCKET_PATH` names an instance, so
several daemons can run side by side.

Examples:
```bash
# Auto-detect hardware
neurobrix serve --model TinyLlama-1.1B-Chat-v1.0

# Target a specific hardware profile and the Triton engine, one hour of idle time
neurobrix serve --model TinyLlama-1.1B-Chat-v1.0 --hardware v100-32g --triton --timeout 3600
```

### chat

Connect to a running daemon for interactive multi-turn chat. The daemon's model must be of the `llm` family.

```bash
neurobrix chat [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--max-tokens` | Maximum tokens per response | model default |
| `--temperature` | Sampling temperature | model default |
| `--repetition-penalty` | Repetition penalty (1.0 = none) | model default |
| `--agent` | Agent REPL: each input is a task the model completes with tools inside the workdir jail | false |
| `--workdir` | Agent workdir jail | current directory |
| `--max-turns` | Agent loop turn bound | family default |
| `--bash-timeout` | Agent bash timeout in seconds | family default |
| `--allow-network` | Allow network inside the agent sandbox | false |
| `--approve-all` | Confirm every agent tool execution interactively | false |
| `--yolo` | No confirmations (jail still enforced) | false |
| `--transcript-dir` | Agent transcript directory | `<workdir>/.nbx_agent/<ts>` |

**In-session commands:**

| Command | Description |
|---------|-------------|
| `/new` | Start a new conversation (clears the daemon's message history) |
| `/context` | Messages, estimated tokens and usage (against the container's `max_length`, else 4096) |
| `/status` | Model, family, engine mode and device memory in use |
| `/quit` (or `/exit`, `/q`) | Exit chat (the daemon keeps running) |

Each answer is printed whole (not streamed). Every turn re-sends the whole conversation to the model (no KV cache is
kept between turns); when the conversation outgrows the plan's KV cache, older turns are summarized and chat prints
`[Context compressed — older turns summarized]`. See [Chat and the KV cache](../serving.md#chat-and-the-kv-cache-today).

### stop

Stop the serving daemon, unload the weights and free device memory.

```bash
neurobrix stop
```

No flags. Escalates: a `shutdown` request on the socket (then up to 5 s for the exit) → `SIGTERM` (3 s) → `SIGKILL`
(`taskkill /F` on Windows), then removes the daemon's socket and pid files. With no daemon running it removes leftover
files and exits 0.

While a daemon runs, `neurobrix run` with the same model (or with `--model` omitted) is answered by the daemon; a
`run` of another model is refused until the daemon is stopped.

---

## Single-Shot Inference

### run

Execute a model once. Loads weights, runs inference, saves output, then unloads.

```bash
neurobrix run --model <name> --prompt <text> [--hardware <profile>] [options]
```

| Flag | Description |
|------|-------------|
| `--model` | Model name (required) |
| `--hardware` | Hardware profile ID (optional — auto-detects if omitted) |
| `--prompt` | Input text (required) |
| `--steps` | Number of inference steps (diffusion) |
| `--cfg` | Classifier-free guidance scale |
| `--height` / `--width` | Output resolution in pixels |
| `--output` | Output file path |
| `--seed` | Random seed for reproducibility |
| `--temperature` | Sampling temperature, 0 = greedy (text, vlm, audio_llm decodes) |
| `--top-k` | Keep the k most likely tokens, 0 = off |
| `--top-p` | Nucleus threshold, 1.0 = off |
| `--repetition-penalty` | Repetition penalty, 1.0 = none |
| `--chat` / `--no-chat` | Force chat template on/off |
| `--set KEY=VALUE` | Override any runtime variable |
| `--compiled` | PyTorch fused graph (explicit default) |
| `--sequential` | PyTorch op-by-op (debug) |
| `--triton` | NeuroBrix Triton kernels — vendor-agnostic compute path |
| `--triton-sequential` | Triton kernels op-by-op (debug) |

Family-specific input flags (`--audio`, `--input-image`, `--mask-image`, `--reference-audio`, `--speaker`, `--video`, `--num-frames`, `--fps`, `--system`, `--max-tokens`, `--mode`) are documented per family in [CLI Usage by Family](../cli_usage.md).

Examples:
```bash
# Image generation (auto-detect hardware)
neurobrix run --model PixArt-Sigma-XL-2-1024-MS \
    --prompt "a sunset over mountains" --steps 20

# Text generation with specific hardware
neurobrix run --model deepseek-moe-16b-chat --hardware c4140-4xv100-custom-nvlink \
    --prompt "Explain quantum computing" --temperature 0.7

# Override runtime variables
neurobrix run --model 1600m-1024 \
    --prompt "A cat" --set global.guidance_scale=7.5
```

> **Note:** For repeated use, prefer `serve` + `chat` to avoid reloading weights each time.

#### Sampling

Every text, vlm and audio_llm decode samples the same way, on both engines (`--compiled` and `--triton`): one sampler
applies the repetition penalty, the temperature, top-k and top-p, then draws (or takes the most likely token when the
temperature is 0). Each of the four values is taken from the first source that sets it:

1. the request: the flag above, or the same key in a `serve` request (the warm path forwards all four and
   `--max-tokens`);
2. the model's container: its `runtime/defaults.json`, the vendor's generation config as the container was built with
   it;
3. the family's documented defaults, in `config/families/<family>.yml` under `defaults:`. For `llm`, `vlm`,
   `audio_llm` and `multimodal`: temperature 0.7, top-k 50, top-p 0.9, repetition penalty 1.0.

A value the container leaves empty (`null`) lets the family's value through. A value no source sets stops the run with
an error naming it; the engine never substitutes one. The repetition penalty counts the prompt's tokens and the
generated ones for a text decode, and the generated ones for a decode fed with image or audio embeddings.

### upscale

Upscale an image with a super-resolution model.

```bash
neurobrix upscale --model <name> --input <img> --output <img> [options]
```

| Flag | Description |
|------|-------------|
| `--model` | Upscaler model name (required) |
| `--input` | Input image path, PNG/JPEG (required) |
| `--output` | Output image path, PNG (required) |
| `--mode` | Execution mode: `compiled` (default) / `sequential` / `triton` / `triton-sequential` |
| `--hardware` | Hardware profile ID (auto-detected if omitted) |

---

## Model Management

### hub

Browse models available on the [NeuroBrix Hub](https://neurobrix.es/models).

```bash
neurobrix hub [options]
```

| Flag | Description |
|------|-------------|
| `--category` | Filter by category: `IMAGE`, `LLM`, `VIDEO`, `AUDIO`, `SPEECH`, `UPSCALER` (case-insensitive) |
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
| `--registry` | Custom registry URL (default: neurobrix.es) |

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
| `--level` | Validation depth: `structure`, `schema`, `coherence` (default), `deep` |
| `--strict` | Exit on first failure |
| `--json` | Output results as JSON |
| `--verbose` / `-v` | Show detailed validation info |

### doctor

Diagnose installation issues (PATH problems, etc.).

```bash
neurobrix doctor
```

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

When `--hardware` is omitted, NeuroBrix auto-detects your system (GPUs, CPU, RAM, interconnects) and creates a `default.yml` profile. Use `--hardware` to target a specific subset of your machine.

| Profile | GPUs | VRAM | Interconnect |
|---------|------|------|--------------|
| `v100-16g` | 1x V100 16GB | 16 GB | — |
| `v100-32g` | 1x V100 32GB | 32 GB | — |
| `v100-32g-2` | 2x V100 32GB | 64 GB | PCIe |
| `v100-32g-3` | 3x V100 32GB | 96 GB | PCIe |
| `v100-32g-x2-nvlink` | 2x V100 32GB | 64 GB | NVLink 300 GB/s |
| `c4140-4xv100-16GB-nvlink` | 4x V100 16GB | 64 GB | NVLink 300 GB/s |
| `c4140-4xv100-custom-nvlink` | 4x V100 (mixed) | 96 GB | NVLink 300 GB/s |

NeuroBrix also runs on CPU-only machines — see [Hardware Profiles](../guide/hardware.md) and [Writing Hardware Profiles](../guide/hardware-profiles.md) for details.

---

## Debug Environment Variables

| Variable | Effect |
|----------|--------|
| `NBX_DEBUG=1` | Verbose logging (forces GPU sync, ~2x slower) |
| `NBX_TRACE_ZEROS=1` | Find first op producing all-zeros from non-zero input |
| `NBX_TRACE_NAN=1` | Trace NaN/Inf in first few ops |
| `NBX_NAN_GUARD=1` | Replace NaN with 0 in every op (expensive) |
