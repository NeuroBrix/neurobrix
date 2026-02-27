# NeuroBrix Architecture

> **Version**: 0.1.0 | **Status**: Production | **Last Update**: February 2026

## Overview

NeuroBrix is a universal deep learning inference engine built on three core layers:

```
.nbx Container ──> Prism Solver ──> Execution Engine ──> Output
 (model data)      (hardware)       (graph runtime)
```

The runtime has **zero domain knowledge**. It does not know what an "image", "token", or "latent" is. It only sees tensors, axes, and execution graphs.

---

## Execution Modes

| Mode | Flag | Implementation | Use Case |
|------|------|----------------|----------|
| **Compiled** | (default) | CompiledSequence + DtypeEngine AMP | Production (80-95% GPU utilization) |
| **Native** | `--sequential` | Sequential ATen dispatcher | Debugging |
| **Triton** | `--triton` | Custom Triton kernels | R&D / benchmarking |

### Compiled Mode (Default)

The compiled mode pre-compiles the entire execution graph at load time into a **CompiledSequence** — a zero-overhead execution path:

- **TensorArena**: `__slots__`-based list for O(1) tensor access via integer slots
- **CompiledOp**: Frozen dataclass with pre-resolved function pointers and slot indices
- **DtypeEngine**: Implements PyTorch AMP autocast rules for numerical stability
- **Dead tensor analysis**: Liveness tracking to free memory during execution
- **Symbolic dimensions**: SymbolArg/ProductArg support for dynamic shapes (batch size, sequence length) resolved at runtime

```
Legacy path: ~15,000 dict lookups per transformer step
Compiled path: 0 dict lookups (integer-indexed memory arena)
```

---

## Prism: Hardware Solver

Prism analyzes the model's memory footprint against your hardware profile and selects the optimal execution strategy automatically.

### Strategy Cascade

Prism applies a scoring cascade — if the best strategy doesn't fit, it tries the next:

```
single_gpu (1000)
  → single_gpu_lifecycle (900)
    → pipeline_parallel (850)
      → component_placement (750)
        → block_scatter (700)
          → weight_sharding (680)
            → component_placement_lazy (400)
              → lazy_sequential (300)
                → zero3 (100)
```

Scores are bandwidth-adjusted: interconnect speed (Gbps) and boundary count affect the final score.

### Strategy Descriptions

| Strategy | Description |
|----------|-------------|
| `single_gpu` | All components on one GPU |
| `single_gpu_lifecycle` | Components loaded/unloaded sequentially on one GPU |
| `component_placement` | Whole components on different GPUs |
| `pipeline_parallel` | Per-layer sequential fill across GPUs (like Accelerate) |
| `block_scatter` | Block-level best-fit distribution across GPUs |
| `weight_sharding` | Weight-file round-robin across GPUs |
| `component_placement_lazy` | Component placement with lazy weight swap |
| `lazy_sequential` | Stream components through limited VRAM |
| `zero3` | CPU offload with GPU compute |

### Memory Estimation

Prism estimates per-component memory from the NBX container:
- **Weights**: Summed from safetensors headers (no tensor loading)
- **KV cache** (LLMs): Computed from `defaults.json` — `max_tokens × num_layers × 2 × num_kv_heads × head_dim × dtype_bytes`
- **Lifecycle**: Components classified as persistent (loaded simultaneously) or transient (loaded on demand)

### Dtype Resolution

```
Hardware profile (supports_dtypes: ["float32", "float16"])
  → Prism resolves optimal dtype per component
  → DtypeEngine applies AMP rules per operation
  → WeightLoader converts weights at load time
```

**Single source of truth**: All dtype string parsing, torch.dtype conversion, and Prism bf16↔fp16 remapping lives in `core/dtype/config.py` (`parse_dtype()`, `DTYPE_MAP`). No runtime file has its own dtype mapping.

### Heterogeneous GPU Support

Prism supports mixed GPU configurations (e.g., 2x V100-16GB + 2x V100-32GB). Block-level sharding is weighted by available VRAM per device.

---

## Execution Flows

The runtime supports three execution flow types, determined by the model's topology:

### Iterative Process (Diffusion Models)

```
pre_loop:   [text_encoder]
loop:       [transformer] × N steps (driven by scheduler)
post_loop:  [vae]
```

Used by: PixArt, Sana, FLUX, CogVideoX

### Autoregressive (Language Models)

```
prefill:    Process full prompt through language_model
decode:     Generate tokens one at a time with KV cache
```

Used by: DeepSeek-MoE, Qwen, TinyLlama

### Forward Pass (Single-Shot)

```
forward:    [component_1] → [component_2] → output
```

Used by: Whisper (audio transcription)

---

## DtypeEngine: Automatic Mixed Precision

The DtypeEngine implements PyTorch AMP autocast rules for numerical stability:

| Category | Behavior | Example Ops |
|----------|----------|-------------|
| **FP32 ops** | Always upcast to fp32 | `pow`, `rsqrt`, `softmax`, `sum`, `mean`, `layer_norm` |
| **FP16 ops** | Run in compute dtype (fp16) | `mm`, `conv2d`, `linear` |
| **Promote ops** | Promote to highest input dtype | `add`, `mul`, `cat` |
| **Overflow protect** | Post-compute clamp for fp16 | `add`, `sub` (residual accumulation) |

**Why this matters**: Without AMP, RMSNorm's `pow → mean → rsqrt` chain overflows in fp16 (rsqrt(Inf) = 0 → all zeros). With AMP, pow upcasts to fp32 and the chain stays stable.

**Special cases**:

- `bmm` is classified as FP32 (not FP16) — intentional deviation from PyTorch for T5-XXL stability
- `polar` and `view_as_complex` are FP32 — protects RoPE complex number computations (DeepSeek, Llama)
- Overflow protection on `add`/`sub` clamps fp16 results to ±65504 to prevent Inf → NaN propagation in deep residual chains

---

## Serving Architecture

NeuroBrix supports two execution modes:

### Serve Mode (Recommended)

```bash
neurobrix serve --model <name> --hardware <profile>
neurobrix chat [--temperature T]
neurobrix stop
```

- Weights loaded once into VRAM, kept warm
- Background daemon with Unix socket communication
- Automatic idle timeout (default: 30 min)
- Session-based KV cache management

### Single-Shot Mode

```bash
neurobrix run --model <name> --hardware <profile> --prompt <text>
```

- Loads model, executes, unloads
- Suitable for batch processing or one-off experiments

---

## Atomic Operations

The runtime operates on ~80 ATen operations that cover all model architectures:

| Category | Operations |
|----------|-----------|
| **Tensor** | matmul, add, mul, div, sub, pow, sqrt, exp, log |
| **Neural** | conv2d, conv_transpose2d, layer_norm, group_norm, softmax, embedding |
| **Activation** | relu, gelu, silu, sigmoid, tanh |
| **Control** | reshape, transpose, cat, split, gather, scatter, slice, pad, where |
| **Special** | scaled_dot_product_attention, interpolate, upsample |

Every neural network is a combination of these primitives. A transformer block = linear + norm + attention + linear. FlashAttention = a single `scaled_dot_product_attention` call.

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
