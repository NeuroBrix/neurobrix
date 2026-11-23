# NeuroBrix Runtime

## Overview

NeuroBrix is an inference engine built on three layers:

```
.nbx container ──> Prism ──> execution engine ──> output
 (model data)     (plan)     (graph runtime)
```

The runtime has **no domain knowledge**. It does not know what an image, a token or a latent is. It sees
tensors, axes and execution graphs. The container declares; the engine obeys.

The engine does no tensor logic of its own. On the Triton branch every tensor, memory block and kernel
comes from the **nbxtensor** library, a separate package the engine depends on. On the PyTorch branch the
same work is done by PyTorch.

---

## Execution modes

There are two compute branches, PyTorch and Triton. Each has a compiled variant and an op-by-op variant:
four modes in all. The branches share the `.nbx` container and the Prism plan.

| Mode | Flag | Branch | Implementation | Use |
|------|------|--------|----------------|-----|
| **Compiled** | `--compiled` | PyTorch | CompiledSequence + DtypeEngine | Default when PyTorch is installed |
| **Sequential** | `--sequential` | PyTorch | ATen op-by-op dispatcher | Op-by-op reference and debugging |
| **Triton** | `--triton` | Triton | TritonSequence on nbxtensor | No PyTorch; default when PyTorch is not installed |
| **Triton sequential** | `--triton-sequential` | Triton | TritonSequentialDispatcher (op-by-op) | Op-by-op kernel debugging |

When no flag is given, the engine runs the compiled PyTorch mode if PyTorch is installed, and the Triton mode
if it is not. The launch line says which one it chose. The four flags are mutually exclusive.

A Triton mode needs a Triton build for the machine: the `triton` wheel on Linux x86_64 and aarch64, the
Metal build on Apple Silicon (`python -m nbxtensor install-metal`). Where none is installed, a Triton mode is
refused by name with the install line. A PyTorch mode without PyTorch is refused the same way.

### Compiled mode

The compiled mode turns the whole execution graph into a **CompiledSequence** at load time:

- **TensorArena**: tensors are held in integer-indexed slots, so the hot loop does no dictionary lookups
- **CompiledOp**: each op is a frozen record with its function and slot indices resolved once
- **DtypeEngine**: mixed-precision rules per op (below)
- **Liveness**: a tensor is freed after its last use
- **Symbolic dimensions**: batch size, sequence length and spatial sizes are bound at run time from the
  container's symbolic shapes (see [Symbolic shapes](symbolic-shapes-contract.md))

### Triton mode (`--triton`)

The Triton mode runs a model with no PyTorch in the process.

- **NBXTensor**: the tensor type of nbxtensor, with its own device memory management
- **TritonSequence**: the same arena-based compiled loop as CompiledSequence, on NBXTensor
- **Kernels**: every compute op runs on nbxtensor's Triton kernels (matrix products, attention, norms,
  convolutions, and more). The engine itself holds no kernel
- **KV cache**: preallocated buffers, grouped-query attention supported
- **Weight loading**: safetensors read by nbxtensor straight into NBXTensor buffers, with no PyTorch
- **Mixed precision**: TritonDtypeEngine applies the same op classes as the compiled branch
- **Decode replay**: on one card, decode steps are replayed from a recorded launch plan (CUDA graphs on NVIDIA)
- **Devices**: nbxtensor drives NVIDIA (CUDA), Apple Silicon (Metal) and CPU (triton-cpu) through its own
  backends. AMD and Intel GPUs, through the same Triton kernels, are targets not yet validated on a card.

```
src/neurobrix/triton/
  sequence.py      — compiled hot loop (arena + closures)
  sequential.py    — op-by-op dispatcher
  kv_cache.py      — KV cache
  dtype.py         — TritonDtypeEngine
  weight_loader.py — safetensors → NBXTensor (through nbxtensor)
  replay.py        — frozen-plan replay of the hot loop
  flow/            — flow handlers (autoregressive, diffusion, audio, ...)
```

---

## Prism: the planner

Prism reads the model's memory needs from the container and the machine's facts from its hardware profile,
then chooses how to place the model.

### How a plan is chosen

Prism tries every strategy and keeps each one that fits. It then ranks them by **what they do with the
weights**. The best plan of the first non-empty rank wins:

| Rank | Strategies | What it costs |
|---|---|---|
| 1. Resident, one device | `single_gpu`, `single_gpu_lifecycle` | nothing moves |
| 2. Resident, spread over devices | `component_placement`, `block_scatter`, `pipeline_parallel`, `weight_sharding` | activations cross the link |
| 3. Resident, the overflowing ops cut into tiles | `op_level_tiling` | tile edges and extra launches |
| 4. Weights moved through the card per use or per phase | `zero3`, `component_placement_lazy`, `lazy_sequential`, `layer_streaming` | bytes over the host link |
| 5. Host compute | `cpu_execution`, `cpu_streaming` | the host's speed |

Inside a rank, the estimated time of each candidate decides. The estimate uses the facts the hardware
profile declares (compute rate, memory bandwidth, link bandwidth). `neurobrix run --explain-plan` prints the
plan, every candidate and why each one was refused.

### Strategies

| Strategy | Description |
|----------|-------------|
| `single_gpu` | All components on one device |
| `single_gpu_lifecycle` | Components loaded and released in turn on one device |
| `component_placement` | Whole components on different devices |
| `pipeline_parallel` | Layers filled in order across devices |
| `block_scatter` | Blocks distributed across devices by best fit |
| `weight_sharding` | Weight files spread across devices |
| `op_level_tiling` | Ops that overflow the device are cut into tiles |
| `component_placement_lazy` | Component placement with weights swapped in when needed |
| `lazy_sequential` | Components streamed through limited memory, one per phase |
| `layer_streaming` | Layers streamed through the device |
| `zero3` | Weights kept in host memory, blocks streamed to the device for compute |
| `cpu_execution`, `cpu_streaming` | Compute on the host |

A mixture-of-experts model can keep its routed experts in an expert store, with only the experts in use on
the device.

### Memory estimation

Prism estimates each component's memory from the container:
- **Weights**: summed from the safetensors headers and `weights_index.json`, with no tensor loaded
- **KV cache** (language models): from the attention shapes the container declares and the requested length
- **Lifecycle**: components are persistent (held together) or transient (loaded on demand)

### Dtype

```
components/<c>/profile.json dtype (the dtype the weights are stored in)
  → the hardware profile's supported dtypes decide what the device computes in
  → DtypeEngine applies the op classes per operation
  → the weight loader converts at load time where needed
```

All dtype string parsing and conversion for the PyTorch branch lives in `core/dtype/config.py`
(`parse_dtype()`). See [the dtype per component](../reference/the-dtype-per-component.md).

### Mixed GPUs

Prism plans on mixed configurations, for example two 16 GB and two 32 GB cards in one machine. Placement
follows each device's capacity.

---

## Execution flows

The model's topology names its flow. The main ones:

### Iterative process (diffusion models)

```
pre_loop:   [text_encoder]
loop:       [transformer] × N steps (driven by the scheduler)
post_loop:  [vae]
```

Used by: PixArt, Sana, Flex.1, CogVideoX, Wan

### Autoregressive (language models)

```
prefill:    the whole prompt through the language model
decode:     one token at a time, with the KV cache
```

Used by: DeepSeek, Qwen, TinyLlama, Granite

### Forward pass (single shot)

```
forward:    [component_1] → [component_2] → output
```

Used by: upscalers

Other flows cover encoder-decoder speech models (Whisper), RNN-T transducers (Parakeet), audio-conditioned
language models (Voxtral, Canary-Qwen, Granite-Speech), vision-language models, and text-to-speech pipelines
(Kokoro, Orpheus, Chatterbox, VibeVoice, OpenAudio).

---

## DtypeEngine: mixed precision

The DtypeEngine applies an op-class policy. The policy file is nbxtensor's (`nbxtensor/dtype/policy.yml`), so
both branches read the same classes:

| Class | Behaviour | Example ops |
|----------|----------|-------------|
| **float32** | Always computed in float32 | `pow`, `rsqrt`, `softmax`, `sum`, `layer_norm`, `polar`, `view_as_complex` |
| **lower** | Computed in the policy's compute dtype | `mm`, `bmm`, `addmm`, `conv2d`, `linear` |
| **promote** | Computed in the widest input dtype | `addcmul`, `addcdiv`, `index_add` |

**Why it matters**: without these rules, an RMSNorm's `pow → mean → rsqrt` chain can overflow in float16.
With them, `pow` runs in float32 and the chain stays finite.

**Hardware-driven protection**: on hardware whose profile has no bfloat16 (a V100, for example), matrix
products and division are computed in float32 where float16 could overflow. On bfloat16 hardware no such
upcast is needed. The decision reads the hardware profile, never a model name.

---

## Serving

### Serve mode

```bash
neurobrix serve --model <name>
neurobrix chat
neurobrix stop
```

- Weights are loaded once and kept warm when the plan allows it
- A background daemon, reached over a Unix socket on macOS and Linux and over TCP on loopback on Windows
- An idle timeout stops the daemon (default 1800 s, `--timeout`)
- The conversation and its KV cache live in the daemon
- `--http` also serves an OpenAI-compatible HTTP API on loopback

See [the daemon wire protocol](../reference/wire-protocol.md).

### Single-shot mode

```bash
neurobrix run --model <name> --prompt <text>
```

- Loads the model, runs it, exits
- Suited to batch jobs and one-off runs

`--hardware <profile>` names a hardware profile. Without it, the engine uses the profile it keeps for the
machine. On the first launch on a machine that has no hardware profile, NeuroBrix creates that profile once; every
later launch reads it. A profile given with `--hardware` is read as it is.

---

## Operations

The runtime executes ATen operations recorded in the container's graphs:

| Category | Operations |
|----------|-----------|
| **Tensor** | matmul, add, mul, div, sub, pow, sqrt, exp, log |
| **Neural** | conv2d, conv_transpose2d, layer_norm, group_norm, softmax, embedding |
| **Activation** | relu, gelu, silu, sigmoid, tanh |
| **Control** | reshape, transpose, cat, split, gather, scatter, slice, pad, where |
| **Special** | scaled_dot_product_attention, interpolate, upsample |

Every network is built from such primitives. A transformer block is linear + norm + attention + linear.

---

## Project structure

```
src/neurobrix/
  cli/                  command-line interface
    commands/           run, serve, chat, upscale, registry (import, list, hub, remove), ...
  core/                 runtime engine
    runtime/            executor, graph engine, CompiledSequence
    prism/              planner
    strategies/         placement strategies
    flow/               execution flows
    dtype/              mixed-precision engine
    cfg/                classifier-free guidance
    module/             tokenizer, scheduler, text and vision processors
    state/              state kept between steps (KV cache, sessions)
    io/                 weight loading (PyTorch branch)
  kernels/              op classification and kernel certification (the kernels are in nbxtensor)
  triton/               Triton-branch engine (--triton)
  nbx/                  .nbx container reader
  serving/              daemon, client, OpenAI-compatible API
  processors/           sub-components a model declares (fetched on demand)
  config/               hardware profiles, vendor files, family configs
```

The tensor library is its own project: [nbxtensor](https://github.com/NeuroBrix/nbxtensor).
