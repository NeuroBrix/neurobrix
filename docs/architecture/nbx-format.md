# The NBX Container Format

> **Version**: 0.1.0 | **Last Update**: February 2026

## What is NBX?

The `.nbx` file is a self-contained archive that holds everything needed to run a deep learning model. It is the **single source of truth** — the runtime reads only from the NBX container and never requires access to the original model code.

NeuroBrix introduces the NBX format as a **universal standard for AI model packaging**. Rather than each framework defining its own way to store and load models, NBX captures the complete execution blueprint — graph, weights, topology, and metadata — in a single portable archive.

## Design Goals

- **Framework-independent**: No dependency on HuggingFace, diffusers, or any specific library at runtime
- **Self-describing**: All shapes, dtypes, execution flow, and component relationships are declared
- **Hardware-agnostic**: A single NBX file works on any supported GPU (dtype conversion happens at runtime)
- **Transport-oriented**: The `.nbx` archive is for distribution; runtime reads from extracted cache

## Container Structure

```
model.nbx (ZIP archive)
├── manifest.json                  # Model identity and metadata
├── topology.json                  # Execution flow and component connections
│
├── runtime/
│   ├── defaults.json              # User-tunable defaults (steps, guidance, max_tokens)
│   └── variables.json             # Variable contracts with resolvers
│
├── components/
│   ├── transformer/
│   │   ├── graph.json             # TensorDAG (ATen ops with shapes/dtypes)
│   │   ├── runtime.json           # Component attributes (channels, extents)
│   │   ├── profile.json           # Component config (backbone type, trace values)
│   │   ├── config.json            # Original model config
│   │   ├── weights_index.json     # Maps tensor names → shard files
│   │   └── weights/
│   │       ├── shard_000.safetensors
│   │       └── shard_001.safetensors
│   ├── vae/
│   │   └── ...
│   └── text_encoder/
│       └── ...
│
└── modules/
    ├── scheduler/                 # Scheduler state and config
    └── tokenizer/                 # Tokenizer config
```

## Key Files

### manifest.json

Model identity: family (image, llm, audio, video), generation type (iterative, autoregressive), list of neural components and modules.

### topology.json

Execution flow definition. Describes the order of component execution, loop structures, state variables, and inter-component connections.

### graph.json (per component)

The computation graph as a TensorDAG — a directed acyclic graph of ATen operations. Each node has:
- `op_uid`: Unique identifier (e.g., `aten.unsqueeze::0`)
- `op_type`: ATen operation (e.g., `aten::mm`, `aten::scaled_dot_product_attention`)
- `input_tensor_ids`: References to input tensors
- `output_tensor_ids`: References to output tensors
- `input_shapes` / `output_shapes`: Tensor dimensions (may include symbolic expressions)
- `input_dtypes` / `output_dtypes`: Data types per tensor
- `device`: Target device at trace time
- `attributes`: Operation-specific parameters
- `parent_module`: Original module path (for debugging)

### profile.json (per component)

Component configuration and trace-time metadata:
- `backbone_type`: Architecture type (e.g., `unet`, `dit`, `decoder`)
- `config`: Original model parameters
- `globals`: Trace-time global values

### runtime.json (per component)

Component attributes used by the runtime for memory allocation:
- `state_channels`, `state_extent_0`, `state_extent_1`: State tensor dimensions
- `attention_head_dim`: Attention configuration
- `interface`: Input/output specifications

### weights_index.json (per component)

Maps weight tensor names to their shard files, enabling partial loading.

### defaults.json

User-tunable parameters with their default values:
- Diffusion: `num_inference_steps`, `guidance_scale`, `height`, `width`
- LLM: `max_tokens`, `temperature`, `top_k`, `top_p`, `repetition_penalty`

### variables.json

Variable resolution contracts. Each variable has a resolver that describes how to compute its value at runtime (from other variables, allocations, or constants).

## How NBX Files Are Created

Models available on the [NeuroBrix Hub](https://neurobrix.es/models) are built using our proprietary tracing technology. We capture the computation graph at a low level using a custom `TorchDispatchMode` mechanism, preserving the exact execution semantics of the original model while making it framework-independent.

This process produces a self-describing container that any compatible runtime can execute without needing access to the original model code or its dependencies.

Our goal is to establish a structured, vendor-neutral format that enables better organization and interoperability across the AI ecosystem — separating the model definition from the execution environment.

## Runtime Cache

The `.nbx` file is only used for transport. At import time, it is extracted to a local cache:

```
~/.neurobrix/
├── store/              Downloaded .nbx archives (transport)
│   └── model-name.nbx
└── cache/              Extracted models (runtime reads from here)
    └── model-name/
        ├── manifest.json
        ├── topology.json
        ├── defaults.json
        ├── runtime/
        ├── components/
        └── modules/
```

The runtime **always** reads from `~/.neurobrix/cache/`, never from the `.nbx` file directly.

## Weight Storage

Weights are stored in [safetensors](https://github.com/huggingface/safetensors) format, sharded into 2GB chunks. The stored dtype preserves the original model precision. Dtype conversion to the target hardware (fp16, fp32) is performed at runtime by the Prism solver.
