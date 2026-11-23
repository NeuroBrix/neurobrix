# The NBX Container Format

## What is NBX?

The `.nbx` file is a self-contained archive that holds everything needed to run a model. It is the **single
source of truth**: the runtime reads only the container and never needs the original model code.

An NBX container holds the complete execution blueprint of a model (graphs, weights, topology and metadata)
in one portable archive. The engine reads it; it does not need the framework the model was written in.

The `.nbx` packages on the [NeuroBrix hub](https://neurobrix.org/models) are built by the NeuroBrix team.
The engine only consumes them.

## Design goals

- **Framework-independent**: no dependency on Hugging Face, diffusers or any model library at run time
- **Self-describing**: shapes, dtypes, execution flow and component links are all declared
- **Hardware-agnostic**: one container runs on every supported device; the hardware profile decides the
  compute dtype at run time
- **Transport-oriented**: the `.nbx` archive is for distribution; the runtime reads the extracted copy

## Container structure

```
model.nbx (ZIP archive)
├── manifest.json                  # model identity and metadata
├── topology.json                  # execution flow and component connections
│
├── runtime/
│   ├── defaults.json              # request defaults (steps, guidance, max_tokens, ...)
│   └── variables.json             # variable contracts and how each is resolved
│
├── components/
│   ├── transformer/
│   │   ├── graph.json             # TensorDAG (ATen ops with shapes and dtypes)
│   │   ├── runtime.json           # component attributes (channels, extents)
│   │   ├── profile.json           # component profile (dtype, values fixed when the container was built)
│   │   ├── config.json            # original model config
│   │   ├── weights_index.json     # tensor names → shard files, with dtype and bytes
│   │   └── weights/
│   │       └── *.safetensors
│   ├── vae/
│   │   └── ...
│   └── text_encoder/
│       └── ...
│
└── modules/
    ├── scheduler/                 # scheduler config
    └── tokenizer/                 # tokenizer files
```

## Key files

### manifest.json

Model identity: family (image, llm, audio, video, ...), generation type (iterative, autoregressive, ...),
the list of neural components and modules.

### topology.json

The execution flow: the order of components, loop structures, state variables and the connections between
components.

### graph.json (per component)

The computation graph as a TensorDAG: a directed acyclic graph of ATen operations. Each node has:
- `op_uid`: unique identifier (e.g. `aten.unsqueeze::0`)
- `op_type`: the ATen operation (e.g. `aten::mm`, `aten::scaled_dot_product_attention`)
- `input_tensor_ids` / `output_tensor_ids`: references to tensors
- `input_shapes` / `output_shapes`: dimensions (may carry symbolic expressions)
- `input_dtypes` / `output_dtypes`: data types per tensor
- `attributes`: operation parameters
- `parent_module`: the original module path (for debugging)

Shapes that can change at run time are marked symbolic. See [Symbolic shapes](symbolic-shapes-contract.md).

### profile.json (per component)

The component's profile: its stored `dtype`, its architecture type, the original model parameters and the
values that were fixed when the container was built.

### runtime.json (per component)

Component attributes Prism reads for memory planning: state channels and extents, attention configuration,
the input and output interface.

### weights_index.json (per component)

Maps each weight tensor to its shard file, with its dtype and size, so weights can be loaded in part and
priced before loading.

### defaults.json

Request defaults:
- Diffusion: `num_inference_steps`, `guidance_scale`, `height`, `width`
- LLM: `max_tokens`, `temperature`, `top_k`, `top_p`, `repetition_penalty`

A value given on the command line wins over `defaults.json`, which wins over the family config. A value found
in none of them is refused by name.

### variables.json

Variable contracts. Each variable says how its value is resolved at run time (from other variables, from
the request, or from a constant).

## Runtime cache

The `.nbx` file is only used for transport. `neurobrix import` extracts it once to a local cache:

```
~/.neurobrix/
├── store/              downloaded .nbx archives (transport)
│   └── model-name.nbx
└── cache/              extracted models (the runtime reads from here)
    └── model-name/
        ├── manifest.json
        ├── topology.json
        ├── runtime/
        ├── components/
        └── modules/
```

The runtime always reads the extracted cache, never the `.nbx` file directly. `NEUROBRIX_HOME`,
`NEUROBRIX_CACHE` and `NEUROBRIX_STORE` move these directories.

## Weight storage

Weights are stored in [safetensors](https://github.com/huggingface/safetensors) format, in the precision the
model was released in. Each component runs in the dtype its weights store; where the hardware cannot compute
that dtype, the hardware profile decides the conversion at run time. See
[the dtype per component](../reference/the-dtype-per-component.md).
