# NBX Symbolic Shapes — Master Contract

## TL;DR

Every NBX `graph.json` tensor descriptor exposes two shape fields:

```jsonc
{
  "tensors": {
    "input::z": {
      "shape": [1, 32, 64, 64],                  // CONCRETE value, the size the container was built at
      "symbolic_shape": {
        "dims": [
          {"type": "symbol", "id": "s0", ...},   // SYMBOLIC (also records its built value)
          32,                                     // concrete dim
          {"type": "symbol", "id": "s1", ...},   // SYMBOLIC
          {"type": "symbol", "id": "s2", ...}    // SYMBOLIC
        ],
        "concrete": [1, 32, 64, 64]
      }
    }
  }
}
```

**Master rule:** any runtime code that reads a graph shape to make a **decision** (allocation size, kernel
choice, dispatch strategy, padding, tiling) MUST read `symbolic_shape.dims` BEFORE it draws a conclusion
from the concrete `shape` array. The concrete shape is only the size the container was built at. The
symbolic markers name the dims the runtime binds to the real value of each request.

## Why symbolic shapes exist

Image VAEs (Sana DC-AE, PixArt's AutoencoderKL) and text encoders (Gemma, T5, CLIP) adapt to the input
by construction: the same graph runs at any reasonable height and width, or any sequence length. The
container records one representative size and marks the dims that vary as symbolic. At run time the engine
binds each symbol and substitutes the actual size.

Without symbolic shapes, every aspect ratio, resolution or token count would need its own graph.

With symbolic shapes honoured, **one** graph serves the whole range, and tiling (the TilingEngine) is used only
for graphs that cannot adapt (fixed-grid models such as Swin2SR).

## Where symbolic shapes appear

| Component family | Typical symbolic dims | Bound at |
|---|---|---|
| Image VAE (decode) | spatial H, W (and possibly batch) | post-loop dispatch |
| Image transformer (DiT) | batch, spatial sequence length, sometimes channels | per-step dispatch |
| Text encoder | batch, sequence length | once per request |
| LLM transformer (autoregressive) | batch, sequence length (KV cache growth) | per token |
| Audio encoder/decoder | batch, time | per request |

Any 4D `[B, C, H, W]` or 5D `[B, C, T, H, W]` tensor whose `symbolic_shape` marks indices ≥ 2 as symbolic is
**spatial-adaptive**: never tile it, never allocate from its built size, never choose a kernel from the
concrete spatial value alone.

A symbol the runtime cannot bind is refused by name; the engine never falls back to the built size.

## Where the contract is enforced

- `core/runtime/graph/tensor_resolver.py` and `core/runtime/shape_resolver.py`: symbol binding.
- `core/runtime/graph/compiled_sequence.py`: refuses a symbol left unbound.
- `core/runtime/graph_executor.py::_compute_computable_buffers()`: recomputes buffers such as
  `sincos_2d_pos_embed` from the bound symbols.
- `core/module/tiling_engine.py::from_component_config()`: refuses to build a TilingEngine when a spatial
  dim is symbolic.

## Implementation note

To consult `symbolic_shape`, walk `tensor["symbolic_shape"]["dims"]` and check
`isinstance(dim, dict) and dim.get("type") == "symbol"` on the dims your decision depends on. See
`tiling_engine.py:from_component_config` for the canonical pattern.
