# NBX Symbolic Shapes — Master Contract

## TL;DR

Every NBX `graph.json` tensor descriptor exposes two shape fields:

```jsonc
{
  "tensors": {
    "input::z": {
      "shape": [1, 32, 64, 64],                  // CONCRETE build-time value
      "symbolic_shape": {
        "dims": [
          {"type": "symbol", "id": "s0", "trace": 1},
          32,                                     // concrete dim
          {"type": "symbol", "id": "s1", "trace": 64},   // SYMBOLIC
          {"type": "symbol", "id": "s2", "trace": 64}    // SYMBOLIC
        ],
        "concrete": [1, 32, 64, 64]
      }
    }
  }
}
```

**Master rule:** any runtime code that inspects an NBX graph shape to draw a
**decision** (allocation size, kernel selection, dispatch strategy, padding,
tiling, etc.) MUST consult `symbolic_shape.dims` BEFORE drawing conclusions
from the concrete `shape` array. The concrete shape is the value that
**happened to be used during build-time graph capture**; the symbolic markers
identify the dims that the runtime can rebind to any value via
CompiledSequence's symbol binding pass.

Reading only `tensor["shape"]` and ignoring `tensor["symbolic_shape"]` is the
pattern that produced the Sana 4Kpx triton crash documented in Layer 6 — the
TilingEngine activated on a DC-AE VAE whose H/W are symbolic, fed an
NBXTensor into the torch-only accumulator path, and `TypeError`'d.

## Why symbolic shapes exist

Diffusion image VAEs (Sana DC-AE, AutoencoderKL of PixArt, FLUX VAE) and
text encoders (Gemma, T5, CLIP) are spatial-/sequence-adaptive by
construction: the same compiled graph runs at any reasonable height/width
or sequence length. The build-time graph capture records one representative
size (`trace_seq_len = 23` for LLM seq, `64` or `128` for spatial), then
marks the runtime-variable dims as symbolic. CompiledSequence's symbol
binding pass walks every op at runtime and substitutes the actual size.

Without symbolic shapes, every aspect ratio / resolution / max_token would
require a separate compiled graph — combinatorial explosion.

With symbolic shapes correctly honored, **one** graph serves the whole
runtime envelope, and tile-only mechanisms (TilingEngine) only kick in for
graphs that genuinely cannot adapt (fixed-grid models like Swin2SR).

## Where symbolic shapes appear

| Component family | Typical symbolic dims | Bound at |
|---|---|---|
| Image VAE (decode) | spatial H, W (and possibly batch) | post-loop dispatch |
| Image transformer (DiT) | batch, spatial seq_len, sometimes channels | per-step dispatch |
| Text encoder | batch, seq_len | once per request |
| LLM transformer (autoregressive) | batch, seq_len (KV cache growth) | per token |
| Audio encoder/decoder | batch, time | per request |

Any 4D `[B, C, H, W]` or 5D `[B, C, T, H, W]` tensor with a `symbolic_shape`
where indices ≥ 2 are symbolic is **spatial-adaptive**: never tile, never
allocate based on build-time concrete size, never make kernel-selection
decisions on the concrete spatial value alone.

## Sites that already honor the contract

- `core/runtime/graph/compiled_sequence.py` — symbol binding pass (Mar 2026)
- `core/runtime/graph_executor.py::_compute_computable_buffers()` —
  recomputes runtime buffers like `sincos_2d_pos_embed` based on bound
  symbols. *Coverage gap*: only handles a hardcoded list of buffer name
  patterns; doesn't yet cover all spatial-dependent constants. See
  `docs/follow-ups/layer8-computable-buffers-extension.md`.
- `core/module/tiling_engine.py::from_component_config()` — refuses to
  instantiate a TilingEngine when any spatial dim is symbolic
  (Layer 6.3, Apr 2026).

## Sites still to audit (audit grep, work as it surfaces)

- `triton/sequence.py` — kernel dispatch / arena slot sizing
- `core/runtime/executor.py` — input synthesis paths, buffer allocation
- `kernels/wrappers.py` — block-size pickers (most already shape-driven
  per-call, but worth a sweep)

The audit pattern: grep for `tensor["shape"]`, `tensor.get("shape")`,
`graph["..."]["shape"]` reads that **drive a runtime decision**. For each
hit, classify A (benign — diagnostic only), B (decision-bearing, may need
symbolic_shape consultation), or C (mismatch build-time/runtime — fix).
Type A hits are noise; type B hits should at least add a comment that the
decision is shape-concrete-driven and confirm that's intended; type C hits
are Layer-6-style bugs.

## Implementation note

The lazy way to consult symbolic_shape is just to walk
`tensor["symbolic_shape"]["dims"]` and check `isinstance(dim, dict) and
dim.get("type") == "symbol"` on the dims that matter for your decision. See
`tiling_engine.py:from_component_config` for the canonical pattern.
