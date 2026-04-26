# Layer 8 — Computable buffers extension (Sana 4Kpx aten.add broadcast)

**Status**: open — blocks Sana 4Kpx in `--triton` and
`--triton-sequential` after Layer 6 unblocked the TilingEngine
parasitic-activation crash.
**Affected models**: Sana_1600M_4Kpx_BF16 in `--triton` and
`--triton-sequential` on V100. Likely also affects any future image
model whose build-time captured size is smaller than the runtime size
on a graph with non-symbolic spatial buffers.
**Out of scope of**: Layer 6 (architectural build-time + runtime work,
~200+ lines).

## Symptom

After Layer 6.3 (TilingEngine refuses spatial-symbolic graphs), Sana 4Kpx
triton advances past the TilingEngine `torch.zeros(device=NBXTensor)`
TypeError, reaches the transformer body, and crashes:

```
RuntimeError: Failed at aten.add::4 (aten::add):
Cannot broadcast (1, 1024, 64, 64) and (1, 1024, 128, 128)
```

The runtime `current_state` has shape `[1, 1024, 128, 128]` (correct —
Sana 4Kpx latent at 128×128). The other operand is shape
`[1, 1024, 64, 64]` — the build-time concrete shape, **not** rebound
to runtime.

## Root cause

NeuroBrix has TWO complementary mechanisms for shape adaptation:

1. **Symbolic shapes** (`docs/architecture/symbolic-shapes-contract.md`)
   — handle activation tensors via CompiledSequence's symbol binding
   pass. Walks every op at runtime, substitutes symbolic dims with
   concrete runtime values. ✅ Working for Sana DiT activations.

2. **Computable buffers** (`is_computable: true` +
   `computation_spec` in graph.json) — handle constants whose shape
   depends on runtime spatial size. The runtime executor's
   `_compute_computable_buffers()` walks them at startup and at every
   shape rebinding. ⚠️ **Coverage gap**.

The Sana 4Kpx crash is in mechanism (2). One of the operands at
`aten.add::4` is a buffer (almost certainly a position embedding or
spatial bias) that:

- Was captured at build-time with concrete shape `[1, 1024, 64, 64]`.
- Is NOT marked `is_computable: true` in the graph.
- Is NOT recomputed when the runtime symbol-binding sets H/W to 128.
- Stays at the build-time 64×64 → broadcast against the 128×128
  activation → `Cannot broadcast` exception.

Native PyTorch sidesteps this because the buffer is a tensor in the
state dict and its consumers happen to call `interpolate` or
`expand` to match the activation. In triton, every op is dispatched
literally per the graph — there is no implicit interpolation or
broadcast smoothing.

## Why Layer 6 doesn't fix this

Layer 6.3 removes the parasitic TilingEngine activation that masked
this bug. Before Layer 6.3, the TilingEngine ran the VAE in 4 tiles of
64×64 (matching the build-time capture), so the buffer's 64×64 shape
was correct for each tile call and the bug was invisible. With tiling
correctly disabled, the VAE runs once at 128×128 and the static buffer
mismatches.

So Layer 6 transforms an **incorrect-tiling** workaround that "worked"
into a clean dispatch path that surfaces a **real** missing-rebind bug.
This is the right direction — the tiling workaround was hiding the
true cost of the missing computable-buffer coverage.

## Solution outline (Layer 8)

Two complementary changes:

1. **Build-time graph capture**: when capturing a buffer whose shape
   contains a dim that's already symbolic in the consumer chain, mark
   it `is_computable: true` and emit a `computation_spec` that
   reconstructs the buffer from the symbolic dims. Patterns to support:
     - 2D position embeddings parameterized by `(H, W)` — the standard
       sincos formula.
     - Attention bias matrices parameterized by `(seq_q, seq_k)`.
     - Any constant tensor allocated with `torch.zeros(...)` /
       `torch.arange(...)` where the size args trace back to symbolic
       dims.

2. **Runtime executor**: extend
   `core/runtime/graph_executor.py::_compute_computable_buffers()` to
   handle the new computation_spec families above, and to **re-run** at
   every spatial rebind, not only at startup.

The build-time work is the bigger half — the graph-capture layer needs
to understand which buffer-creating ops capture symbolic dims as size
args. The runtime work is straightforward dispatch.

## Validation criteria for Layer 8

- Sana 4Kpx in `--triton` and `--triton-sequential` produce coherent
  images at 4096×4096 (cosine vae.output_0 vs native ≥ 0.95).
- Sana 1024 (3 modes) — zero regression (its build-time capture size =
  runtime size, so symbolic rebind is a no-op; the new computable
  buffers must reconstruct the same value at the same shape).
- LLM harness 14/14 zero regression — LLMs use the same mechanism for KV
  cache positional embeddings (RoPE), and any change to
  `_compute_computable_buffers()` must preserve their behavior bit-perfect.

## Cross-references

- Layer 6 commit (parent of this follow-up): see
  `docs/architecture/symbolic-shapes-contract.md`.
- Layer 7 (`docs/follow-ups/layer7-prism-dtype-override.md`) is the
  parallel architectural blocker for PixArt VAE conv overflow.
- Pre-existing partial implementation of `_compute_computable_buffers()`:
  `sincos_2d_pos_embed` was hoisted into shared `load_weights()` for
  PixArt native compatibility (commit 8ae49dd); this follow-up extends
  the same mechanism to runtime spatial rebinding.
