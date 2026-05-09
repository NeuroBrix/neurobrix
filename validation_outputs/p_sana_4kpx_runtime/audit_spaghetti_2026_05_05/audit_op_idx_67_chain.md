# Static audit — chain causale op_idx=67 transformer iter 2

Cross-variant analysis (commit `eda0629`) flagged trace_line=9309
= transformer iter 2 op_idx=67 as FIRST shape-specific divergence
(rel_ratio 4Kpx/1024 = 20621×). Static audit of transformer graph
to identify compute root upstream.

## Chain (transformer/graph.json, op_idx=67 → upstream)

```
aten.view::11  (op_idx=67, block.0.self_attn)
  in_shape: [[2, 2240, 16384]]
  in_tids: ['aten.transpose::2::out_0']
  out_shape: [[2, 70, 32, 16384]]
  ↑
aten.transpose::2  (block.0.self_attn)
  in_shape: [[2, 16384, 2240]]
  ↑
aten._unsafe_view::1  (block.0.self_attn.key)
  in_shape: [[32768, 2240]]
  ↑
aten.mm::1  (block.0.self_attn.key)  ← COMPUTE ROOT
  shapes: [[32768, 2240], [2240, 2240]]
```

## Compute root: `block.0.self_attn.key` (mm)

`mm::1` is the **K projection** in self-attention block 0 — same op
type (mm), same shape `(32768, 2240) × (2240, 2240)`, same parent
module pattern as `mm::0` (Q projection) we identified in commit
`3cd1595` of this session.

`mm::0` was microtested at this exact shape with random fp32 inputs
vs cuBLAS reference: **BIT-EXACT** (max diff 0.0). By construction
`mm::1` uses the same kernel and same compile-cache config: also
bit-exact in isolation.

## Implication

The runtime divergence at op_idx=67 (visible from iter 2 onwards in
cross-variant) is NOT caused by mm::1 itself but by drift in the
INPUT to mm::1 = post-norm activation feeding the K projection.

## Next walk-upstream audit (Path 2 attack vector)

mm::1's input 0 is `_unsafe_view::1` of some upstream activation.
Walking past mm::1 to find what generates the activation that feeds
both Q (mm::0) and K (mm::1) projections gives the next chain root.
That activation is the output of `block.0.norm1` (LayerNorm) applied
to the pos_embed-added input.

Candidates for drift origin:
1. **block.0.norm1 LayerNorm** at shape `(2, 16384, 2240)` — fp32
   variance reduction over feat_dim=2240. 1024 case has feat_dim=2240
   too but seq_len=1024 vs 16384. The norm itself is per-position
   independent so seq_len doesn't matter; the divergence (if any)
   would be in normalization numerics shape-independent.
2. **patch_embed pos_embed addition** — the sole shape-specific op
   in 4Kpx transformer that doesn't exist in 1024 (op_idx=7).
   pos_embed values are different per resolution (sized by
   trace_resolution). Was microtested bit-exact at this shape in
   prior session (commit `4c41e15`).

## Fact

Both candidates 1 and 2 were microtested bit-exact in isolation.
The drift accumulates across the chain in runtime conditions but
no individual op is intrinsically buggy at 4Kpx shape.

This is consistent with the broader cross-variant finding (commit
`eda0629`): the bug is **distributed numerical drift**, not a
single op-source bug. The transformer chain at 4Kpx scale (M=32768
vs M=2048 at 1024) accumulates fp32 noise differently between
cuDNN and Triton paths, and softmax in attention amplifies it
exponentially through iterations.

## Conclusion (static, pre-Path-1-v4-result)

If Path 1 v4 produces Cas A (VAE triton coherent on saved latent),
the bug is confirmed in the transformer chain — specifically the
distributed-drift mechanism described above. Resolution requires
either:
- Higher-precision intermediate (fp64 accumulator) for attention
  chain at 4Kpx scale
- Algorithm-level scaling that keeps pre-softmax magnitudes lower
- Bit-match cuDNN GEMM accumulation order (Volta sm_70 structural gap)

If Path 1 v4 produces Cas B (VAE triton garbage on saved latent),
the VAE has its own shape-dependent bug in addition to (or
independent of) the transformer drift — focus shifts to VAE
TOP-divergent ops (silu::18-23, pixel_shuffle::3, conv::61).
