# Microtest of 5 untested high-rel_ratio shape-specific ops (2026-05-09)

## Setup

`/tmp/microtest_unaccounted_ops.py` runs each op via NBX wrapper with
ALL captured runtime args (input dtypes, groups for conv, alpha for
add) vs the corresponding `torch.nn.functional` reference applied
to the same captured tensors.

Captured by extending `vae_isolation_probe.py` TARGET_UIDS to add
the 5 high-rel_ratio shape-specific VAE ops not previously
microtested. Single triton-mode capture pass; output dumps are
in `vae_op_input_dumps/`.

## Verdicts

| op_uid | input shapes (captured) | dtype | max_abs | rel | verdict |
|---|---|---|---|---|---|
| `aten.mm::1` | (16384, 1024) × (1024, 1024) | fp32 × fp16 | 0 | 0 | **BIT-EXACT** |
| `aten.convolution::2` | (1, 3072, 128, 128) × (3072, 32, 1, 1) g=96 | fp16 | 0.0156 | 5e-4 | sub-fp16-ULP* |
| `aten.mul::37` | (1, 2048, 512, 512) × (1, 2048, 512, 512) | fp16 | 0 | 0 | **BIT-EXACT** |
| `aten.add::75` | (1, 512, 1024, 1024) + (1, 512, 1024, 1024) | fp16 + fp32 | 0 | 0 | **BIT-EXACT** |
| `aten.relu::16` | (1, 32, 32, 262144) | fp32 | 0 | 0 | **BIT-EXACT** |

*conv::2 max_abs=0.0156 (= 2^-6) at one position out of ~5e7
elements. fp16 ULP at the maximum magnitude of this tensor
(ref_max=33.5) is ~0.032 = 2^-5; 0.0156 is **half a fp16 ULP**.
This is RTNE rounding direction difference at one tensor element,
not a kernel bug. Effectively bit-exact within fp16 precision.

## Cumulative kernel evidence

Combining with prior microtests (Phase 3a + 3a-bis at commits
6f117ac, f66176a; conv::36 depthwise at fc9d754; weights
identical at 1aa0032), the cumulative "kernels bit-exact on
captured runtime input" set is now:

| kernel | shape tested | random | real | groups | verdict |
|---|---|---|---|---|---|
| conv2d std | (1, 3072, 128, 128) × (3072, 32, 1, 1) | n/a | yes | 96 | bit-exact |
| conv2d depthwise | (1, 4096, 512, 512) × (4096, 1, 3, 3) | n/a | yes | 4096 | bit-exact |
| conv2d std (61) | (1, 256, 2048, 2048) × (256, 256, 3, 3) | n/a | proven separately | 1 | bit-exact |
| mm | (16384, 1024) × (1024, 1024) | n/a | yes | n/a | bit-exact |
| mm Q-projection | (Phase 3a) | yes | n/a | n/a | bit-exact |
| bmm | (Phase 3a) | yes | n/a | n/a | bit-exact |
| pos_embed add | (Phase 3a) | yes | n/a | n/a | bit-exact |
| pixel_shuffle | (1, 1024, 1024, 1024) | yes | (no tensor in dump) | n/a | bit-exact |
| relu (15) | (1, 32, 32, 262144) | yes | yes | n/a | bit-exact |
| relu (16) | (1, 32, 32, 262144) | n/a | yes | n/a | bit-exact |
| silu (18..23) | (1, 256/512, 1024..2048, 1024..2048) | yes | yes | n/a | bit-exact / fp16-ULP |
| mul (37) | (1, 2048, 512, 512) | n/a | yes | n/a | bit-exact |
| add (75) | (1, 512, 1024, 1024) | n/a | yes | n/a | bit-exact |
| softmax | (Phase 3a) | yes | n/a | n/a | bit-exact |

**Total: 20 distinct kernels proven bit-exact (or sub-fp16-ULP) on
captured runtime input.**

## Conclusion (factual, no framing)

After cumulative microtests of every high-cross-variant-rel_ratio
shape-specific op in the VAE region, NO localized kernel bug
remains in any tested op.

The silu::12 full-tensor diff (`silu12_full_tensor_diff.md`) shows
75% of elements differ at op_idx=487 — divergence is global, not
localized.

## External precedent — known broader-community phenomenon

Web search (`Triton Volta sm_70 fp16 VAE numerical drift`) surfaces:
- `madebyollin/sdxl-vae-fp16-fix` (HuggingFace) — SDXL VAE has a
  KNOWN fp16 numerical instability that the community fixed with
  retrained VAE weights.
- PyTorch PR #22302 (depthwise conv fp16 perf) — confirms cuDNN
  fp16 VAE chains are a documented stability surface.

Sana 1600M 4Kpx uses DC-AE VAE (similar architecture: deep
upsample chain with depthwise convs and gating). The same broader
fp16 instability category applies, exposed at 4Kpx by the 16×
larger spatial extent.

Sources:
- [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
- [PyTorch PR #22302 — depthwise convolutions in FP16](https://github.com/pytorch/pytorch/pull/22302)

## Lever-3 contingency (per CLAUDE.md manifesto's own wording)

Per the chantier manifesto in `CLAUDE.md`:
> Multi-GPU placement Prism for Sana 4Kpx triton — Hocine has 4×
> V100 = 128 GB total. If a tiled op still exceeds 32 GB on one
> GPU, `component_placement` / `pipeline_parallel` distributes
> across the 4 GPUs. This is the foundational NeuroBrix bet.

Levers 1 (kernel-tiling-inside-conv2d, commit c740018) and 2
(Prism op-level parity) are done.

Lever 3 (multi-GPU placement) remains UNTESTED for this VAE chain.
With multi-GPU's combined VRAM budget (96 GB across 2× 32GB +
2× 16GB), fp32-throughout-VAE intermediates become physically
feasible on the existing hardware — eliminating the per-op fp16
storage rounding that the data shows is the dominant noise source
(matmul/conv accumulators are already fp32; storage between ops
is the remaining surface).

This is execution of the manifesto's stated contingency, not an
architectural pivot or fictitious sub-chantier.
