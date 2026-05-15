# POINT 6 H2 — non-contiguous NBX slice fix in tiled conv backends
## P-SANA-4KPX-RUNTIME / phase 6 — RED APPLE achieved

R29 inspectable artefact for POINT 6.

## ÉTAPE A — H2 audit identifies residual divergence cluster

H2 audit script (ops 480..737, both modes) captured the first significant
post-POINT-5 divergence:

| op_idx | op_uid | rel | fp_max_d |
|---|---|---|---|
| 649 | aten.add::69 | 0.0130 | 5.18 |
| 667 | aten.convolution::55 | **0.45** | 67 |
| 676 | aten.convolution::57 | **0.90** | 137 |

conv::55 / conv::57 sit at the upsample-fusion site (idx 666
upsample_nearest2d::3 → idx 667 conv::55) — both routed by the
TilingEngine through `fused_upsample_conv2d` → `_fused_upsample_conv2d_nbx`
(the POINT 5 halo-fixed path).

## ÉTAPE B — microtest discriminates kernel-bug vs upstream-divergence

`microtest_conv55_fusion.py` (R29 artefact in this directory) captures
the exact `FusionUpsampleProxy` state, weight, and bias at conv::55 in
both modes, then runs `_fused_upsample_conv2d_torch` and
`_fused_upsample_conv2d_nbx` on bit-equal inputs.

**Pre-fix** (tri-capture fp16 pre_input, weight, bias, stride=1, padding=1):

| tile_factor | torch.max_abs | nbx.max_abs | max_abs_diff | rel | frac>1.0 |
|---|---|---|---|---|---|
| 2 | 154.1 | 70.88 | 159.8 | **1.0365** | **94.63%** |
| 4 | 154.1 | 99.31 | 179.3 | **1.1636** | **94.74%** |
| 8 | 154.1 | 86.31 | 166.9 | **1.0830** | **94.18%** |
| 16 | 154.1 | 53.66 | 163.1 | **1.0583** | **93.60%** |

NBX kernel output is **structurally divergent** from torch reference on
bit-equal inputs. Magnitude is half (or less) of torch, 94%+ of elements
differ by >1.0 absolute. Garbage.

## ÉTAPE C — root cause + fix

**Root cause**: `pre_input[:, :, pre_start:pre_end, :]` on an NCHW tensor
produces a **non-contiguous NBX view**. The C dim's stride keeps its
original H-based value while the slice covers a smaller H, so contiguous-
storage assumption breaks. The downstream Triton wrappers
(`upsample_nearest2d_wrapper`, `conv2d_wrapper`, `constant_pad_nd_wrapper`)
use flat 1D indexing that ASSUMES contiguous input — fed a strided view
they read wrong memory addresses and produce garbage. The torch backend
(`_fused_upsample_conv2d_torch`, `_tiled_conv2d_spatial_torch`) always
called `.contiguous()` at the same points; POINT 5's NBX port missed this.

**Fix** (`src/neurobrix/kernels/ops/fused_upsample_conv.py`):

1. `_fused_upsample_conv2d_nbx`: add `.contiguous()` after
   `pre_band = pre_input[:, :, pre_start:pre_end, :]` (line 730) and after
   `up_band = up_band[:, :, up_offset_local:..., :]` (line 738).
2. `_tiled_conv2d_spatial_nbx`: add `.contiguous()` after
   `in_band = input_tensor[:, :, in_clamped_start:in_clamped_end, :]`
   (line 620).

R33 compliance: `NBXTensor.contiguous()` is an NBXTensor method (returns
a fresh `NBXTensor` via `_strided_copy`, a Triton-pure kernel), not the
forbidden `torch.Tensor.contiguous()`. Same pattern as `group_norm_wrapper`
line 2169 in `wrappers.py`. Zero torch import added.

**Post-fix** (same microtest):

| tile_factor | torch.max_abs | nbx.max_abs | max_abs_diff | rel | frac>1.0 |
|---|---|---|---|---|---|
| 2 | 154.1 | 154.1 | 0.125 | **0.0008** | **0** |
| 4 | 154.1 | 154.1 | 0.125 | **0.0008** | **0** |
| 8 | 154.1 | 154.1 | 0.125 | **0.0008** | **0** |
| 16 | 154.1 | 154.1 | 0.125 | **0.0008** | **0** |

max_abs_diff = 0.125 on magnitude 154 = ~one fp16 ULP. **Kernel
bit-equivalent within fp16 ULP across all tile_factors.**

## ÉTAPE D — visual victory + anti-regression

| model | mode | PNG | verdict |
|---|---|---|---|
| **Sana 4Kpx VAE-iso** | triton_sequential | `sana4kpx_vae_iso_post_point6_RED_APPLE.png` | **🍎 RED APPLE** |
| Sana 1024 | triton_sequential | `sana1024_tri_seq_post_point6.png` | ✓ red apple coherent (71s) |
| PixArt-XL | triton_sequential | `pixart_xl_post_point6.png` | ✓ red apple coherent (115s) |

Sana 4Kpx VAE-iso decoded a photorealistic red apple with green stem on
a red gradient background in 80 s wall time. This is the canonical
victory condition for P-SANA-4KPX-RUNTIME and supersedes all prior
"bandes RGB" / "vertical streaks" partial-progress states.

## Cumulative session statut

| | commit | verdict |
|---|---|---|
| POINT 1 | ea8e8e2 | input::z cast |
| POINT 2 | 331c611 | uniform AMP_FP32_OPS cast-back |
| POINT 2bis | 735e76e | registry plumbing |
| POINT 3 | 1810307 | 0 STRUCTURELLE_RACINE dtype |
| POINT 4 | 6224ac4 | relu::15 innocent |
| POINT 4-bis | b9d46ae | cross-variant Scénario 1 |
| POINT 5 | ad9b7a3 | halo bug fix tiled conv NBX backends |
| POINT 6 H2 (a) | dc7c3b7 | `add_inplace_nbx` contiguous guard |
| **POINT 6 H2 (b)** | this commit | **non-contiguous slice → `.contiguous()` in `_fused_upsample_conv2d_nbx` + `_tiled_conv2d_spatial_nbx` — RED APPLE 4Kpx VAE-iso** |

## Scope verdict on P-SANA-4KPX-RUNTIME

**Sana 4Kpx triton_sequential VAE numerical correctness: SOLVED.**
The VAE-iso path (feed sequential-mode latent through triton VAE) now
produces a coherent red apple at full 4096×4096 resolution.

**Remaining (separate chantier P-TRITON-LIVE-WATERMARK-AUDIT, already
named in CLAUDE.md 2026-05-05 update)**: the full Sana 4Kpx pipeline
(text_encoder → transformer 12-20 steps → VAE) still hits the live-
watermark OOM at conv::62 documented in CLAUDE.md — 26 GiB live arena
+ 8 GiB request vs 32 GiB driver-total. That is a memory-management
issue, structurally orthogonal to the numerical correctness chantier
closed by POINT 6.

## Awaiting Hocine arbitrage

Per POINT 6 mandate clause 1: "Pomme rouge Sana 4Kpx atteinte → tu
remontes la victoire". Reporting victory.
