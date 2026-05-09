# silu::12 full tensor diff seq vs tri (Sana 4Kpx VAE-isolation, 2026-05-09)

## Setup

`/tmp/capture_silu12_both_modes.py` builds RuntimeExecutor in BOTH
modes back-to-back, each with the same captured latent
(`vae_isolation_input.pt`), and registers an op_uid_interceptor for
`aten.silu::12` that:
- Saves the input tensor
- Dispatches via the mode-correct path:
  - sequential → `F.silu(x)` (torch.Tensor)
  - triton_sequential → `nbx_silu(x)` (NBXTensor → torch.Tensor for storage)
- Saves the output tensor
- Returns to the pipeline so VAE continues to completion

This gives us silu::12 OUTPUT in both modes (input = conv::35 output).

## Result — divergence is global, not localized

silu::12 output shape `(1, 4096, 512, 512)` fp16 = ~2 GB tensor.

| metric | value |
|---|---|
| max abs diff | 0.5596 |
| mean abs diff | 0.00358 |
| std abs diff | 0.00466 |
| frac elements differing at all | **75.6%** |
| frac elements diff > 1e-3 | **57.0%** |
| frac elements diff > 1e-2 | 6.9% |
| frac elements diff > 1e-1 | 0.01% |
| sign flips | **214,115 elements** |

**Per channel**: 4094 of 4096 channels have max_diff > 0.01.
Top 10 channels: max diff 0.47–0.56, mean 0.004–0.010 per channel.
Channel divergence is roughly UNIFORM across the 4096 channels —
not concentrated in a specific subset.

## Interpretation (factual only)

- silu::12 (op_idx=487 in VAE-only trace) has divergence way beyond
  per-op fp16 ULP: most elements differ, 0.36% mean delta, 200K+
  sign flips.
- Channel-uniform: every channel sees comparable max divergence.
- The fingerprint at silu::12 (5 sample positions out of 1B
  elements) showed only 6 ULPs at one position — drastically
  underestimating actual divergence.

## What this rules out / does not rule out

- Rules out: "single-position fingerprint underestimate is bounded
  to a few ULPs" — actually it's badly out of phase with the full
  tensor reality.
- Does NOT yet rule out: a single upstream kernel with a
  shape-dependent bug that, propagated through attention/conv
  mixing layers, looks channel-uniform by op 487.

## Open microtests required before any architectural framing

Per advisor (and per Hocine's ALWAYS list "microtest avec inputs
RUNTIME capturés"), the following high-cross-variant-rel_ratio ops
in the VAE region have NEVER been microtested on captured input:

| op_uid | full-trace op_idx | rel_ratio | input shape | comment |
|---|---|---|---|---|
| `aten.mm::1` | 13 (VAE-only) | first big amplification | (16384, 1024) × (1024, 1024) | per-position bisection: 134 ULPs at one position |
| `aten.convolution::2` | 22 | 12.6 | (1, 3072, 128, 128) × (3072, 32, 1, 1) | groups=96 grouped 1×1 conv |
| `aten.mul::37` | 559 | 276 | (1, 2048, 512, 512) × (1, 2048, 512, 512) | gating |
| `aten.add::75` | 665 | 504 | (1, 512, 1024, 1024) + (1, 512, 1024, 1024) | residual add |
| `aten.relu::16` | 586 | 580 | unknown | metadata or compute? |

If all bit-exact: factual basis for "no localized kernel bug
remaining" (15 → 20 kernels proven), and lever 3 (multi-GPU
placement per CLAUDE.md manifesto) is the next manifested
contingency — not an architectural pivot but the manifesto's own
stated path.

If any divergent: the localized bug target.
