# P-CONTAINER-EMBED-ORPHAN-SCALARS

## Status: **OPEN** — named follow-up (opened 2026-05-15, P-NEUROBRIX-UPSCALERS-V1)

## Problem

Some architectures build a tensor in the forward pass with a
Python-scalar loop counter, e.g. an attention-mask
construction:

```python
img_mask = torch.zeros(...)
for h_ in h_slices:
    for w_ in w_slices:
        img_mask[:, h_, w_, :] = cnt   # cnt: Python int
        cnt += 1
```

These `cnt` scalars are captured in the container graph as
`param::constant_T_*` with **no embedded constant data** and
flagged as parameters. They are not real trained weights, so
the runtime cannot resolve them from the weight set.

Two NeuroBrix-side mitigations already shipped:

- SwinIR `mean` (U6): the vendored arch registers it as a
  buffer, so it embeds as a genuine graph constant.
- The compiled path now materialises unresolved
  `constant_*`/`[0]`/missing-norm slots as a 0-dim
  `torch.empty` (`bind_weights`, NeuroBrix `c0a1445`),
  matching the sequential resolver.

Both are correct at the **container trace size**: empirically
the uninitialised mask value is numerically negligible there
(SwinIR/HAT reach cos 0.999998 vs the fp32 reference in both
compiled and sequential). It is **not guaranteed** at larger
tiled input sizes, where the shifted-window mask perturbs more
of the field.

## Work

The clean long-term fix is on the model-packaging side: embed
in-forward-loop Python-scalar constants as real container
constant data (carrying their true values), so any future
`mask[slice] = cnt` pattern resolves exactly without relying on
the 0-dim placeholder. This removes the residual approximation
and is a prerequisite for correct arbitrary-size tiling of
shifted-window transformers (relates to **BL-1**, container
`profile.json` missing `upscale`/`window_size`).

## Evidence points

- SwinIR `mean` (U6) — `feedback`/verdict `U6_verdict.md`.
- HAT `cnt` mask loop (U7) — `U7_verdict.md`, compiled
  constant-prepop dead-code analysis.

## Acceptance

Shifted-window transformer upscalers produce bit-faithful
output at arbitrary input sizes (not just the container trace
size), with the placeholder approximation removed.
