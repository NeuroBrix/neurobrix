# P-TRITON-IM2COL-KERNEL

## Status: **OPEN** — named follow-up (opened 2026-05-15, P-NEUROBRIX-UPSCALERS-V1)

## Problem

HAT's Overlapping Cross-Attention Block (OCAB) uses
`nn.Unfold`, captured as `aten::im2col`. No Triton kernel
exists for it, so HAT (`hat-s-x4`, `hat-l-x4`) fails in
`triton` and `triton-sequential` modes with a clear
`Missing op: aten::im2col` (R33: no silent torch fallback).
`compiled` and `sequential` modes work (cos 0.999998 vs the
fp32 reference).

## Scope

This is a **Level-1 Triton-pure primitive** (an op the runtime
must support), not a deferrable fused optimisation — so it is
named here rather than closed silently. Per the Triton-pure
2-level doctrine, the chantier that exposed it (U7 HAT) closed
at the highest level reached (compiled + sequential) with this
gap named.

## Work

Write a Triton-pure `im2col` wrapper + `@triton.jit` kernel in
`src/neurobrix/kernels/` honouring the OCAB unfold semantics:

- `kernel_size = overlap_win_size` (`window + int(window*overlap_ratio)`)
- `stride = window_size`
- `padding = (overlap_win_size - window_size) // 2`
- input layout NCHW, output the standard im2col column matrix
  the downstream `view`/`permute` in OCAB expects.

Add to the dispatch table; NBXTensor in/out; zero torch on the
execution path. Numerical-equivalence test vs the compiled
reference at the HAT container trace size, then HAT
triton/triton-seq 4-mode revalidation.

## Acceptance

`hat-s-x4` / `hat-l-x4` reach 4/4 modes with cos ≥ 0.99 vs the
fp32 reference and coherent R29 output. Generalises to any
future unfold-based architecture.
