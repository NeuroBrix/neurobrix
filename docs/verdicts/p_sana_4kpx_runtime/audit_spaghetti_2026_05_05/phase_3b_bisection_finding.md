# Phase 3b bisection — REFUTED (add::0 is baseline noise, not the bug)

## 2026-05-09 retraction

The verdict below ("first ULP at op_idx=5 add::0 is the bug source")
is **refuted by the cross-variant report**:

`diff_dag_cross_variant_report.md` lists the shape-specific
(rel_4kpx / rel_1024 ≥ 10) divergent add ops as **only** `add::1`,
`add::11`, `add::20`, `add::75`. **`add::0` is NOT in that list** —
it diverges by the same ~1 fp16 ULP at Sana 1024 (where the model
produces a coherent PNG) as it does at 4Kpx.

The Phase 3b same-variant bisection re-introduced the noise floor
that the Phase 1 cross-variant pivot was designed to filter out.
The "first ULP at add::0" is the cuDNN-vs-Triton fp16 noise, common
to both variants — fixing it would not produce a fix.

The first MEANINGFUL (shape-specific) divergence in chronological
trace order is `aten.transpose::2` at full-trace op_idx=67
(rel_4kpx=5.05, rel_1024=0.0002, rel_ratio=20622×, abs_4kpx=30.05).
Within the VAE component the first shape-specific divergence is
`aten.convolution::2` at full-trace line 35048 (rel_ratio=12.6).

Next chantier step: tighter VAE-only bisection with threshold
`rel ≥ 0.5 AND abs ≥ 0.5` to filter the noise floor and identify
the first MEANINGFUL VAE-internal shape-specific divergence on the
VAE-isolation logs (where both modes feed the same captured latent).

The original 2026-05-08 finding follows for the historical record.

---

# (HISTORICAL — REFUTED 2026-05-09) fp16 ULP introduced at op_idx=5 add::0

## Bisection of ops 0..488 in VAE-only seq vs tri trace

| op_idx | op_uid | max_d (positions 0-3) | Note |
|---|---|---|---|
| 0-3 | unsqueeze, expand, clone, view | 0 | metadata, identical |
| 4 | aten.convolution::0 (decoder.conv_in) | 0 | first compute, BIT-EXACT |
| **5** | **aten.add::0 (decoder)** | **0.001** | **first ULP divergence** |
| 6+ | ... | grows | propagates downstream |
| 488 | aten.convolution::36 (depthwise) | 2.43 + sign flip | amplification event |

## op_idx=5 add::0 analysis

Inputs to add::0:
- `aten.convolution::0::out_0` — output of decoder.conv_in
- `aten.view::0::out_0` — view of the saved latent

Both inputs are **bit-exact identical** between seq and tri (verified
in trace at op_idx=3 view and op_idx=4 conv).

Yet add::0 output differs by exactly **1 fp16 ULP** (1/2^10 = 0.001):
```
  seq: [1.33,   -0.4761, 1.06, 1.38, 3.451]
  tri: [1.331,  -0.4762, 1.06, 1.38, 3.451]

True value (computed in fp32): -0.6689 + 1.999 = 1.3301
  seq stores 1.33 (fp16 truncation: 1.3301 -> 1.33 in fp16)
  tri stores 1.331 (closer to true value, may be fp32 storage)
```

## Root cause

**Dtype-management timing difference between sequential and triton
paths at op output storage**:

- Sequential (NativeATenDispatcher → torch ops): the result is cast
  to fp16 (compute_dtype) BEFORE being stored in the result tensor.
  Rounding 1.3301 → 1.33 in fp16.
- Triton (NBX wrappers): NBX `add` may keep the result in fp32 longer,
  storing 1.331 (closer to fp32 true value) before downstream reads
  it as fp16.

OR the reverse — but the bit-pattern divergence at exactly fp16 ULP
points to a per-op dtype-storage difference, not a kernel-arithmetic
bug.

## Why this isn't "distributed drift" architectural

The previous "distributed drift" framing concluded the bug was
unfixable without architectural changes (fp32 intermediates, etc.).

This bisection refutes that: the divergence has a SINGLE precise
introduction point — op_idx=5 add::0 — caused by a localizable
dtype-storage timing discrepancy. The drift then propagates and
amplifies through depthwise conv::36, but the SOURCE is fixable at
add::0.

## Resolution path (focused, NOT architectural)

Audit `wrappers.py:add` at line 758+ to understand its output dtype
storage:
- `_NBX_COMPUTE_DTYPE` global controls compute precision (fp16 on
  Volta for Sana 4Kpx VAE)
- `_matmul_out_dtype` exists for mm/bmm but elementwise ops may not
  have explicit output cast
- The `add_forward_kernel` triton kernel itself stores in whatever
  dtype the output buffer was allocated as

If output buffer is allocated fp32 by NBX's `NBXTensor.empty_like(a)`
where a is fp32 (input from saved latent at op 5), the result stays
fp32 (1.331). If sequential casts to fp16 right after, it gets 1.33.

**Fix: ensure NBX add wrapper output dtype matches what sequential
mode produces** — likely cast to compute_dtype (fp16) at output
storage time. One-line wrapper fix.

This is a localizable, fixable bug — NOT distributed drift.

## What to verify next

Read NBX `add` wrapper code (wrappers.py line 758) to confirm:
1. What dtype does it allocate output as?
2. Does it cast result to compute_dtype (fp16) before storage?
3. Does sequential mode (NativeATenDispatcher) cast differently?

Reading these 30 lines of code should reveal the dtype-storage
discrepancy and suggest the one-line fix.
