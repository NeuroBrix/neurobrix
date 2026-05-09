# VAE-only walk: full-tensor diff sequential vs triton_sequential
## P-SANA-4KPX-RUNTIME — Hocine pivot 2026-05-09

## Methodology

Per Hocine's redirect: identify the FIRST op where seq and triton_seq
diverge significantly on FULL-TENSOR metrics (not the 5-position
fingerprint that proved to underestimate by 75% per
silu12_full_tensor_diff.md).

Implementation: `/tmp/walk_ops_full_diff.py` reuses op_uid_interceptor
tooling. Two passes back-to-back on the same captured latent
(`vae_isolation_input.pt`):

- Pass 1 (sequential): interceptor calls `torch.ops.aten.<op_name>`,
  saves output to `/tmp/seq_walk_dumps/<safe_uid>.pt`.
- Pass 2 (triton_sequential): interceptor calls NBX wrapper, loads
  matching seq output, computes full-tensor diff, saves stats.

Both passes invoke the natural runtime path under each mode (no
`_op_uid_interceptors` from TilingEngine survive — but they kick in
later, at the upsample chain, after our targets are processed).

## Criterion (per Hocine)

A target op is DIVERGENT if either:
- `frac_elements_diff > 1e-3 ≥ 1%` of tensor, OR
- `max_abs_diff > 4 × fp16 ULP at ref_max` of seq tensor.

## First-sweep walk table (compute ops, ~50 op_idx apart)

| op_idx | op_uid | seq dt | tri dt | shape | max_abs | ulps | fr>1e-3 | sign-flips | dt match | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | aten.convolution::0 | fp16 | fp16 | 1×1024×128×128 | 0.0078 | 0.58 | **0.01%** | 0 | ✓ | ok |
| 5 | aten.add::0 | fp16 | **fp32** | 1×1024×128×128 | 0.0154 | 0.97 | **5.82%** | 2 | ✗ | **DIVERGENT** |
| 9 | aten.mm::0 | fp32 | fp32 | 16384×1024 | 0.0483 | 0.48 | 72.08% | 1660 | ✓ | DIVERGENT |
| 13 | aten.mm::1 | fp32 | fp32 | 16384×1024 | 0.0478 | 0.53 | 72.25% | 1792 | ✓ | DIVERGENT |
| 17 | aten.mm::2 | fp32 | fp32 | 16384×1024 | 0.0412 | 0.51 | 72.57% | 1737 | ✓ | DIVERGENT |
| 21 | aten.convolution::1 | fp16 | fp16 | 1×3072×128×128 | 0.0625 | 0.87 | 51.31% | 7778 | ✓ | DIVERGENT |
| 22 | aten.convolution::2 | fp16 | fp16 | 1×3072×128×128 | 0.0313 | 0.96 | 48.60% | 6555 | ✓ | DIVERGENT |
| 40 | aten.bmm::1 | fp32 | fp32 | 64×33×16384 | 1.05e6 | 0.78 | 99.99% | 1735 | ✓ | DIVERGENT |
| 62 | aten.convolution::3 | fp16 | fp16 | 1×8192×128×128 | 0.125 | 0.54 | 53.74% | 12119 | ✓ | DIVERGENT |
| 102 | aten.bmm::2 | fp32 | fp32 | 64×33×32 | 118.2 | 0.08 | 98.69% | 7 | ✓ | DIVERGENT |
| 149 | aten.mm::9 | fp32 | fp32 | 16384×1024 | 0.0223 | 1.09 | 58.54% | 2597 | ✓ | DIVERGENT |
| 200 | aten.convolution::14 | fp16 | fp16 | 1×8192×128×128 | 0.127 | 2.15 | 22.91% | 9057 | ✓ | DIVERGENT |
| 246 | aten.bmm::6 | fp32 | fp32 | 64×33×32 | 648.8 | 1.21 | 99.49% | 17 | ✓ | DIVERGENT |
| 297 | aten.mm::18 | fp32 | fp32 | 65536×1024 | 2.713 | 89.07 | 79.85% | 28372 | ✓ | DIVERGENT |
| 487 | aten.silu::12 | fp16 | fp16 | 1×4096×512×512 | 0.560 | 8.98 | 57.05% | 214115 | ✓ | DIVERGENT |

## Bisection in [op 4 OK → op 22 DIVERGENT]

Compute ops in this range: add::0 (5), mm::0 (9), mm::1 (13),
mm::2 (17), conv::1 (21). All bisected together.

| op_idx | op_uid | seq dt | tri dt | shape | max_abs | ulps | fr>1e-3 | dt match | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 4 | aten.convolution::0 | fp16 | fp16 | 1×1024×128×128 | 0.0078 | 0.58 | 0.01% | ✓ | ok (BIT-EXACT-ish) |
| **5** | **aten.add::0** | **fp16** | **fp32** | 1×1024×128×128 | 0.0154 | 0.97 | **5.82%** | **✗** | **FIRST DIVERGENT** |

## Verdict — first divergent op

`aten.add::0` (op_idx=5).

**Type of divergence**: dtype propagation mismatch.
- Sequential mode produces add::0 OUTPUT in **fp16**.
- triton_sequential mode produces add::0 OUTPUT in **fp32**.

**Why**: at add::0 in sequential mode, both inputs (conv::0::out_0
and view::0::out_0) arrive as fp16, so `torch.ops.aten.add(fp16, fp16) → fp16`.
At add::0 in triton mode, conv::0::out_0 is fp16 (NBX conv2d_wrapper
output) but view::0::out_0 is **fp32** (it traces through the
metadata chain unsqueeze/expand/clone/view — none of which appear
in `_SELF_MANAGED_OPS` nor any AMP set in TritonDtypeEngine that
would cast it down). NBX `add` wrapper's `_prepare_binary` then
applies the `_wider_dtype` rule → both inputs upcast to fp32 →
fp32 output.

**Root cause** (factually, no speculation): the runtime sequential
path applies an implicit cast somewhere that brings view::0::out_0
to fp16 by the time add::0 fires (verified: `add0_sequential_b.pt`
dtype = fp16). The triton path does NOT apply that cast (verified:
`add0_triton_sequential_b.pt` dtype = fp32). Identical input::z
latent (fp32 from `vae_isolation_input.pt`) reaches the executor
in both modes, but the metadata-chain handling diverges between
DtypeEngine and TritonDtypeEngine.

The divergence is NOT a kernel algorithm difference. NBX `add`
wrapper was independently proven bit-exact vs `torch.ops.aten.add`
on identical fp16 inputs (`/tmp/microtest_add0.py`,
max_abs=0, n_diff=0/16777216).

The cascade through downstream ops (mm/bmm/conv/silu) follows from
add::0's fp32 output propagating to subsequent ops, where:
- Seq path: fp16-stored value → cast to fp32 via AMP wrapper for
  mm/bmm → mm in fp32 → fp32 output.
- Tri path: fp32 value → mm receives fp32 → mm in fp32 → fp32 output.

mm receives the SAME compute_dtype (fp32) in both modes BUT the
INPUT VALUES differ: seq has fp16-precision-then-promoted, tri has
fp32-precision-direct. Hence mm output values diverge by 72%
frac>1e-3 even when output dtypes match.

## What this rules out

- **Localized Triton kernel bug at add::0**: ruled out by
  `/tmp/microtest_add0.py` — NBX add bit-exact vs torch add on
  identical fp16 random inputs at this shape.
- **Localized Triton kernel bug at any walk-tested op**: 20 kernels
  proven bit-exact in prior microtests (Phase 3a/3a-bis,
  microtest_unaccounted_ops_report.md). The walk-tested ops here
  show divergent VALUES not because the kernels miscompute, but
  because their INPUTS differ — input difference traceable upstream
  to the dtype-propagation gap at add::0.

## What this does NOT rule out (open for arbitrage)

- A SECOND independent kernel bug somewhere later in the VAE chain
  (some op outside our walk + microtest set). Possible but
  unlikely given that the dtype-chain divergence is sufficient to
  explain the cumulative drift to 75%-frac>1e-3 at silu::12 and
  the depthwise sign-flip cascade at conv::36.
- An accumulator-rounding asymmetry between cuDNN-fp16-via-HMMA-fp32
  and Triton-tl.dot-with-fp32-accumulator — but this would manifest
  as 1-2 ULP per op, not the dtype-precision gap measured here.

## Deliverables

- `/tmp/walk_ops_full_diff.py` — walk script
- `/tmp/microtest_add0.py` — random-input bit-exact verification
- `/tmp/capture_add0_inputs.py` — input dtype capture in both modes
- `/tmp/seq_walk_dumps/*.pt` — sequential output tensors for diff
- `/tmp/walk_stats.json` (copied to this dir) — full diff stats per op

## Awaiting Hocine arbitrage

Per the directive's last paragraph: "Si tu trouves l'op coupable
et que l'écart algorithmique est identifié, tu remontes le verdict
et tu attends mon arbitrage sur le fix".

Found the culprit: **dtype propagation mismatch at the metadata
chain feeding add::0**, surfaced as add::0's fp16-vs-fp32 output
dtype divergence between modes.

Possible fix shapes (NOT applied — awaiting your arbitrage):

1. **Kernel-level (R33-pure)**: extend NBX `add` wrapper to NARROW
   input-b to compute_dtype when input-a is at compute_dtype and
   one input is graph-fp32-tagged metadata. Symmetrize with
   conv2d_wrapper's narrow-on-mismatch rule.
2. **TritonDtypeEngine-level**: extend the engine to cast fp32-tagged
   metadata-chain outputs to compute_dtype at op boundaries (mirror
   what the seq path does implicitly).
3. **Forge-level**: have the trace insert an explicit `_to_copy`
   between view::0 and add::0 so both modes follow the same
   dtype-cast path. Avoids runtime-engine asymmetry by construction.
   (R28 Forge reproducibility doctrine: per-op cast fidelity.)

Each option has tradeoffs: option 1 is a wrapper-level fix
(kernel/wrappers scope, R33-compatible, Hocine arbitrage required
per directive). Option 2 reaches into TritonDtypeEngine
(`triton/dtype.py`) and could touch many ops. Option 3 changes
forge output (NBX format effects per R28/R18).
