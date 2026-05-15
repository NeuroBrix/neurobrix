# P-S5-RMS_NORM-16G-NUMERICAL — closure 2026-05-13

Sub-chantier opened by the previous session's condition #2 escalation
(commit `bcadad8`) and closed in this session at commit `8af7848`.

## Diagnosis

Per-tiled-op bit-diff between Sana 4Kpx 32g compiled (coherent) and
16g compiled (black) using the existing `_maybe_dump_tid_native` hook
in `compiled_sequence.py` (made memory-safe by replacing
`tensor.detach().float()` with `torch.linalg.vector_norm(flat,
dtype=torch.float32)` — without that fix the dump itself OOMed on 16g
for the 4 GiB chain tensors).

Result: **first divergent op = `aten.silu::1` shape `[1, 4096, 128,
128]` in the transformer**. l2 norm differs by 3.4× (25502 ref vs
86345 on 16g). Head values are 4 orders of magnitude smaller on 16g
(e.g. `-5e-05` vs `-0.246`). Divergence accumulates through ~200
transformer attention `add` ops and reaches NaN at
`aten.permute::10`. The chain wrapper inherits the corrupted state
and produces all-zeros downstream.

**Root cause**: 20 transformer depthwise convolutions
(`block.X.ffn.conv_depth`) tile-flag on 16g (cuDNN workspace
estimator inflates them to ~12 GiB workspace each — known
estimator pessimism for depthwise, where cuDNN actually needs a
tiny workspace). On 32g the threshold is higher and they pass
through the native cuDNN call without tiling. The tiled
`tiled_conv2d_spatial` path on 16g processes depthwise convs
correctly per-band but accumulates a small numerical error per
band; over 20 blocks this amplifies into orders-of-magnitude
divergence.

## Fix (commit `8af7848`)

`solver.py:_detect_op_level_tiling_pairs` skips standalone-conv
tiling for ops whose weight shape is `[out_c, 1, kh, kw]`
(in_channels_per_group == 1 = depthwise signature). R34 conforming
— purely structural, no model-name lookup. Depthwise convs run
native on 16g; their actual cuDNN workspace requirement is small,
so they fit even without tiling.

## Validation

| Cell | Status | Wall | PNG |
|---|---|---|---|
| Sana 4Kpx 32g compiled (anti-régression) | ✓ PASS | 22.9 s | `32g_compiled_antireg_redapple.png` |
| Sana 4Kpx **16g compiled** | ✓ **PASS** | 22.1 s | `16g_compiled_redapple.png` |
| Sana 4Kpx **16g sequential** | ✓ **PASS** | ~3 min | `16g_sequential_redapple.png` |
| Sana 4Kpx 16g triton | ⏳ NOT PASS — striped (separate bug) | 19.4 min | `16g_triton_striped_NOT_PASS.png` |

Anti-régression:
- TinyLlama compiled GPU: PASS (3 coherent tokens)
- Sana 1024 BF16 compiled: PASS coherent red apple

## Triton 16g residual issue (out of scope for this chantier)

`--triton` on 16g produces a striped/garbage image (`16g_triton_
striped_NOT_PASS.png`). This is a SEPARATE bug surface from the
depthwise/numerical issue this chantier fixed. The striped pattern
is the signature of a triton-mode VAE memory-pressure bug,
historically tracked under `P-SANA-4KPX-RUNTIME` lineage. The
depthwise tile-skip applies identically on triton mode (same
solver code path), so the numerical bug fixed here doesn't
re-fire. The triton-specific issue requires its own dedicated
investigation in the triton/ subtree — opened as backlog
`P-TRITON-VAE-16G-STRIPED`.

## Cells unlocked: 2 (16g compiled + 16g sequential)

Brings the P-PRISM-NEVER-REFUSE v2 matrix from 8/16 to **10/16 ✓**.

## Remaining ⏳ cells (out of scope for this chantier):

- 16g triton / triton_sequential → P-TRITON-VAE-16G-STRIPED
- 2×16g × 4 modes → S4 Gap A (independent of this fix; can be
  pursued now since 16g compiled / sequential are unblocked,
  meaning Prism can produce `component_placement` plans that
  assign each component to a 16 GiB GPU)
