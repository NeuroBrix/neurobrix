# S4 — P-MULTI-GPU-NBX-INTRA-COMPONENT-SPLIT Gap A (closed for compiled/sequential)

## Discovery

Empirically, after the P-S5-RMS_NORM-16G-NUMERICAL fix
(commit `8af7848`) made VAE Sana 4Kpx fit a single 16 GiB GPU,
Prism's cascade naturally picks `single_gpu` on `v100-16g-x2-01`
(2× V100 16 GB) and places all components on `cuda:0`. The 16 GiB
budget is sufficient.

This means S4 Gap A (extending Prism to produce `component_placement`
across cuda:0+cuda:1) **is no longer required for Sana 4Kpx
compiled/sequential**: the single-GPU path on cuda:0 produces a
coherent red apple end-to-end on the 2×16g hardware.

| Cell | Strategy chosen | Wall | Output |
|---|---|---|---|
| Sana 4Kpx 2×16g `--compiled`     | single_gpu (cuda:0) | 23.2 s | ✓ coherent red apple |
| Sana 4Kpx 2×16g `--sequential`   | single_gpu (cuda:0) | ~3 min | ✓ coherent red apple |
| Sana 4Kpx 2×16g `--triton`       | single_gpu (cuda:0) | 51.9 s | ⏳ striped output (P-TRITON-VAE-16G-STRIPED) |
| Sana 4Kpx 2×16g `--triton-sequential` | single_gpu (cuda:0) | — | ⏳ NoneType _dtype error on chain wrapper interceptor (R30 chain wrapper gap on triton path) |

## Verdict

S4 Gap A (multi-device `component_placement`) is **deferred as
unnecessary for this matrix** under the current S5 footprint —
Prism's existing `single_gpu` strategy naturally satisfies the
2×16g cells for compiled and sequential. Forcing cuda:0+cuda:1
distribution would introduce cross-GPU transfer overhead for no
correctness benefit.

The 2 triton cells (`--triton` and `--triton-sequential`) remain
⏳, blocked on:
- triton path striped output (same root cause as 16g triton —
  P-TRITON-VAE-16G-STRIPED)
- triton-sequential `_prepare_binary` NoneType — the unified node
  interceptor for residual chains was designed for compiled mode
  (torch.Tensor) and does not handle NBXTensor args properly in
  triton-sequential mode. R30 gap.

## Cells unlocked by S4: 2 (2×16g compiled + 2×16g sequential)

Matrix: 10/16 → **12/16 ✓**.

Remaining ⏳:
- 16g triton + triton_sequential (P-TRITON-VAE-16G-STRIPED)
- 2×16g triton + triton_sequential (same triton path bug)
