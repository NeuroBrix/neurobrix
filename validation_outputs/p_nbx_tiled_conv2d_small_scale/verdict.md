# P-NBX-TILED-CONV2D-SMALL-SCALE — wrapper math fix (2026-05-14)

## Outcome

The TWO coupled math bugs in the 4 tiled-conv wrappers
(`_tiled_conv2d_spatial_torch`, `_tiled_conv2d_spatial_nbx`,
`_fused_upsample_conv2d_torch`, `_fused_upsample_conv2d_nbx`) are
**identified, fixed, and validated bit-exact via microtest at
1024^2 and 2048^2** (commits `176bc7e` + `63edb03`).

The matrix cell 16g triton remains ⏳ — but for a different reason
than the wrapper bug. The failure mode shifted from "striped/garbage
output" to "OOM at conv::64 live_tracked=12898MB driver_free=208MB",
which matches the live-watermark gap previously documented in the
v2 verdict at conv::62 (26.4 GB on 32g). That is a separate
sub-chantier (P-TRITON-LIVE-WATERMARK-AUDIT).

## Root causes identified

### Bug 1 — edge-band double-padding

```python
pad_top_real = max(0, -in_read_start) + (pad_h if is_top_band else 0)
```

`in_inner_start = oh_start*sh_st - pad_h` means the top band
(oh_start=0) has `in_inner_start = -pad_h`, so `in_read_start = -pad_h`,
so `max(0, -in_read_start) = pad_h`. The extra `+ pad_h` term
double-counted the image-edge padding, padding the top band by
`2*pad_h` zero rows instead of `pad_h`. Same on bot.

### Bug 2 — internal-frontier halo offset

```python
output[:, :, oh_start:oh_start + actual_band_h, :] = conv_band[:, :, :actual_band_h, :]
```

When `halo_top > 0` (internal band frontier), `conv_band[0]` is the
convolution at the halo row, not at the band's first useful output
row. The slice `conv_band[:band_h]` mapped F.conv2d row
`oh_start + halo_top - 1` to output row `oh_start`. Off by halo_top.

## Fix

Both bugs corrected in all 4 wrappers. See commits `176bc7e`,
`63edb03`.

## Validation

`scripts/microtest_tiled_conv2d_small_scale.py` exercises
(spatial ∈ {1024, 2048}, kh ∈ {1, 3, 5}, pad_h ∈ {0, 1, 2},
tile_factor ∈ {1, 2, 4, 8}, with/without bias) on cuda:0 V100-16g.

| Scale | Coverage | Result |
|---|---|---|
| 1024^2 sanity sweep (kh, pad) | 10 configs × 2 tf | 20/20 cos=1.0000 max_abs=0.0000 |
| 1024^2 NBX path full | 4 configs × 4 tf | 16/16 cos=1.0000 max_abs=0.0000 |
| 2048^2 NBX path (in_c=32) | 4 tf | 4/4 cos=1.0000 max_abs=0.0000 |

Production Sana 4Kpx 16g compiled (anti-regression):
22.98s, coherent red apple PNG saved at
`validation_outputs/p_nbx_tiled_conv2d_small_scale/sana_4kpx_16g_compiled_postfix_antiregression.png`.

## Residual blocker for 16g triton

Sana 4Kpx 16g triton post-wrapper-fix:

```
[ERROR] Pipeline failed: Failed at aten.convolution::64 (aten::convolution):
  GPU malloc failed (error 2) for 270532608 bytes
  [device cuda:0 live_tracked=12898MB pool_cached=0MB (0 blocks)
   driver_free=208MB / driver_total=16151MB]
```

This is the live-watermark gap previously diagnosed at conv::62
(26.4 GB on 32g triton, 5.4 GB driver-free, 8 GiB request short by
2.7 GB). The 16g manifestation is at conv::64 with ~270 MB short
of a 12.9 GB live set. Compiled mode runs the same VAE in 22.98s
with a lower live watermark via PyTorch autograd-disabled cleanup
+ caching allocator.

The fix lives in `triton/sequence.py` `kill_slots` lifecycle audit
or `triton/memory_pool.py` deferred-free retention — out of scope
for P-NBX-TILED-CONV2D-SMALL-SCALE.

## Hocine validation: TODO

Visual inspection of the anti-regression PNG: red apple centered,
diffuse (1 inference step is too few to converge — normal). The
wrapper fix produces visually-identical output to pre-fix
production runs because the 1-row-per-tile-conv shift was
visually negligible on real Sana VAE input (random-data microtest
saturated to cos~0 only because random data has no spatial
coherence).

## Relaunch commands

```bash
# Anti-regression compiled (fast, ~23s):
CUDA_VISIBLE_DEVICES=0 neurobrix run --model Sana_1600M_4Kpx_BF16 \
  --prompt "a red apple" --steps 1 --hardware v100-16g --compiled \
  --output sana_4kpx_16g_compiled.png

# 32g triton (anti-regression for the wrapper fix, ~25 min):
CUDA_VISIBLE_DEVICES=2 neurobrix run --model Sana_1600M_4Kpx_BF16 \
  --prompt "a red apple" --steps 1 --hardware v100-32g --triton \
  --output sana_4kpx_32g_triton.png

# 16g triton (blocked on live-watermark gap, ~OOM in <1 min):
CUDA_VISIBLE_DEVICES=0 neurobrix run --model Sana_1600M_4Kpx_BF16 \
  --prompt "a red apple" --steps 1 --hardware v100-16g --triton \
  --output sana_4kpx_16g_triton.png
```
