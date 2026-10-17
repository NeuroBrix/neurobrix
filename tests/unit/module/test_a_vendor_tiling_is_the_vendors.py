"""The engine's vendor tilings (`core/module/vendor_tiling.py`) compute what the vendor's own tiled paths compute.

The brick reproduces two vendor algorithms from declared numbers: diffusers' AutoencoderKL-family `tiled_decode`
(`ramp_crop`: CogVideoX's, DC-AE's) and Allegro's `tiled_decode` (`mask_sum`). Run here against the VENDOR's
method on the vendor's own model (random weights, float64, the CPU), the tile function being the vendor's own decode
of one tile — so what is compared is the cut and the blend alone, the part the brick owns. The geometry is shrunk
through the vendor's own knobs (`enable_tiling(...)` for CogVideoX; Allegro's `kernel` / `stride` attributes) so a
small latent makes two tiles on each axis.

What each test would do if the code were wrong:
  * a blend weight reversed, a blend over the wrong rows, `blend_h` reading the left tile before its own vertical
    blend (the vendor blends in place), a crop of the wrong size: the seams differ — red, measured in place;
  * a mask ramp on the wrong end, a tile placed at the input stride instead of the output's, the output not zero where
    no tile reaches: red.

Needs the vendor oracle's diffusers (`/home/mlops/venvs/vendor_oracle`, docs/internal/environments.md): skipped by
name elsewhere. A vendor import here is a TEST's, never the engine's (R34).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from neurobrix.core.module.vendor_tiling import mask_sum, ramp_crop  # noqa: E402


def test_ramp_crop_is_cogvideox_tiled_decode():
    from diffusers import AutoencoderKLCogVideoX
    torch.manual_seed(0)
    vae = AutoencoderKLCogVideoX(block_out_channels=(32, 32, 32, 32), latent_channels=4, norm_num_groups=8,
                                 sample_height=128, sample_width=192).double().eval()
    vae.enable_tiling(tile_sample_min_height=64, tile_sample_min_width=96)
    rs = 2 ** (len(vae.config.block_out_channels) - 1)
    z = torch.randn(1, 4, 3, 13, 19, dtype=torch.float64)
    with torch.no_grad():
        ref = vae.tiled_decode(z).sample

        def run_tile(tile):                      # the vendor's own decode of one tile, its frame batches included
            vae.use_tiling = False
            out = vae._decode(tile).sample
            vae.use_tiling = True
            return out

        tl_h, tl_w = vae.tile_latent_min_height, vae.tile_latent_min_width
        decl = {"tile": [tl_h, tl_w],
                "stride": [int(tl_h * (1 - vae.tile_overlap_factor_height)),
                           int(tl_w * (1 - vae.tile_overlap_factor_width))],
                "blend": [int(vae.tile_sample_min_height * vae.tile_overlap_factor_height),
                          int(vae.tile_sample_min_width * vae.tile_overlap_factor_width)],
                "in_axes": [3, 4], "out_axes": [3, 4]}
        decl["crop"] = [vae.tile_sample_min_height - decl["blend"][0], vae.tile_sample_min_width - decl["blend"][1]]
        assert rs == vae.tile_sample_min_height // tl_h
        got = ramp_crop(z, decl, run_tile)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert float((got - ref).abs().max()) <= 1e-12 * float(ref.abs().max())


def test_mask_sum_is_allegro_tiled_decode():
    from diffusers import AutoencoderKLAllegro
    torch.manual_seed(0)
    vae = AutoencoderKLAllegro(block_out_channels=(32, 32, 32, 32), latent_channels=4, norm_num_groups=8,
                               sample_size=64).double().eval()
    rt, rs = vae.config.temporal_compression_ratio, vae.spatial_compression_ratio
    vae.kernel, vae.stride = (8, 64, 64), (4, 40, 48)          # the vendor's own knobs, shrunk: 2 tiles per axis
    vae.enable_tiling()
    z = torch.randn(1, 4, 3, 13, 14, dtype=torch.float64)       # 3 = (12-8)/4+1 frames... (12 // rt = 3 latent)
    with torch.no_grad():
        ref = vae.tiled_decode(z)

        zq = z.permute(0, 2, 1, 3, 4).flatten(0, 1)
        zq = vae.post_quant_conv(zq).unflatten(0, (1, -1)).permute(0, 2, 1, 3, 4)

        def run_tile(tile):                       # the vendor's decoder on one post-quant tile, as tiled_decode runs it
            return vae.decoder(tile)

        decl = {"tile_in": [vae.kernel[0] // rt, vae.kernel[1] // rs, vae.kernel[2] // rs],
                "stride_in": [vae.stride[0] // rt, vae.stride[1] // rs, vae.stride[2] // rs],
                "tile_out": list(vae.kernel), "stride_out": list(vae.stride),
                "in_axes": [2, 3, 4], "out_axes": [2, 3, 4], "out_scale": [[rt, 1], [rs, 1], [rs, 1]]}
        got = mask_sum(zq, decl, run_tile)
    ref = ref.permute(0, 2, 1, 3, 4)                              # the vendor returns [B, T, C, H, W]
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert float((got - ref).abs().max()) <= 1e-12 * float(ref.abs().max())


def test_mask_sum_is_allegro_tiled_encode_with_its_affine_tail_per_tile():
    """The vendor blends its encoder's tiles BEFORE quant_conv (and the latent's mean); the engine's tile graph ends
    with them. quant_conv is affine and the ramps sum to one wherever tiles cover, so the two orders agree there to
    float noise — the one place they could not is outside every tile, which the vendor's documented sizes never leave
    (the tiles cover 88 x 720 x 1280 exactly)."""
    from diffusers import AutoencoderKLAllegro
    torch.manual_seed(0)
    vae = AutoencoderKLAllegro(block_out_channels=(32, 32, 32, 32), latent_channels=4, norm_num_groups=8,
                               sample_size=64).double().eval()
    rt, rs = vae.config.temporal_compression_ratio, vae.spatial_compression_ratio
    vae.kernel, vae.stride = (8, 64, 64), (4, 40, 48)
    vae.enable_tiling()
    x = torch.randn(1, 3, 12, 104, 112, dtype=torch.float64)     # (12-8)/4+1 = 2, (104-64)/40+1 = 2, (112-64)/48+1 = 2
    with torch.no_grad():
        ref = vae.tiled_encode(x)

        def run_tile(tile):                         # the encoder AND its affine tail, as the engine's graph runs it
            h = vae.encoder(tile)
            b = h.shape[0]
            h = vae.quant_conv(h.permute(0, 2, 1, 3, 4).flatten(0, 1)).unflatten(0, (b, -1)).permute(0, 2, 1, 3, 4)
            return h

        decl = {"tile_in": list(vae.kernel), "stride_in": list(vae.stride),
                "tile_out": [vae.kernel[0] // rt, vae.kernel[1] // rs, vae.kernel[2] // rs],
                "stride_out": [vae.stride[0] // rt, vae.stride[1] // rs, vae.stride[2] // rs],
                "in_axes": [2, 3, 4], "out_axes": [2, 3, 4], "out_scale": [[1, rt], [1, rs], [1, rs]]}
        got = mask_sum(x, decl, run_tile)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    # The vendor's ramps are float32 (`torch.arange(n).float() / n`): a head and a tail weight sum to one within a
    # float32 rounding, so quant_conv's bias, counted once by the vendor and once per weight here, differs by that
    # rounding — measured 1.05e-8 of the latent's range in float64, under the float32 resolution (1.2e-7) the
    # engine runs at.
    assert float((got - ref).abs().max()) <= 1.2e-7 * float(ref.abs().max())


def test_ramp_crop_is_autoencoderkl_tiled_decode():
    """The image VAEs (PixArt, Flex): AutoencoderKL's `tiled_decode` is the same cut and blend on rank-4 latents."""
    from diffusers import AutoencoderKL
    torch.manual_seed(0)
    vae = AutoencoderKL(block_out_channels=(32, 32, 32, 32), down_block_types=("DownEncoderBlock2D",) * 4,
                        up_block_types=("UpDecoderBlock2D",) * 4, latent_channels=4, norm_num_groups=8,
                        sample_size=64).double().eval()
    vae.enable_tiling()
    z = torch.randn(1, 4, 13, 19, dtype=torch.float64)            # tiles of 8 latent at stride 6
    with torch.no_grad():
        ref = vae.tiled_decode(z).sample

        def run_tile(tile):
            return vae.decoder(vae.post_quant_conv(tile))

        tl, ts = vae.tile_latent_min_size, vae.tile_sample_min_size
        blend = int(ts * vae.tile_overlap_factor)
        decl = {"tile": [tl, tl], "stride": [int(tl * (1 - vae.tile_overlap_factor))] * 2, "blend": [blend, blend],
                "crop": [ts - blend, ts - blend], "in_axes": [2, 3], "out_axes": [2, 3]}
        got = ramp_crop(z, decl, run_tile)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert float((got - ref).abs().max()) <= 1e-12 * float(ref.abs().max())
