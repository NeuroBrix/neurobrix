"""
VAE Tiling Strategy for High-Resolution Decoding.

Implements tiled VAE decode for large latents (4K+) to avoid OOM.
Uses overlapping tiles with linear blending to avoid visible seams.

DATA-DRIVEN:
- Reads VAE trace size from graph.json input shape
- Derives compression_ratio from decoder_block_out_channels
- Detects at runtime if latent > trace_size → apply tiling

ZERO FALLBACK: All values from config, crash if missing.
"""

import torch
import json
import logging
from typing import Dict, Any, Optional, Callable, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class VAETilingStrategy:
    """
    Tiled VAE decode strategy for high-resolution images.

    Splits large latents into overlapping tiles, decodes each tile,
    then blends them together to avoid seams.

    All parameters are derived from the model's traced graph and config.
    """

    def __init__(
        self,
        trace_latent_size: int,
        spatial_compression_ratio: int,
        overlap_pixels: int = 64,
    ):
        """
        Initialize tiling strategy.

        Args:
            trace_latent_size: Size the VAE was traced at (e.g., 32 for 32x32)
            spatial_compression_ratio: VAE compression ratio (e.g., 32 for DC-AE)
            overlap_pixels: Overlap between tiles in pixel space (default 64)
        """
        self.trace_latent_size = trace_latent_size
        self.spatial_compression_ratio = spatial_compression_ratio

        # Compute tile sizes in pixel and latent space
        self.tile_pixel_size = trace_latent_size * spatial_compression_ratio
        self.tile_latent_size = trace_latent_size

        # Overlap in latent space
        self.overlap_latent = overlap_pixels // spatial_compression_ratio
        self.overlap_pixels = self.overlap_latent * spatial_compression_ratio

        # Stride = tile_size - overlap
        self.stride_latent = self.tile_latent_size - self.overlap_latent
        self.stride_pixels = self.tile_pixel_size - self.overlap_pixels

    @classmethod
    def from_vae_graph(cls, vae_graph_path: Path, vae_profile_path: Path) -> "VAETilingStrategy":
        """
        Create tiling strategy from VAE graph and profile.

        DATA-DRIVEN: Reads trace size from graph input, compression from profile.

        Args:
            vae_graph_path: Path to VAE graph.json
            vae_profile_path: Path to VAE profile.json

        Returns:
            VAETilingStrategy instance
        """
        # Read trace latent size from graph input shape
        with open(vae_graph_path) as f:
            graph = json.load(f)

        input_ids = graph.get("input_tensor_ids", [])
        tensors = graph.get("tensors", {})

        trace_latent_size = None
        for input_id in input_ids:
            tensor = tensors.get(str(input_id), {})
            shape = tensor.get("shape", [])
            # VAE input is [B, C, H, W] - get H
            if len(shape) == 4:
                trace_latent_size = shape[2]
                break

        if trace_latent_size is None:
            raise ValueError(
                f"ZERO FALLBACK: Cannot determine trace_latent_size from {vae_graph_path}.\n"
                f"Expected 4D input tensor [B, C, H, W] in graph inputs."
            )

        # Read compression ratio from decoder blocks
        with open(vae_profile_path) as f:
            profile = json.load(f)

        config = profile.get("config", {})
        decoder_blocks = config.get("decoder_block_out_channels")

        if not decoder_blocks:
            raise ValueError(
                f"ZERO FALLBACK: 'decoder_block_out_channels' not found in {vae_profile_path}.\n"
                f"Cannot derive spatial_compression_ratio."
            )

        spatial_compression_ratio = 2 ** (len(decoder_blocks) - 1)

        logger.info(
            f"[VAETiling] Data-driven init: trace_latent={trace_latent_size} "
            f"(from graph), compression={spatial_compression_ratio} "
            f"(from {len(decoder_blocks)} decoder blocks)"
        )

        return cls(
            trace_latent_size=trace_latent_size,
            spatial_compression_ratio=spatial_compression_ratio,
        )

    @classmethod
    def from_model_cache(cls, model_cache_path: Path) -> "VAETilingStrategy":
        """
        Create from model .cache directory.

        Args:
            model_cache_path: Path to ~/.neurobrix/cache/<model_name>/

        Returns:
            VAETilingStrategy instance
        """
        vae_graph = model_cache_path / "components" / "vae" / "graph.json"
        vae_profile = model_cache_path / "components" / "vae" / "profile.json"

        if not vae_graph.exists():
            raise FileNotFoundError(f"VAE graph not found: {vae_graph}")
        if not vae_profile.exists():
            raise FileNotFoundError(f"VAE profile not found: {vae_profile}")

        return cls.from_vae_graph(vae_graph, vae_profile)

    def should_tile(self, latent: torch.Tensor) -> bool:
        """
        Check if latent requires tiling.

        DATA-DRIVEN: Compare latent size with trace size.

        Args:
            latent: Latent tensor [B, C, H, W]

        Returns:
            True if latent is larger than trace size
        """
        if latent.dim() != 4:
            return False

        _, _, height, width = latent.shape
        return height > self.trace_latent_size or width > self.trace_latent_size

    def tiled_decode(
        self,
        latent: torch.Tensor,
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Decode latent using overlapping tiles with blending.

        Uses the exact same algorithm as diffusers AutoencoderDC.tiled_decode:
        1. Split into overlapping tiles (stride < tile_size)
        2. Decode each tile
        3. Blend overlapping regions with linear interpolation
        4. Concatenate results

        Args:
            latent: Latent tensor [B, C, H, W]
            decode_fn: Function to decode a tile: fn(tile) -> decoded_tile

        Returns:
            Decoded image tensor [B, 3, H*ratio, W*ratio]
        """
        batch_size, num_channels, height, width = latent.shape

        tile_h = self.tile_latent_size
        tile_w = self.tile_latent_size
        stride_h = self.stride_latent
        stride_w = self.stride_latent
        blend_h = self.overlap_pixels
        blend_w = self.overlap_pixels

        # Split into overlapping tiles and decode
        rows = []
        tile_count = 0

        for i in range(0, height, stride_h):
            row = []
            for j in range(0, width, stride_w):
                # Extract tile - may extend beyond latent boundary
                i_end = min(i + tile_h, height)
                j_end = min(j + tile_w, width)
                tile = latent[:, :, i:i_end, j:j_end]

                # Pad tile to full size if at edge
                actual_h, actual_w = tile.shape[2], tile.shape[3]
                if actual_h < tile_h or actual_w < tile_w:
                    pad_h = tile_h - actual_h
                    pad_w = tile_w - actual_w
                    # Use 'replicate' mode (not 'reflect') because edge tiles may be
                    # smaller than the required padding (reflect requires pad < input)
                    tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h), mode='replicate')

                # Decode tile
                decoded_tile = decode_fn(tile)

                # Crop padding if we padded
                if actual_h < tile_h or actual_w < tile_w:
                    decoded_h = actual_h * self.spatial_compression_ratio
                    decoded_w = actual_w * self.spatial_compression_ratio
                    decoded_tile = decoded_tile[:, :, :decoded_h, :decoded_w]

                row.append(decoded_tile)
                tile_count += 1

            rows.append(row)

        # Blend tiles together (diffusers algorithm)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # Blend with tile above
                if i > 0:
                    tile = self._blend_v(rows[i - 1][j], tile, blend_h)

                # Blend with tile to the left
                if j > 0:
                    tile = self._blend_h(row[j - 1], tile, blend_w)

                # Take stride portion (non-overlapping part for assembly)
                tile_stride_h = self.stride_pixels
                tile_stride_w = self.stride_pixels

                # Handle last tile in row/column - take full remaining size
                is_last_row = (i == len(rows) - 1)
                is_last_col = (j == len(row) - 1)

                if is_last_row:
                    tile_stride_h = tile.shape[2]
                if is_last_col:
                    tile_stride_w = tile.shape[3]

                cropped = tile[:, :, :tile_stride_h, :tile_stride_w]
                result_row.append(cropped)

            result_rows.append(torch.cat(result_row, dim=3))

        decoded = torch.cat(result_rows, dim=2)

        # Calculate expected size
        expected_h = height * self.spatial_compression_ratio
        expected_w = width * self.spatial_compression_ratio

        # Trim to exact size
        decoded = decoded[:, :, :expected_h, :expected_w]

        return decoded

    def _blend_v(
        self,
        top: torch.Tensor,
        bottom: torch.Tensor,
        blend_extent: int
    ) -> torch.Tensor:
        """
        Blend two tiles vertically (top above bottom).

        Uses linear interpolation in the overlap region.
        Matches diffusers blend_v exactly.
        """
        # Clone to allow in-place updates (inference mode tensors are read-only)
        bottom = bottom.clone()
        blend_extent = min(top.shape[2], bottom.shape[2], blend_extent)
        for y in range(blend_extent):
            weight = y / blend_extent
            bottom[:, :, y, :] = (
                top[:, :, -blend_extent + y, :] * (1 - weight) +
                bottom[:, :, y, :] * weight
            )
        return bottom

    def _blend_h(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        blend_extent: int
    ) -> torch.Tensor:
        """
        Blend two tiles horizontally (left of right).

        Uses linear interpolation in the overlap region.
        Matches diffusers blend_h exactly.
        """
        # Clone to allow in-place updates (inference mode tensors are read-only)
        right = right.clone()
        blend_extent = min(left.shape[3], right.shape[3], blend_extent)
        for x in range(blend_extent):
            weight = x / blend_extent
            right[:, :, :, x] = (
                left[:, :, :, -blend_extent + x] * (1 - weight) +
                right[:, :, :, x] * weight
            )
        return right

    def __repr__(self) -> str:
        return (
            f"VAETilingStrategy("
            f"trace_latent={self.trace_latent_size}, "
            f"compression={self.spatial_compression_ratio}, "
            f"tile={self.tile_pixel_size}px, "
            f"overlap={self.overlap_pixels}px)"
        )
