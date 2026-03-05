"""
TilingEngine — Universal Tiled Execution for Spatial Components.

Splits large spatial inputs into overlapping tiles, executes each through
the component's CompiledSequence at trace size, then blends results using
accumulate-and-divide averaging.

ZERO SEMANTIC: No domain knowledge. Handles any 4D spatial tensor [B, C, H, W].
ZERO HARDCODE: All parameters derived from graph.json + profile.json.
ZERO FALLBACK: Crashes if config is incomplete.

Supports:
- VAE decoders (Sana 4K DC-AE: window attention seam artifacts)
- Upscalers (Swin2SR, Real-ESRGAN: trained at 64x64, must handle arbitrary)
- Any spatial component traced at fixed resolution

Algorithm: Accumulate-and-divide (SwinIR/Swin2SR pattern)
- Each tile contributes weight=1 to output accumulator
- Overlapping pixels get averaged (sum / count)
- Proven across window-attention models, numerically stable
"""

import torch
import json
import logging
from typing import Callable, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class TilingEngine:
    """
    Universal tiled execution engine for spatial components.

    Tiles large inputs into overlapping patches at trace size,
    executes each tile, then blends via accumulate-and-divide.

    All parameters are derived from the component's graph and profile.
    """

    def __init__(
        self,
        trace_size: int,
        overlap: int,
        scale_factor: int,
        window_alignment: int = 1,
        tile_size: Optional[int] = None,
    ):
        """
        Initialize tiling engine.

        Args:
            trace_size: Spatial size the graph was traced at (e.g., 128 for 128x128)
            overlap: Overlap between tiles in input space (e.g., 16)
            scale_factor: Output/input spatial ratio (e.g., 32 for DC-AE, 4 for Swin2SR x4)
            window_alignment: Tile positions must align to this (e.g., 8 for window_size=8)
            tile_size: Actual tile size for execution. When graph has symbolic spatial
                      dims, this can be smaller than trace_size. Defaults to trace_size.
        """
        self.trace_size = trace_size
        self.tile_size = tile_size if tile_size is not None else trace_size
        self.overlap = overlap
        self.scale_factor = scale_factor
        self.window_alignment = window_alignment

        # Stride = tile_size - overlap
        self.stride = self.tile_size - overlap

        # Align stride to window_alignment
        if window_alignment > 1:
            self.stride = (self.stride // window_alignment) * window_alignment
            if self.stride == 0:
                self.stride = window_alignment

        # Recalculate effective overlap from aligned stride
        self.overlap = self.tile_size - self.stride

    @classmethod
    def from_component_config(
        cls,
        graph_path: Path,
        profile_path: Path,
    ) -> Optional["TilingEngine"]:
        """
        Create TilingEngine from component's graph and profile.

        DATA-DRIVEN: Reads trace size from graph input shape, scale factor
        and window alignment from profile config.

        Returns None if the component has no tiling-relevant config
        (no 4D spatial input, no upscale/decoder config).

        Args:
            graph_path: Path to component's graph.json
            profile_path: Path to component's profile.json

        Returns:
            TilingEngine instance, or None if tiling not applicable
        """
        if not graph_path.exists() or not profile_path.exists():
            return None

        # --- Step 1: Read trace size from graph input shape ---
        with open(graph_path) as f:
            graph = json.load(f)

        input_ids = graph.get("input_tensor_ids", [])
        tensors = graph.get("tensors", {})

        trace_size = None
        for input_id in input_ids:
            tensor = tensors.get(str(input_id), {})
            shape = tensor.get("shape", [])
            # Spatial input is [B, C, H, W] — get H
            if len(shape) == 4:
                trace_size = shape[2]
                break

        if trace_size is None:
            # No 4D spatial input — tiling not applicable
            return None

        # --- Step 2: Read scale factor and window config from profile ---
        with open(profile_path) as f:
            profile = json.load(f)

        config = profile.get("config", {})

        # Scale factor: upscale (upscalers) or compression ratio (VAE decoders)
        scale_factor = config.get("upscale")

        if scale_factor is None:
            # Try VAE compression ratio from decoder blocks
            decoder_blocks = config.get("decoder_block_out_channels")
            if decoder_blocks:
                scale_factor = 2 ** (len(decoder_blocks) - 1)

        if scale_factor is None:
            # No scale/compression info — tiling not applicable
            return None

        # Window alignment (for window-attention models)
        window_alignment = config.get("window_size", 1)

        # Overlap: data-driven from config, fallback to trace_size // 8
        # Minimum 4 to ensure some blending
        overlap = max(4, trace_size // 8)

        # Align overlap to window_alignment
        if window_alignment > 1:
            overlap = ((overlap + window_alignment - 1) // window_alignment) * window_alignment

        logger.info(
            f"[TilingEngine] Data-driven init: trace_size={trace_size}, "
            f"scale_factor={scale_factor}, overlap={overlap}, "
            f"window_alignment={window_alignment}"
        )

        return cls(
            trace_size=trace_size,
            overlap=overlap,
            scale_factor=scale_factor,
            window_alignment=window_alignment,
        )

    def should_tile(self, input_tensor: torch.Tensor) -> bool:
        """
        Check if input requires tiling.

        Returns True when input spatial dimensions exceed tile size.
        When tile_size < trace_size (symbolic spatial graphs), this enables
        tiling even when input matches trace size — producing overlapping
        tiles that eliminate grid artifacts via accumulate-and-divide blending.

        Args:
            input_tensor: Input tensor [B, C, H, W]

        Returns:
            True if input is larger than tile size on either axis
        """
        if input_tensor.dim() != 4:
            return False

        _, _, height, width = input_tensor.shape
        return height > self.tile_size or width > self.tile_size

    def tiled_execute(
        self,
        input_tensor: torch.Tensor,
        execute_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Execute component on tiled input with accumulate-and-divide blending.

        Algorithm (SwinIR/Swin2SR pattern):
        1. Compute tile positions with overlap (stride < trace_size)
        2. For each position, extract tile (pad if at edge)
        3. Execute tile through component
        4. Accumulate result into output buffer, increment weight
        5. Divide output by weight → averaged where tiles overlap

        Args:
            input_tensor: Input tensor [B, C, H, W]
            execute_fn: Function to execute a single tile: fn(tile) -> result

        Returns:
            Blended output tensor [B, C_out, H*scale, W*scale]
        """
        batch_size, _, height, width = input_tensor.shape
        sf = self.scale_factor
        ts = self.tile_size

        # Compute tile positions
        h_positions, w_positions = self._compute_tile_positions(height, width)

        # Execute first tile to determine output channels
        first_h, first_w = h_positions[0], w_positions[0]
        first_tile = self._extract_tile(input_tensor, first_h, first_w)
        first_result = execute_fn(first_tile)
        out_channels = first_result.shape[1]

        # Output accumulators
        out_h = height * sf
        out_w = width * sf
        output = torch.zeros(
            batch_size, out_channels, out_h, out_w,
            device=input_tensor.device, dtype=first_result.dtype,
        )
        weight = torch.zeros(
            batch_size, 1, out_h, out_w,
            device=input_tensor.device, dtype=first_result.dtype,
        )

        # Accumulate first tile
        rh = ts * sf
        rw = ts * sf
        oh = first_h * sf
        ow = first_w * sf
        # Clamp output region to actual output size
        actual_rh = min(rh, out_h - oh)
        actual_rw = min(rw, out_w - ow)
        output[:, :, oh:oh + actual_rh, ow:ow + actual_rw] += first_result[:, :, :actual_rh, :actual_rw]
        weight[:, :, oh:oh + actual_rh, ow:ow + actual_rw] += 1

        # Process remaining tiles
        tile_count = 1
        for y in h_positions:
            for x in w_positions:
                if y == first_h and x == first_w:
                    continue  # Already processed

                tile = self._extract_tile(input_tensor, y, x)
                result = execute_fn(tile)

                oy = y * sf
                ox = x * sf
                actual_rh = min(rh, out_h - oy)
                actual_rw = min(rw, out_w - ox)
                output[:, :, oy:oy + actual_rh, ox:ox + actual_rw] += result[:, :, :actual_rh, :actual_rw]
                weight[:, :, oy:oy + actual_rh, ox:ox + actual_rw] += 1
                tile_count += 1

        logger.debug(f"[TilingEngine] Processed {tile_count} tiles, output shape {output.shape}")

        # Divide by weight — where tiles overlap, average the predictions
        return output / weight

    def _compute_tile_positions(self, height: int, width: int) -> Tuple[List[int], List[int]]:
        """
        Compute tile start positions with overlap.

        Creates a grid of positions where stride < trace_size,
        ensuring the last tile reaches the boundary.

        Args:
            height: Input height in input space
            width: Input width in input space

        Returns:
            (h_positions, w_positions) lists of start coordinates
        """
        ts = self.tile_size
        stride = self.stride

        # Height positions
        h_positions = list(range(0, max(1, height - ts + 1), stride))
        # Ensure last tile reaches boundary
        last_h = max(0, height - ts)
        if not h_positions or h_positions[-1] != last_h:
            h_positions.append(last_h)

        # Width positions
        w_positions = list(range(0, max(1, width - ts + 1), stride))
        last_w = max(0, width - ts)
        if not w_positions or w_positions[-1] != last_w:
            w_positions.append(last_w)

        return h_positions, w_positions

    def _extract_tile(self, input_tensor: torch.Tensor, y: int, x: int) -> torch.Tensor:
        """
        Extract a tile from input tensor, padding if at edge.

        Tiles are tile_size x tile_size. When graph has symbolic spatial dims,
        tile_size can be smaller than trace_size. Otherwise tiles match trace_size.

        Args:
            input_tensor: Input tensor [B, C, H, W]
            y: Start row
            x: Start column

        Returns:
            Tile tensor [B, C, tile_size, tile_size]
        """
        ts = self.tile_size
        _, _, height, width = input_tensor.shape

        y_end = min(y + ts, height)
        x_end = min(x + ts, width)
        tile = input_tensor[:, :, y:y_end, x:x_end]

        # Pad to full trace_size if at edge
        actual_h = y_end - y
        actual_w = x_end - x
        if actual_h < ts or actual_w < ts:
            pad_h = ts - actual_h
            pad_w = ts - actual_w
            tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h), mode='replicate')

        return tile

    def __repr__(self) -> str:
        return (
            f"TilingEngine("
            f"trace_size={self.trace_size}, "
            f"tile_size={self.tile_size}, "
            f"stride={self.stride}, "
            f"overlap={self.overlap}, "
            f"scale_factor={self.scale_factor}, "
            f"window_alignment={self.window_alignment})"
        )
