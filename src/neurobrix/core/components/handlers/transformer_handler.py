"""
Transformer Component Handler

Handles DiT-style transformers: Transformer2DModel, SanaTransformer2DModel, etc.

Responsibilities:
- Scale positional embeddings for resolution mismatch
- Calculate sequence length based on patch_size
- Support sincos recomputation when interpolation_scale is set

ZERO HARDCODE: All values from config (patch_size, sample_size, interpolation_scale).
"""

import math
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F

from ..base import ComponentHandler, ComponentConfig
from ..registry import register_handler
from ..config_loader import get_vae_config_for_transformer


@register_handler("transformer")
class TransformerComponentHandler(ComponentHandler):
    """
    Handles DiT-style transformers.

    DATA-DRIVEN:
    - patch_size from profile.json
    - sample_size from profile.json
    - interpolation_scale from profile.json (for 4K models)
    - VAE scale_factor loaded from connected VAE component
    """

    # Supported transformer class names
    SUPPORTED_CLASSES = {
        "Transformer2DModel",
        "SanaTransformer2DModel",
        "DiTTransformer2DModel",
        "PixArtTransformer2DModel",
    }

    def __init__(self, config: ComponentConfig, cache_path: str):
        """Initialize with config and load VAE config for scale factor."""
        super().__init__(config, cache_path)

        # Load VAE config to get vae_scale_factor
        self._vae_config: Optional[ComponentConfig] = None
        self._vae_scale_factor: Optional[int] = None
        self._load_vae_config()

    def _load_vae_config(self) -> None:
        """Load VAE config from cache to get scale factor."""
        try:
            vae_config = get_vae_config_for_transformer(self.cache_path)
            if vae_config:
                self._vae_config = vae_config
                self._vae_scale_factor = vae_config.vae_scale_factor
        except Exception as e:
            pass  # VAE config not available

    @classmethod
    def can_handle(cls, class_name: str, component_type: str) -> bool:
        """Check if this handler supports the component."""
        if component_type == "transformer":
            return True
        if class_name in cls.SUPPORTED_CLASSES:
            return True
        # Handle class name variations
        class_lower = class_name.lower()
        return "transformer" in class_lower or "dit" in class_lower

    def prepare_weights(
        self,
        weights: Dict[str, torch.Tensor],
        runtime_height: int,
        runtime_width: int
    ) -> Dict[str, torch.Tensor]:
        """
        Scale positional embeddings for resolution mismatch.

        Uses config-driven values:
        - VAE scale factor from connected VAE component
        - patch_size from profile.json
        - interpolation_scale from profile.json (for sincos recomputation)

        Args:
            weights: Loaded weight tensors
            runtime_height: Runtime pixel height
            runtime_width: Runtime pixel width

        Returns:
            Modified weights with scaled pos_embed
        """
        # Find pos_embed weights
        pos_embed_keys = [k for k in weights if 'pos_embed' in k]
        if not pos_embed_keys:
            return weights

        # Get config values (DATA-DRIVEN)
        vae_scale = self._get_vae_scale()
        patch_size = self.config.patch_size or 1
        sample_size = self.config.sample_size or 32
        interp_scale = self.config.interpolation_scale

        # Calculate runtime sequence length
        runtime_latent_h = runtime_height // vae_scale
        runtime_latent_w = runtime_width // vae_scale
        runtime_patches_h = runtime_latent_h // patch_size
        runtime_patches_w = runtime_latent_w // patch_size
        runtime_seq = runtime_patches_h * runtime_patches_w

        for key in pos_embed_keys:
            pos_embed = weights[key]
            if pos_embed.dim() != 3:
                continue

            traced_seq = pos_embed.shape[1]
            if traced_seq == runtime_seq:
                continue  # No scaling needed

            embed_dim = pos_embed.shape[2]
            traced_grid = int(traced_seq ** 0.5)

            # Choose scaling method based on interpolation_scale
            if interp_scale is not None and interp_scale > 0:
                # Use sincos recomputation for models with interpolation_scale
                scaled_embed = self._recompute_sincos_pos_embed(
                    runtime_patches_h,
                    runtime_patches_w,
                    embed_dim,
                    sample_size,
                    interp_scale,
                    pos_embed.device,
                    pos_embed.dtype
                )
            else:
                # Use bilinear interpolation for learned embeddings
                scaled_embed = self._interpolate_pos_embed(
                    pos_embed,
                    traced_grid,
                    traced_grid,
                    runtime_patches_h,
                    runtime_patches_w
                )

            weights[key] = scaled_embed

        return weights

    def get_latent_scale(self) -> int:
        """
        Get VAE spatial compression factor.

        Override of base class method for transformer-specific VAE scale lookup.
        Delegates to _get_vae_scale() which uses VAE config loaded during init.

        Returns:
            Scale factor (e.g., 8 for AutoencoderKL, 32 for DC-AE)
        """
        return self._get_vae_scale()

    def _get_vae_scale(self) -> int:
        """
        Get VAE spatial compression factor.

        DATA-DRIVEN: Uses VAE config loaded during initialization.
        ZERO FALLBACK: Crashes if VAE scale cannot be determined.

        Returns:
            VAE scale factor
        """
        if self._vae_scale_factor is not None:
            return self._vae_scale_factor

        # ZERO FALLBACK: Crash explicitly - don't guess
        raise RuntimeError(
            "ZERO FALLBACK: Cannot determine VAE scale factor for transformer. "
            "Expected 'vae_scale_factor' to be loaded from VAE config. "
            "Ensure VAE component has 'block_out_channels' or 'vae_scale_factor' in profile.json."
        )

    def _recompute_sincos_pos_embed(
        self,
        grid_h: int,
        grid_w: int,
        embed_dim: int,
        base_size: int,
        interpolation_scale: float,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Recompute sincos positional embeddings at runtime resolution.

        This is used for models like Sana that use sincos embeddings.
        The algorithm matches the diffusers implementation.

        Args:
            grid_h: Runtime grid height (patches)
            grid_w: Runtime grid width (patches)
            embed_dim: Embedding dimension
            base_size: Base sample size from config
            interpolation_scale: Scale factor for position normalization
            device: Target device
            dtype: Target dtype

        Returns:
            Recomputed positional embedding [1, seq, embed_dim]
        """
        # Create position grid
        grid_h_positions = torch.arange(grid_h, device=device, dtype=torch.float32)
        grid_w_positions = torch.arange(grid_w, device=device, dtype=torch.float32)

        # Apply interpolation scale (matches diffusers)
        grid_h_positions = grid_h_positions / (grid_h / base_size) / interpolation_scale
        grid_w_positions = grid_w_positions / (grid_w / base_size) / interpolation_scale

        # Create meshgrid
        grid_y, grid_x = torch.meshgrid(grid_h_positions, grid_w_positions, indexing='ij')

        # Flatten to sequence
        positions_h = grid_y.flatten()
        positions_w = grid_x.flatten()

        # Compute sincos embeddings
        half_dim = embed_dim // 2
        emb_h = self._get_1d_sincos_embed(positions_h, half_dim, device)
        emb_w = self._get_1d_sincos_embed(positions_w, half_dim, device)

        # Concatenate [emb_w, emb_h] to match diffusers convention
        # NOTE: Diffusers uses meshgrid(w, h) which swaps order, then calls
        # grid[0] "emb_h" (actually W) and grid[1] "emb_w" (actually H).
        # To match, we use meshgrid(h, w) and swap concatenation order.
        pos_embed = torch.cat([emb_w, emb_h], dim=-1)

        # Add batch dimension and convert dtype
        pos_embed = pos_embed.unsqueeze(0).to(dtype)

        return pos_embed

    def _get_1d_sincos_embed(
        self,
        positions: torch.Tensor,
        dim: int,
        device: torch.device,
        max_period: float = 10000.0
    ) -> torch.Tensor:
        """
        Get 1D sincos embeddings for positions.

        Args:
            positions: Position values [seq]
            dim: Half of embedding dimension
            device: Target device
            max_period: Maximum period for sincos

        Returns:
            Sincos embeddings [seq, dim]
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, device=device, dtype=torch.float32) / half
        )

        args = positions.unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding

    def _interpolate_pos_embed(
        self,
        pos_embed: torch.Tensor,
        traced_h: int,
        traced_w: int,
        runtime_h: int,
        runtime_w: int
    ) -> torch.Tensor:
        """
        Bilinear interpolation for learned positional embeddings.

        Used when no interpolation_scale is set (learned embeddings).

        Args:
            pos_embed: Original embedding [1, seq, dim]
            traced_h: Traced grid height
            traced_w: Traced grid width
            runtime_h: Runtime grid height
            runtime_w: Runtime grid width

        Returns:
            Interpolated embedding [1, new_seq, dim]
        """
        embed_dim = pos_embed.shape[2]
        runtime_seq = runtime_h * runtime_w

        # Reshape to 2D grid: [1, seq, dim] -> [1, dim, h, w]
        pos_2d = pos_embed.squeeze(0).transpose(0, 1).reshape(1, embed_dim, traced_h, traced_w)

        # Bilinear interpolation
        pos_2d_scaled = F.interpolate(
            pos_2d.float(),
            size=(runtime_h, runtime_w),
            mode='bilinear',
            align_corners=False
        )

        # Reshape back to [1, seq, dim]
        scaled_embed = pos_2d_scaled.reshape(embed_dim, runtime_seq).transpose(0, 1).unsqueeze(0)
        scaled_embed = scaled_embed.to(pos_embed.dtype)

        return scaled_embed
