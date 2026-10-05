"""
Transformer Component Handler

Handles DiT-style transformers: Transformer2DModel, SanaTransformer2DModel, etc.

Responsibilities:
- Report the connected VAE's spatial scale (data-driven, from the VAE's profile)

A positional table is never resized at load: the graph reads it symbolically, and a computable
table is built by the executor from the graph's own spec.

ZERO HARDCODE: All values from config (patch_size, sample_size, interpolation_scale).
"""
from __future__ import annotations

from typing import Optional

from ..base import ComponentHandler, ComponentConfig
from ..registry import register_handler
from ..config_loader import get_vae_config_for_transformer


@register_handler("transformer")
class TransformerComponentHandler(ComponentHandler):
    """
    Handles DiT-style transformers.

    DATA-DRIVEN:
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
