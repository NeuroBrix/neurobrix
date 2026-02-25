"""
VAE Component Handler

Handles AutoencoderKL, AutoencoderDC (DC-AE), and other VAE variants.

Responsibilities:
- Apply scaling_factor to latents before decoding (from profile.json)
- Apply OutputProcessor for post-decoding normalization (DC-AE clamp)

ZERO HARDCODE: All values from config, none hardcoded.
"""

from typing import Dict, Any, Optional, List

import torch

from ..base import ComponentHandler, ComponentConfig
from ..registry import register_handler
from neurobrix.core.module.output_processor import OutputProcessor, VAE_CLAMP_REGISTRY


@register_handler("vae")
class VAEComponentHandler(ComponentHandler):
    """
    Handles all VAE variants: AutoencoderKL, AutoencoderDC, etc.

    DATA-DRIVEN:
    - scaling_factor from profile.json config.scaling_factor
    - vae_scale_factor derived from block_out_channels
    """

    # Supported VAE class names
    SUPPORTED_CLASSES = {
        "AutoencoderKL",
        "AutoencoderDC",
        "AutoencoderTiny",
        "AsymmetricAutoencoderKL",
    }

    @classmethod
    def can_handle(cls, class_name: str, component_type: str) -> bool:
        """Check if this handler supports the component."""
        if component_type == "vae":
            return True
        if class_name in cls.SUPPORTED_CLASSES:
            return True
        # Handle class name variations
        class_lower = class_name.lower()
        return "autoencoder" in class_lower or "vae" in class_lower

    def transform_inputs(self, inputs: Dict[str, Any], phase: str) -> Dict[str, Any]:
        """
        Apply scaling_factor before VAE decode in post_loop phase.

        The scaling_factor is read from profile.json, not hardcoded.
        This scales the latents by 1/scaling_factor before VAE decoding.

        Args:
            inputs: Input dictionary
            phase: Execution phase

        Returns:
            Transformed inputs
        """
        if phase != "post_loop":
            return inputs

        # Find the latent tensor key
        latent_key = self._find_latent_key(inputs)
        if not latent_key:
            return inputs

        # Get scaling_factor from config (DATA-DRIVEN)
        scaling_factor = self.config.scaling_factor
        if scaling_factor is None or scaling_factor == 0:
            return inputs

        # Apply scaling: latents / scaling_factor
        latent = inputs[latent_key]
        if isinstance(latent, torch.Tensor):
            inputs[latent_key] = latent / scaling_factor

        return inputs

    def get_latent_scale(self) -> int:
        """
        Get VAE spatial compression factor.

        Derives from block_out_channels: 2^(len(blocks)-1)
        Falls back to class-based defaults only if derivation fails.

        Returns:
            Scale factor (e.g., 8 for AutoencoderKL, 32 for DC-AE)
        """
        # First try derived value from config
        if self.config.vae_scale_factor is not None:
            return self.config.vae_scale_factor

        # Try to derive from block_out_channels
        blocks = self.config.block_out_channels
        if blocks:
            return 2 ** (len(blocks) - 1)

        # ZERO FALLBACK: Crash explicitly if we cannot determine the scale factor
        # Do NOT guess based on class name - that breaks universality
        raise RuntimeError(
            f"ZERO FALLBACK: Cannot determine VAE scale factor for {self.config.class_name}. "
            "Missing both 'vae_scale_factor' and 'block_out_channels' in profile.json. "
            "Model data incomplete. Re-import: neurobrix remove <model> && neurobrix import <org>/<model>"
        )

    def _find_latent_key(self, inputs: Dict[str, Any]) -> Optional[str]:
        """
        Find the latent tensor key in inputs.

        Searches for common latent input names.

        Args:
            inputs: Input dictionary

        Returns:
            Key name or None
        """
        # Common latent input names in priority order
        latent_keys = ["z", "latent", "latents", "hidden_states", "args"]

        for key in latent_keys:
            if key in inputs:
                value = inputs[key]
                if isinstance(value, torch.Tensor) and value.dim() == 4:
                    return key

        # Search for any 4D tensor
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.dim() == 4:
                return key

        return None
