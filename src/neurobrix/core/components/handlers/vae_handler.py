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
from neurobrix.core.runtime.tensor_compat import is_tensor as _is_tensor


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
        Apply latent denormalization and scaling_factor before VAE decode.

        Two data-driven steps (from profile.json):
        1. Per-channel denormalization: latents = latents / latents_std + latents_mean
           (only if latents_mean/latents_std exist in profile config)
        2. Scaling: latents = latents / scaling_factor

        Args:
            inputs: Input dictionary
            phase: Execution phase

        Returns:
            Transformed inputs
        """
        if phase != "post_loop":
            return inputs

        # Find the latent tensor key (4D or 5D for video)
        latent_key = self._find_latent_key(inputs)
        if not latent_key:
            return inputs

        latent = inputs[latent_key]
        if not _is_tensor(latent):
            return inputs

        # Step 1: Per-channel latent denormalization (DATA-DRIVEN)
        # Some VAEs (LTX2Video, etc.) train with normalized latent space.
        # The pipeline must denormalize before decoding.
        latents_mean = self.config.get("latents_mean")
        latents_std = self.config.get("latents_std")
        if latents_mean is not None and latents_std is not None:
            view_shape = [1, -1] + [1] * (latent.dim() - 2)
            if isinstance(latent, torch.Tensor):
                mean_t = torch.tensor(latents_mean, device=latent.device, dtype=latent.dtype)
                std_t = torch.tensor(latents_std, device=latent.device, dtype=latent.dtype)
                mean_t = mean_t.view(*view_shape)
                std_inv = (1.0 / std_t).view(*view_shape)
            else:
                # NBXTensor path
                import numpy as np
                from neurobrix.kernels.nbx_tensor import NBXTensor
                mean_arr = np.asarray(latents_mean, dtype=np.float32)
                std_arr = np.asarray(1.0 / np.asarray(latents_std, dtype=np.float32), dtype=np.float32)
                mean_t = NBXTensor.from_numpy(mean_arr)
                std_inv = NBXTensor.from_numpy(std_arr)
                if mean_t.nbx_dtype != latent.nbx_dtype:
                    mean_t = mean_t.to(latent.nbx_dtype)
                    std_inv = std_inv.to(latent.nbx_dtype)
                mean_t = mean_t.view(*view_shape)
                std_inv = std_inv.view(*view_shape)
            latent = latent / std_inv + mean_t

        # Step 2: scaling_factor (DATA-DRIVEN)
        scaling_factor = self.config.scaling_factor
        if scaling_factor is not None and scaling_factor != 0 and scaling_factor != 1.0:
            latent = latent / scaling_factor

        inputs[latent_key] = latent
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

        Searches for common latent input names. Supports 4D (image) and 5D (video).

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
                if isinstance(value, torch.Tensor) and value.dim() in (4, 5):
                    return key

        # Search for any 4D/5D tensor
        for key, value in inputs.items():
            if _is_tensor(value) and value.dim() in (4, 5):
                return key

        return None
