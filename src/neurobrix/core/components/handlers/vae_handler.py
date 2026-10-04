"""
VAE Component Handler

Handles AutoencoderKL, AutoencoderDC (DC-AE), and other VAE variants.

Responsibilities:
- Apply scaling_factor to latents before decoding (from profile.json)
- Apply OutputProcessor for post-decoding normalization (DC-AE clamp)

ZERO HARDCODE: All values from config, none hardcoded.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # R33: the ATen branch imports it; shared code only annotates
    import torch
from neurobrix.core.runtime.tensor_compat import is_torch_tensor

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
        Apply scaling_factor before VAE decode: latents = latents / scaling_factor
        (profile.json config; 1.0 or absent = untouched).

        The per-channel latent statistics are the flow's single step
        (`resolution.latent_statistics`), never re-applied here.

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

        # Seam diagnostic (NBX_DEBUG): the latent as it enters and leaves this
        # handler. It arrives already in the decoder's space when the VAE
        # declares statistics (the flow's affine); here only scaling_factor
        # may change it, so "in" and "out" differ only by that factor — a
        # difference beyond it means a second owner of the affine is back.
        import os as _os
        _dbg = _os.environ.get("NBX_DEBUG") == "1"
        if _dbg and is_torch_tensor(latent):
            print(f"[VAE-SEAM] in   key={latent_key} shape={list(latent.shape)} "
                  f"mean={latent.float().mean().item():.4f} std={latent.float().std().item():.4f}")
        _dump = _os.environ.get("NBX_DUMP_LATENT")
        if _dump and is_torch_tensor(latent):
            import torch
            torch.save(latent.detach().cpu(), _dump)
            print(f"[VAE-SEAM] dumped the decoder-space latent (after the flow's affine) -> {_dump}")

        # The per-channel statistics (latents_mean / latents_std) are NOT applied
        # here. They belong to the flow, which maps the loop output into the
        # decoder's space once, in both engines, before the post-loop decode
        # (`resolution.latent_statistics` via `_apply_latent_affine`). This
        # handler applied them a second time: every container declaring them
        # (mochi-1-preview, the Wan 2.1 / 2.2 VAEs, SANA-Video) decoded
        # `(x*std + mean)*std + mean` — measured 2026-10-04 on mochi's first
        # decode tile (conv_in l2 534.777 = the vendor's conv_in on the
        # twice-mapped latent 534.765; once-mapped 539.601) and on Wan2.1-T2V
        # (the latent entering here already equal to final*std + mean, to 0.0).

        # scaling_factor (DATA-DRIVEN)
        scaling_factor = self.config.scaling_factor
        if scaling_factor is not None and scaling_factor != 0 and scaling_factor != 1.0:
            latent = latent / scaling_factor

        if _dbg and is_torch_tensor(latent):
            print(f"[VAE-SEAM] out  shape={list(latent.shape)} "
                  f"mean={latent.float().mean().item():.4f} std={latent.float().std().item():.4f} "
                  f"(scaling_factor={self.config.scaling_factor})")

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
                if is_torch_tensor(value) and value.dim() in (4, 5):
                    return key

        # Search for any 4D/5D tensor
        for key, value in inputs.items():
            if _is_tensor(value) and value.dim() in (4, 5):
                return key

        return None
