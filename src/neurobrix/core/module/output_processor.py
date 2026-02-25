"""
Output Processor Module - Data-Driven Post-Processing

Handles VAE output normalization and corrections based on component configuration.
Supports different VAE architectures (AutoencoderKL, AutoencoderDC, etc.) without
hardcoding model-specific logic.

Configuration is read from:
1. Component profile.json (config._class_name)
2. Runtime defaults (output_processing section)

ZERO HARDCODE: All behavior driven by configuration, not model names.
"""

import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from neurobrix.core.runtime.loader import RuntimePackage


@dataclass
class ProcessorConfig:
    """Configuration for output processing."""
    vae_class: str = ""
    clamp_before_normalize: bool = False
    clamp_range: Tuple[float, float] = (-1.0, 1.0)
    normalize: bool = True

    # Future extensibility
    gamma_correction: float = 1.0
    color_space: str = "rgb"


# Registry of VAE classes that require pre-normalization clamping
# Data-driven: can be extended via config without code changes
VAE_CLAMP_REGISTRY: Dict[str, Dict[str, Any]] = {
    "AutoencoderDC": {
        "clamp_before_normalize": True,
        "description": "DC-AE can produce values outside [-1, 1]"
    },
    # Add future VAE types here as needed
}


class OutputProcessor:
    """
    Data-driven output processor for VAE outputs.

    Determines processing steps based on VAE architecture and configuration,
    not model names. This ensures universality and extensibility.
    """

    def __init__(self, config: ProcessorConfig):
        self.config = config

    @classmethod
    def from_package(cls, pkg: "RuntimePackage") -> "OutputProcessor":
        """
        Create processor from RuntimePackage.

        Reads VAE configuration to determine required processing steps.
        ZERO HARDCODE: Uses component config, not model name matching.
        """
        config = ProcessorConfig()

        # Try to load VAE profile to get _class_name
        # Use cache_path (where files are extracted), not root_path (.nbx file)
        try:
            vae_profile_path = pkg.cache_path / "components" / "vae" / "profile.json"
            if vae_profile_path.exists():
                with open(vae_profile_path) as f:
                    vae_profile = json.load(f)

                vae_class = vae_profile.get("config", {}).get("_class_name", "")
                config.vae_class = vae_class

                # Check registry for this VAE class
                if vae_class in VAE_CLAMP_REGISTRY:
                    registry_config = VAE_CLAMP_REGISTRY[vae_class]
                    config.clamp_before_normalize = registry_config.get(
                        "clamp_before_normalize", False
                    )
        except Exception:
            pass  # Silently continue with defaults

        # Check for custom output_processing in defaults
        if hasattr(pkg, 'defaults') and pkg.defaults:
            output_proc = pkg.defaults.get("output_processing", {})
            if output_proc:
                config.clamp_before_normalize = output_proc.get(
                    "clamp_before_normalize",
                    config.clamp_before_normalize
                )
                if "clamp_range" in output_proc:
                    config.clamp_range = tuple(output_proc["clamp_range"])
                config.gamma_correction = output_proc.get(
                    "gamma_correction",
                    config.gamma_correction
                )

        return cls(config)

    @classmethod
    def from_vae_class(cls, vae_class: str) -> "OutputProcessor":
        """
        Create processor directly from VAE class name.
        Useful for testing or when package is not available.
        """
        config = ProcessorConfig(vae_class=vae_class)

        if vae_class in VAE_CLAMP_REGISTRY:
            registry_config = VAE_CLAMP_REGISTRY[vae_class]
            config.clamp_before_normalize = registry_config.get(
                "clamp_before_normalize", False
            )

        return cls(config)

    def process(
        self,
        tensor: torch.Tensor,
        output_range: Tuple[float, float]
    ) -> torch.Tensor:
        """
        Apply all configured processing steps to the tensor.

        Args:
            tensor: Raw VAE output tensor
            output_range: Expected output range [min, max]

        Returns:
            Processed tensor normalized to [0, 1]
        """
        min_val, max_val = output_range

        # Step 1: Pre-normalization clamping (for VAEs that exceed expected range)
        if self.config.clamp_before_normalize:
            tensor = tensor.clamp(min_val, max_val)

        # Step 2: Normalize to [0, 1]
        if self.config.normalize:
            tensor = (tensor - min_val) / (max_val - min_val)

        # Step 3: Gamma correction (if configured)
        if self.config.gamma_correction != 1.0:
            tensor = tensor.pow(1.0 / self.config.gamma_correction)

        return tensor

    def get_info(self) -> Dict[str, Any]:
        """Return processor configuration for debugging."""
        return {
            "vae_class": self.config.vae_class,
            "clamp_before_normalize": self.config.clamp_before_normalize,
            "clamp_range": self.config.clamp_range,
            "normalize": self.config.normalize,
            "gamma_correction": self.config.gamma_correction,
        }


def register_vae_processor(
    vae_class: str,
    clamp_before_normalize: bool = False,
    description: str = ""
) -> None:
    """
    Register a new VAE class with its processing requirements.

    This allows extending the processor without modifying code.
    Can be called from configuration or plugin system.

    Args:
        vae_class: The _class_name from VAE config (e.g., "AutoencoderDC")
        clamp_before_normalize: Whether to clamp before normalizing
        description: Human-readable description of why this is needed
    """
    VAE_CLAMP_REGISTRY[vae_class] = {
        "clamp_before_normalize": clamp_before_normalize,
        "description": description
    }
