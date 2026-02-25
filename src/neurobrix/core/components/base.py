"""
Component Handler Base Classes

Defines the ComponentConfig dataclass and ComponentHandler ABC that all
component-specific handlers must implement.

ZERO HARDCODE: All values come from config, never hardcoded.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import torch


@dataclass
class ComponentConfig:
    """
    Normalized config loaded from profile.json + runtime.json.

    All fields are optional except class_name and component_type.
    This dataclass serves as the single source of truth for component metadata.

    ZERO HARDCODE: Values are loaded from cache, not hardcoded defaults.
    """
    # Required fields
    class_name: str           # e.g., "AutoencoderDC", "Transformer2DModel"
    component_type: str       # "vae", "transformer", "text_encoder", "unet"

    # === VAE-specific fields ===
    scaling_factor: Optional[float] = None      # From profile.json (e.g., 0.41407)
    latent_channels: Optional[int] = None       # From profile.json (e.g., 32 for DC-AE, 4 for KL)
    vae_scale_factor: Optional[int] = None      # Derived: 2^(len(blocks)-1)
    block_out_channels: Optional[List[int]] = None  # From profile.json

    # === Transformer-specific fields ===
    patch_size: Optional[int] = None            # From profile.json (e.g., 2 for PixArt, 1 for Sana)
    sample_size: Optional[int] = None           # From profile.json (e.g., 128, 32)
    interpolation_scale: Optional[float] = None  # From profile.json (e.g., 2.0 for 4K models)
    attention_head_dim: Optional[int] = None    # From profile.json
    num_attention_heads: Optional[int] = None   # From profile.json
    num_layers: Optional[int] = None            # From profile.json

    # === Text encoder-specific fields ===
    hidden_size: Optional[int] = None           # From profile.json (e.g., 4096 for T5-XXL)
    max_position_embeddings: Optional[int] = None
    vocab_size: Optional[int] = None

    # === Runtime attributes (from runtime.json) ===
    state_channels: Optional[int] = None        # From runtime.json attributes
    state_extent_0: Optional[int] = None        # Trace-time latent height
    state_extent_1: Optional[int] = None        # Trace-time latent width

    # === Raw config for extension ===
    raw_profile: Dict[str, Any] = field(default_factory=dict)
    raw_runtime: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from raw_profile or raw_runtime."""
        if key in self.raw_profile.get("config", {}):
            return self.raw_profile["config"][key]
        if key in self.raw_runtime.get("attributes", {}):
            return self.raw_runtime["attributes"][key]
        return default


class ComponentHandler(ABC):
    """
    Abstract base class for component-specific behavior.

    Each component type (VAE, Transformer, TextEncoder) has a handler that:
    - Transforms inputs before execution (e.g., latent scaling)
    - Modifies weights after loading (e.g., pos_embed scaling)
    - Decides when to use special execution (e.g., tiled decoding)
    - Transforms outputs after execution

    ZERO HARDCODE: All behavior is driven by ComponentConfig.
    """

    def __init__(self, config: ComponentConfig, cache_path: str):
        """
        Initialize handler with config.

        Args:
            config: Normalized component configuration
            cache_path: Path to extracted NBX cache
        """
        self.config = config
        self.cache_path = cache_path
        # Runtime state (set by executor before execution)
        self._runtime_height: Optional[int] = None
        self._runtime_width: Optional[int] = None

    @classmethod
    @abstractmethod
    def can_handle(cls, class_name: str, component_type: str) -> bool:
        """
        Check if this handler supports the given component.

        Args:
            class_name: Component class name (e.g., "AutoencoderDC")
            component_type: Component type (e.g., "vae")

        Returns:
            True if this handler can handle the component
        """
        pass

    # =========================================================================
    # Runtime Resolution (called by executor)
    # =========================================================================

    def set_runtime_resolution(self, height: int, width: int) -> None:
        """
        Set runtime resolution for dynamic calculations.

        Called by RuntimeExecutor before execution to inform handler
        of the current generation resolution.

        Args:
            height: Runtime pixel height
            width: Runtime pixel width
        """
        self._runtime_height = height
        self._runtime_width = width

    # =========================================================================
    # Pre-execution Hooks
    # =========================================================================

    def transform_inputs(self, inputs: Dict[str, Any], phase: str) -> Dict[str, Any]:
        """
        Transform inputs before execution.

        Override in subclass for component-specific input transformations.
        For example, VAE applies scaling_factor to latents in post_loop phase.

        Args:
            inputs: Input dictionary
            phase: Execution phase ("pre_loop", "loop", "post_loop")

        Returns:
            Transformed inputs dictionary
        """
        return inputs

    def prepare_weights(
        self,
        weights: Dict[str, torch.Tensor],
        runtime_height: int,
        runtime_width: int
    ) -> Dict[str, torch.Tensor]:
        """
        Modify weights after loading.

        Override in subclass for component-specific weight modifications.
        For example, Transformer scales pos_embed for resolution mismatch.

        Args:
            weights: Loaded weight tensors
            runtime_height: Runtime pixel height
            runtime_width: Runtime pixel width

        Returns:
            Modified weights dictionary
        """
        return weights

    # =========================================================================
    # Metadata Getters (DATA-DRIVEN)
    # =========================================================================

    def get_latent_scale(self) -> int:
        """
        Get VAE spatial compression factor.

        Returns the factor by which the VAE compresses spatial dimensions.
        Derived from block_out_channels: 2^(len(blocks)-1)

        Returns:
            Scale factor (e.g., 8 for AutoencoderKL, 32 for DC-AE)
        """
        if self.config.vae_scale_factor is not None:
            return self.config.vae_scale_factor

        # Derive from block_out_channels
        blocks = self.config.block_out_channels
        if blocks:
            return 2 ** (len(blocks) - 1)

        # ZERO FALLBACK: Should not reach here if config is complete
        raise RuntimeError(
            f"ZERO FALLBACK: Cannot determine latent scale for {self.config.class_name}. "
            f"Missing block_out_channels in profile.json"
        )
