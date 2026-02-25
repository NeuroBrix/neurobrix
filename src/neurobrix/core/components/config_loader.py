"""
Component Config Loader

Loads and normalizes component configuration from cache files.

Sources:
1. components/{name}/profile.json - Architectural config (from HuggingFace)
2. components/{name}/runtime.json - Runtime attributes (from trace)

ZERO FALLBACK: Raises explicit errors if required files are missing.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base import ComponentConfig


def load_component_config(cache_path: str, component_name: str) -> ComponentConfig:
    """
    Load and normalize config from cache.

    This function reads profile.json and runtime.json for a component
    and creates a normalized ComponentConfig with all relevant fields.

    Args:
        cache_path: Path to extracted NBX cache
        component_name: Name of the component (e.g., "vae", "transformer")

    Returns:
        ComponentConfig with normalized values

    Raises:
        RuntimeError: If profile.json is missing (ZERO FALLBACK)
    """
    cache = Path(cache_path)
    profile_path = cache / "components" / component_name / "profile.json"
    runtime_path = cache / "components" / component_name / "runtime.json"

    # Load profile.json (REQUIRED)
    if not profile_path.exists():
        raise RuntimeError(
            f"ZERO FALLBACK: Missing profile.json for component '{component_name}' "
            f"at {profile_path}. Re-import: neurobrix remove <model> && neurobrix import <org>/<model>"
        )

    with open(profile_path) as f:
        profile = json.load(f)

    # Load runtime.json (OPTIONAL)
    runtime: Dict[str, Any] = {}
    if runtime_path.exists():
        with open(runtime_path) as f:
            runtime = json.load(f)

    # Extract config section from profile
    cfg = profile.get("config", {})
    attrs = runtime.get("attributes", {})

    # Determine component type
    class_name = cfg.get("_class_name", "Unknown")
    comp_type = _infer_component_type(component_name, class_name)

    # Derive VAE scale factor from block_out_channels
    block_channels = _get_block_channels(cfg)
    vae_scale = 2 ** (len(block_channels) - 1) if block_channels else None

    return ComponentConfig(
        # Required
        class_name=class_name,
        component_type=comp_type,

        # VAE fields
        scaling_factor=cfg.get("scaling_factor"),
        latent_channels=cfg.get("latent_channels") or attrs.get("state_channels"),
        vae_scale_factor=vae_scale,
        block_out_channels=block_channels,

        # Transformer fields
        patch_size=cfg.get("patch_size"),
        sample_size=cfg.get("sample_size"),
        interpolation_scale=cfg.get("interpolation_scale"),
        attention_head_dim=cfg.get("attention_head_dim"),
        num_attention_heads=cfg.get("num_attention_heads"),
        num_layers=cfg.get("num_layers"),

        # Text encoder fields
        hidden_size=cfg.get("hidden_size") or cfg.get("d_model"),
        max_position_embeddings=cfg.get("max_position_embeddings"),
        vocab_size=cfg.get("vocab_size"),

        # Runtime attributes
        state_channels=attrs.get("state_channels"),
        state_extent_0=attrs.get("state_extent_0"),
        state_extent_1=attrs.get("state_extent_1"),

        # Raw config for extension
        raw_profile=profile,
        raw_runtime=runtime,
    )


def _infer_component_type(name: str, class_name: str) -> str:
    """
    Infer component type from name or class.

    Uses heuristics based on naming conventions to determine
    the component type when not explicitly specified.

    Args:
        name: Component name (e.g., "vae", "transformer")
        class_name: Class name from config (e.g., "AutoencoderDC")

    Returns:
        Component type string ("vae", "transformer", "text_encoder", "unet", "unknown")
    """
    name_lower = name.lower()
    class_lower = class_name.lower()

    # VAE detection
    if "vae" in name_lower or "autoencoder" in class_lower or "decoder" in class_lower:
        return "vae"

    # Transformer detection
    if "transformer" in class_lower or "dit" in class_lower or "sana" in class_lower:
        return "transformer"

    # Text encoder detection
    if ("encoder" in name_lower and "text" in name_lower) or "t5" in class_lower or "clip" in class_lower:
        return "text_encoder"

    # UNet detection
    if "unet" in class_lower:
        return "unet"

    return "unknown"


def _get_block_channels(cfg: Dict[str, Any]) -> Optional[List[int]]:
    """
    Get block_out_channels from config.

    Handles different naming conventions:
    - block_out_channels (standard VAE)
    - encoder_block_out_channels (DC-AE style)
    - decoder_block_out_channels (DC-AE style)

    Args:
        cfg: Config dictionary from profile.json

    Returns:
        List of channel counts or None
    """
    # Try standard naming
    if "block_out_channels" in cfg:
        return cfg["block_out_channels"]

    # Try DC-AE style (encoder)
    if "encoder_block_out_channels" in cfg:
        return cfg["encoder_block_out_channels"]

    # Try DC-AE style (decoder)
    if "decoder_block_out_channels" in cfg:
        return cfg["decoder_block_out_channels"]

    return None


def get_vae_config_for_transformer(cache_path: str) -> Optional[ComponentConfig]:
    """
    Load VAE config for a transformer component.

    Transformers need to know the VAE's scale factor for proper
    positional embedding calculations.

    Args:
        cache_path: Path to extracted NBX cache

    Returns:
        ComponentConfig for VAE if found, None otherwise
    """
    cache = Path(cache_path)
    components_dir = cache / "components"

    if not components_dir.exists():
        return None

    # Look for VAE component
    for comp_dir in components_dir.iterdir():
        if comp_dir.is_dir():
            name = comp_dir.name.lower()
            if "vae" in name or "decoder" in name:
                try:
                    return load_component_config(str(cache), comp_dir.name)
                except RuntimeError:
                    continue

    return None
