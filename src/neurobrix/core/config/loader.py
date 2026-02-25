# core/config/loader.py
"""
NeuroBrix Configuration Loader.

ZERO HARDCODE: Loads family/vendor configs as source of truth.

Usage:
    from neurobrix.core.config.loader import get_family_config, get_vendor_config

    image_config = get_family_config("image")
    llm_config = get_family_config("llm")
    nvidia_config = get_vendor_config("nvidia", "volta")
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache


# Config root directory
CONFIG_ROOT = Path(__file__).parent.parent.parent / "config"


@lru_cache(maxsize=16)
def get_family_config(family: str) -> Dict[str, Any]:
    """
    Load family-level configuration.

    ZERO HARDCODE: Family defaults (image, llm, audio, video) from config files.

    Args:
        family: Family name (image, llm, audio, video)

    Returns:
        Dict with family configuration

    Raises:
        FileNotFoundError: If family config not found
    """
    config_path = CONFIG_ROOT / "families" / f"{family}.yml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"ZERO FALLBACK: Family config not found: {config_path}\n"
            f"Available families: {list_families()}"
        )

    with open(config_path) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=16)
def get_vendor_config(vendor: str, architecture: str) -> Dict[str, Any]:
    """
    Load vendor/architecture configuration.

    ZERO HARDCODE: Hardware-specific defaults from config files.

    Args:
        vendor: Vendor name (nvidia, amd)
        architecture: Architecture name (volta, ampere, hopper, cdna, cdna2)

    Returns:
        Dict with vendor/architecture configuration

    Raises:
        FileNotFoundError: If vendor/architecture config not found
    """
    config_path = CONFIG_ROOT / "vendors" / vendor / f"{architecture}.yml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"ZERO FALLBACK: Vendor config not found: {config_path}\n"
            f"Available: {list_vendors()}"
        )

    with open(config_path) as f:
        return yaml.safe_load(f)


def list_families() -> list:
    """List available family configurations."""
    families_dir = CONFIG_ROOT / "families"
    if not families_dir.exists():
        return []
    return [p.stem for p in families_dir.glob("*.yml")]


def list_vendors() -> Dict[str, list]:
    """List available vendor/architecture configurations."""
    vendors_dir = CONFIG_ROOT / "vendors"
    if not vendors_dir.exists():
        return {}

    result = {}
    for vendor_dir in vendors_dir.iterdir():
        if vendor_dir.is_dir():
            result[vendor_dir.name] = [p.stem for p in vendor_dir.glob("*.yml")]
    return result


def get_rope_base(family: str = "llm", model_config: Optional[Dict] = None) -> float:
    """
    Get RoPE base value.

    Priority:
    1. Model-specific config (runtime/defaults.json or component config)
    2. Family config (config/families/llm.yml)
    3. Universal default (10000.0)

    Args:
        family: Family name for default lookup
        model_config: Optional model-specific config dict

    Returns:
        RoPE base value (e.g., 10000.0)
    """
    # 1. Check model-specific config
    if model_config:
        rope_theta = model_config.get("rope_theta")
        if rope_theta is not None:
            return float(rope_theta)

    # 2. Check family config
    try:
        family_cfg = get_family_config(family)
        arch = family_cfg.get("architecture", {})
        rope = arch.get("rope", {})
        base = rope.get("base")
        if base is not None:
            return float(base)
    except FileNotFoundError:
        pass

    # 3. Universal default
    return 10000.0


def get_default(family: str, key: str, generation_type: str | None = None) -> Any:
    """
    Get a default value from family config.

    Args:
        family: Family name (image, llm, etc.)
        key: Config key (e.g., "temperature", "num_inference_steps")
        generation_type: Optional generation type (diffusion, autoregressive)

    Returns:
        Default value, or None if not found
    """
    try:
        family_cfg = get_family_config(family)

        # Try generation-type specific defaults first
        if generation_type:
            gen_cfg = family_cfg.get(generation_type, {})
            defaults = gen_cfg.get("defaults", {})
            if key in defaults:
                return defaults[key]

        # Fall back to global defaults
        defaults = family_cfg.get("defaults", {})
        return defaults.get(key)

    except FileNotFoundError:
        return None


# ============================================================================
# Family Config Helpers (consolidated from core/runtime/family_config.py)
# ============================================================================


def get_output_processing(family: str) -> Dict[str, Any]:
    """
    Get output processing config for a family.

    Returns dict with keys like:
    - layout: "CHW", "HWC", "TCHW", "sequence"
    - channel_axis: int
    - batch_axis: int
    - valid_channels: List[int]
    - bit_depth: int
    - output_range: List[float]
    """
    try:
        config = get_family_config(family)
        return config.get("output_processing", {})
    except FileNotFoundError:
        return {}


def get_family_defaults(family: str) -> Dict[str, Any]:
    """
    Get default values for a family.

    Returns dict with keys like:
    - num_inference_steps: int
    - guidance_scale: float
    - height: int
    - width: int
    - negative_prompt: str
    """
    try:
        config = get_family_config(family)
        return config.get("defaults", {})
    except FileNotFoundError:
        return {}


def cascade_default(
    cli_value: Optional[Any],
    defaults: Dict[str, Any],
    defaults_key: str,
    family: str,
    family_key: Optional[str] = None
) -> Any:
    """
    Cascade through value sources silently.

    Resolution order:
    1. CLI value (if not None)
    2. runtime/defaults.json value (if key exists)
    3. Family config value (if key exists)
    4. CRASH (ZERO FALLBACK)

    Args:
        cli_value: Value from command line (may be None)
        defaults: Dict from runtime/defaults.json
        defaults_key: Key to look up in defaults
        family: Family name for family config
        family_key: Key in family config (defaults to defaults_key)

    Returns:
        First non-None value from cascade

    Raises:
        RuntimeError: If all sources are empty
    """
    if family_key is None:
        family_key = defaults_key

    # 1. CLI value
    if cli_value is not None:
        return cli_value

    # 2. runtime/defaults.json
    if defaults_key in defaults and defaults[defaults_key] is not None:
        return defaults[defaults_key]

    # 3. Family config
    family_defaults = get_family_defaults(family)
    if family_key in family_defaults and family_defaults[family_key] is not None:
        return family_defaults[family_key]

    # 4. CRASH
    raise RuntimeError(
        f"ZERO FALLBACK: No value for '{defaults_key}' found.\n"
        f"Checked: CLI arg, runtime/defaults.json, config/families/{family}.yml\n"
        f"Provide value via --{defaults_key.replace('_', '-')} or update configs."
    )


# ============================================================================
# Vendor Config Helpers (consolidated from core/runtime/vendor_config.py)
# ============================================================================


def get_device_prefix(vendor: str, architecture: str) -> str:
    """
    Get device prefix for vendor/architecture.

    Maps:
    - nvidia (any) → "cuda"
    - amd (any) → "cuda" (via HIP/ROCm)

    Returns:
        Device prefix string for torch.device()
    """
    config = get_vendor_config(vendor, architecture)
    return config.get("device_prefix", "cuda")


def get_block_sizes(vendor: str, architecture: str) -> Dict[str, Any]:
    """
    Get kernel block sizes for vendor/architecture.

    Returns dict with keys like:
    - default: int
    - gemm: {block_m, block_n, block_k, num_warps}
    - bmm: {block_m, block_n, block_k}
    - conv2d: {block_h, block_w}
    - softmax_cap: int
    - layernorm_tile: int
    - etc.
    """
    config = get_vendor_config(vendor, architecture)
    return config.get("block_sizes", {})


def get_gemm_config(vendor: str, architecture: str) -> Dict[str, int]:
    """
    Get GEMM-specific block configuration.

    Returns:
        Dict with block_m, block_n, block_k, num_warps, num_stages
    """
    block_sizes = get_block_sizes(vendor, architecture)
    gemm_config = block_sizes.get("gemm", {})

    # Provide defaults if missing (but prefer data-driven)
    return {
        "block_m": gemm_config.get("block_m", 64),
        "block_n": gemm_config.get("block_n", 64),
        "block_k": gemm_config.get("block_k", 32),
        "num_warps": gemm_config.get("num_warps", 4),
        "num_stages": gemm_config.get("num_stages", 2),
    }


def get_bmm_config(vendor: str, architecture: str) -> Dict[str, int]:
    """
    Get Batched Matrix Multiply block configuration.
    """
    block_sizes = get_block_sizes(vendor, architecture)
    bmm_config = block_sizes.get("bmm", {})

    return {
        "block_m": bmm_config.get("block_m", 64),
        "block_n": bmm_config.get("block_n", 64),
        "block_k": bmm_config.get("block_k", 32),
        "num_warps": bmm_config.get("num_warps", 4),
    }


def get_sdpa_block_size(vendor: str, architecture: str, head_dim: int) -> tuple:
    """
    Get SDPA block sizes based on head dimension.

    Args:
        vendor: Vendor name
        architecture: Architecture name
        head_dim: Attention head dimension

    Returns:
        Tuple of (block_m, block_n, num_warps)
    """
    config = get_vendor_config(vendor, architecture)
    thresholds = config.get("sdpa_thresholds", [])

    # Default fallback
    block_m, block_n, num_warps = 64, 64, 4

    for thresh in thresholds:
        if "head_dim_ge" in thresh and head_dim >= thresh["head_dim_ge"]:
            block_m = thresh.get("block_m", block_m)
            block_n = thresh.get("block_n", block_n)
            num_warps = thresh.get("num_warps", num_warps)
            break
        elif "head_dim_lt" in thresh and head_dim < thresh["head_dim_lt"]:
            block_m = thresh.get("block_m", block_m)
            block_n = thresh.get("block_n", block_n)
            num_warps = thresh.get("num_warps", num_warps)
            # Don't break - this is the default case

    return block_m, block_n, num_warps


def get_precision_support(vendor: str, architecture: str) -> Dict[str, bool]:
    """
    Get precision support flags for vendor/architecture.

    Returns:
        Dict with supports_fp16, supports_bf16, supports_tf32, supports_fp8
    """
    config = get_vendor_config(vendor, architecture)
    return config.get("precision", {
        "supports_fp16": True,
        "supports_bf16": False,
        "supports_tf32": False,
        "supports_fp8": False,
    })


def list_available_vendor_configs() -> list:
    """List all available vendor/arch configs."""
    vendors_dir = CONFIG_ROOT / "vendors"
    if not vendors_dir.exists():
        return []

    configs = []
    for vendor_dir in vendors_dir.iterdir():
        if vendor_dir.is_dir():
            for arch_file in vendor_dir.glob("*.yml"):
                configs.append(f"{vendor_dir.name}/{arch_file.stem}")

    return sorted(configs)
