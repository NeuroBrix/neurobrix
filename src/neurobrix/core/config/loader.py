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

import copy
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


@lru_cache(maxsize=1)
def get_precision_calibration_policy() -> Dict[str, Any]:
    """Load config/precision_calibration.yml (the DtypeEngine fp32-island
    policy: headroom, reference path). ZERO FALLBACK: a missing file raises —
    the engine never guesses how much margin an fp16 store needs."""
    config_path = CONFIG_ROOT / "precision_calibration.yml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"ZERO FALLBACK: precision calibration policy not found: {config_path}"
        )
    with open(config_path) as f:
        policy = yaml.safe_load(f) or {}
    if "headroom_bits" not in policy:
        raise KeyError("ZERO FALLBACK: precision_calibration.yml carries no headroom_bits")
    return policy


@lru_cache(maxsize=1)
def get_tiling_policy() -> Dict[str, Any]:
    """The tiling policy (nbxtensor/ops/tiling.yml: the TilingEngine's engine-wide split sizes, read through its two
    mirror copies `core/module/tiling_sizes.py` and `triton/tiling_sizes.py`). ZERO FALLBACK: a missing file raises — no split is ever cut
    by a size the data does not state."""
    from nbxtensor.ops.tiling_policy import tiling_table      # the one file, nbxtensor's (nbxtensor/ops/tiling.yml)
    return tiling_table()


class UnsupportedArchitectureError(FileNotFoundError):
    """No hardware profile describes the detected GPU.

    Subclasses FileNotFoundError so existing handlers keep working, while
    letting a caller tell "this GPU has no profile" apart from "a config file
    is missing from the install".
    """


# The arch string of a device's model name is nbxtensor's rule (one rule for the library's Metal backend, the census
# and the vendor files keyed by it).
from nbxtensor.backend.device_names import arch_string as device_arch_string  # noqa: E402


def vendor_config_for_device(vendor: str, architecture: Optional[str], compute_capability: Optional[str],
                             model: Optional[str] = None) -> Dict[str, Any]:
    """The vendor file that describes ONE device, by the rule the kernels resolve their active profile with
    (`kernels/ops/_configs.py`, the shared-memory budget): the file whose declared `compute_capability` equals the
    device's, else one whose `compute_capability_matches` prefixes it, else the same major family of a dotted
    capability, else the architecture's file. The Apple M4 Pro reports `apple-m4-pro` and its facts live in the CHIP
    file `apple/apple_m4_pro.yml`, not in the family file `apple_silicon.yml` (the Mac, 2026-10-25 17:20: the time
    estimate found no measured facts there). A device whose detection reports no capability (autodetect writes "0.0"
    for every non-NVIDIA brand; the Mac's generated profile, 17:37) is resolved by the arch string of its MODEL name
    (`device_arch_string`: "Apple M4 Pro" -> "apple-m4-pro"), the string the chip files are keyed by."""
    import yaml
    wanted = str(compute_capability or "").strip().lower()
    by_model = device_arch_string(model)
    exact = family = None
    if wanted or by_model:
        for path in sorted((CONFIG_ROOT / "vendors" / vendor).glob("*.yml")):
            try:
                cfg = _parsed_vendor_file(str(path))   # parsed once per process, never edited (copied below)
            except Exception:  # noqa: BLE001 — an unreadable vendor file is not this device's
                continue
            declared = str(cfg.get("compute_capability", "")).strip().lower()
            if declared and (declared == wanted or declared == by_model):
                exact = cfg
                break
            prefixes = [str(m).strip().lower() for m in (cfg.get("compute_capability_matches") or [])]
            if family is None and (
                    any(w.startswith(m) for m in prefixes for w in (wanted, by_model) if w)
                    or (declared and "." in declared and "." in wanted and declared.split(".")[0] == wanted.split(".")[0])):
                family = cfg
    if exact is not None:
        return copy.deepcopy(exact)          # the parse is shared: a caller never edits it
    if family is not None:
        return copy.deepcopy(family)
    if not architecture:
        raise UnsupportedArchitectureError(f"no vendor file describes {vendor!r} (compute capability {wanted or '?'}) and the device names no architecture")
    return get_vendor_config(vendor, str(architecture))


def get_vendor_config(vendor: str, architecture: str) -> Dict[str, Any]:
    """
    Load vendor/architecture configuration.

    ZERO HARDCODE: Hardware-specific defaults from config files.

    Args:
        vendor: Vendor name (nvidia, amd, apple)
        architecture: Architecture name (volta, ampere, hopper, cdna, cdna2,
            cdna3, apple_silicon)

    Returns:
        Dict with vendor/architecture configuration

    Raises:
        UnsupportedArchitectureError: when no profile describes this GPU. The
            refusal is deliberate — guessing block sizes and memory limits for
            an unknown architecture produces wrong results rather than an
            error. But it must SAY that, which is what this message is for:
            hardware detection can name architectures no profile covers
            (Blackwell, RDNA, Vega, pre-Volta NVIDIA), and users met a bare
            FileNotFoundError with a path in it before running a single op.
    """
    config_path = CONFIG_ROOT / "vendors" / vendor / f"{architecture}.yml"

    if not config_path.exists():
        available = list_vendors()
        known = ", ".join(f"{v}/{a}" for v, arches in sorted(available.items())
                          for a in sorted(arches)) or "none"
        raise UnsupportedArchitectureError(
            f"No hardware profile for {vendor}/{architecture}.\n"
            f"\n"
            f"NeuroBrix reads every hardware-dependent value (block sizes, "
            f"shared-memory limits,\n"
            f"precision support) from a per-architecture profile, and refuses "
            f"to guess them: a\n"
            f"guessed block size produces wrong output, not an error.\n"
            f"\n"
            f"Your GPU was detected as: {vendor} / {architecture}\n"
            f"Profiles available:       {known}\n"
            f"\n"
            f"If your GPU is one NeuroBrix should support, this is a gap on "
            f"our side, not a\n"
            f"limit of your hardware — please open an issue naming the "
            f"architecture above.\n"
            f"Expected profile path:   {config_path}\n"
            f"\n"
            f"What closing this gap needs — one YAML at the path above, with "
            f"values READ FROM\n"
            f"the vendor's own specification, never estimated from a "
            f"neighbouring part:\n"
            f"  architecture, compute_capability   identity, so the profile "
            f"can be resolved\n"
            f"  memory.max_shared_memory_per_block the binding constraint on "
            f"every autotune tile\n"
            f"  memory.max_registers_per_block, memory.warp_size\n"
            f"  block_sizes.{{gemm,bmm,conv2d}}      starting tiles, sized to "
            f"the budget above\n"
            f"  pipelining.max_num_stages          conservative until "
            f"measured on the card\n"
            f"`config/vendors/nvidia/volta.yml` is the worked reference; "
            f"apple_silicon.yml shows\n"
            f"how a profile records that its values are documentation-sourced "
            f"and unmeasured."
        )

    # A copy of the file parsed ONCE per process: the vendor files do not change while an engine runs, and this
    # reader sits under per-op paths (kernels/wrappers._arch_param, read by every attention launch). Parsed at every
    # call it cost whisper-large's Triton decode 4.9 s a token instead of 0.13 (2026-10-26).
    return copy.deepcopy(_parsed_vendor_file(str(config_path)))


def vendor_file(path) -> Dict[str, Any]:
    """The device a vendor file describes, whole: the engine's file at `path` merged with the library's half of the same
    device (nbxtensor/backend/devices) — what a reader of `config/vendors/<vendor>/<name>.yml` means (a copy)."""
    return copy.deepcopy(_parsed_vendor_file(str(path)))


@lru_cache(maxsize=64)
def _parsed_vendor_file(path: str) -> Dict[str, Any]:
    """One vendor file, parsed once per process. Never handed out: callers get a copy (`get_vendor_config`) or one
    leaf value (`vendor_param`), so no caller can alter what the next one reads."""
    with open(path) as f:
        engine_half = yaml.safe_load(f) or {}
    # The device's kernel-relevant half lives in the library (nbxtensor/backend/devices/<vendor>/<name>.yml, API.md
    # section 11): the engine reads the two merged, a key written in both refused by name.
    from nbxtensor.backend import device_files
    p = Path(path)
    return device_files.merge(device_files.read(p.parent.name, p.stem), engine_half, f"{p.parent.name}/{p.stem}: ")


_ABSENT = object()


def vendor_param(vendor: str, architecture: str, section: str, key: str, default):
    """One `<section>.<key>` value of a vendor file, without copying the file — the per-op readers' entry
    (kernels/wrappers._arch_param). `default` when the section or the key is absent; a mapping or list value comes
    back as a copy. Raises like `get_vendor_config` when no file describes the architecture."""
    path = CONFIG_ROOT / "vendors" / vendor / f"{architecture}.yml"
    if not path.exists():
        get_vendor_config(vendor, architecture)          # the refusal that names the missing profile
    value = (_parsed_vendor_file(str(path)).get(section) or {}).get(key, _ABSENT)
    if value is _ABSENT:
        return default
    return copy.deepcopy(value) if isinstance(value, (dict, list)) else value


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
