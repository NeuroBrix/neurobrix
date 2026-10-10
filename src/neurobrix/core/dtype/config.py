# core/dtype/config.py
"""
Dtype Configuration - Single Source of Truth

Consolidates dtype constants previously duplicated across:
- core/prism/common/dtype_resolver.py
- core/prism/common/allocation.py
- core/runtime/weight_loader.py

ZERO HARDCODE: All dtype-related constants defined here.
"""
from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # R33: torch is the ATen branch's; this table is shared
    import torch

# The torch views of the dtype table are built on first use, by the ATen
# branch. A --triton process reads the string helpers only and never imports
# torch (R33). Widths live in config/dtypes.yml, read through core/dtype/itemsize.py; `DTYPE_MAP` / `DTYPE_TO_STR` stay importable
# through the module __getattr__ below.
_DTYPE_NAMES = ("bfloat16", "float16", "float32", "float64",
                "int8", "int16", "int32", "int64", "uint8", "bool")
_TORCH_MAPS = None


def _torch_maps():
    global _TORCH_MAPS
    if _TORCH_MAPS is None:
        import torch
        dtype_map = {name: getattr(torch, name) for name in _DTYPE_NAMES}
        _TORCH_MAPS = (dtype_map, {v: k for k, v in dtype_map.items()})
    return _TORCH_MAPS


def __getattr__(name):
    if name == "DTYPE_MAP":
        return _torch_maps()[0]
    if name == "DTYPE_TO_STR":
        return _torch_maps()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert dtype string to torch.dtype.

    Args:
        dtype_str: Dtype string (e.g., "float16", "bfloat16")

    Returns:
        torch.dtype (defaults to float32 if unknown)
    """
    import torch
    return _torch_maps()[0].get(dtype_str, torch.float32)


def parse_dtype(dtype_str: str, compute_dtype: Optional[torch.dtype] = None) -> torch.dtype:
    """
    Parse dtype string to torch.dtype with optional Prism remap.

    Handles both "float16" and "torch.float16" formats.
    When compute_dtype is provided, remaps half-precision dtypes:
    bf16→fp16 when Prism wants fp16 (and vice versa).

    This is the SINGLE implementation of dtype parsing + Prism remap.
    All runtime code must use this instead of inline dicts or ad-hoc remaps.

    Args:
        dtype_str: Dtype string ("float16", "torch.float16", etc.)
        compute_dtype: Prism compute dtype for half-precision remap (optional)

    Returns:
        Resolved torch.dtype
    """
    # Strip "torch." prefix
    import torch
    clean = dtype_str[6:] if dtype_str.startswith("torch.") else dtype_str
    parsed = _torch_maps()[0].get(clean, torch.float32)

    # Prism remap: bf16↔fp16 when hardware wants a different half-precision
    if compute_dtype is not None:
        if parsed == torch.bfloat16 and compute_dtype == torch.float16:
            return torch.float16
        if parsed == torch.float16 and compute_dtype == torch.bfloat16:
            return torch.bfloat16

    return parsed


def strip_aten_prefix(op_type: str) -> str:
    """
    Strip 'aten::' prefix and variant suffix from op_type string.

    "aten::_softmax" → "_softmax"
    "aten::mm"       → "mm"
    "custom::rms_norm" → "rms_norm"

    SINGLE implementation — all runtime code must use this.
    """
    # Strip namespace prefix (aten::, custom::, etc.)
    if "::" in op_type:
        op_name = op_type.split("::", 1)[1]
    else:
        op_name = op_type
    # Strip variant suffix (.default, .int, etc.)
    return op_name.split(".")[0]


def dtype_to_str(dtype: torch.dtype) -> str:
    """
    Convert torch.dtype to string.

    Args:
        dtype: torch.dtype

    Returns:
        Dtype string (defaults to "float32" if unknown)
    """
    return _torch_maps()[1].get(dtype, "float32")


# ============================================================================
# Hardware Support Mapping
# ============================================================================

# Which architectures support which dtypes
HARDWARE_DTYPE_SUPPORT: Dict[str, list] = {
    # NVIDIA
    "volta": ["float32", "float16"],  # V100
    "turing": ["float32", "float16"],  # RTX 20xx
    "ampere": ["float32", "float16", "bfloat16"],  # A100, RTX 30xx
    "hopper": ["float32", "float16", "bfloat16", "fp8"],  # H100

    # AMD
    "cdna": ["float32", "float16"],  # MI100
    "cdna2": ["float32", "float16", "bfloat16"],  # MI200
    "cdna3": ["float32", "float16", "bfloat16", "fp8"],  # MI300

    # CPU fallback
    "cpu": ["float32", "float64"],
}


def architecture_supports_dtype(arch: str, dtype_str: str) -> bool:
    """
    Check if a hardware architecture supports a dtype.

    Args:
        arch: Architecture string (e.g., "volta", "ampere")
        dtype_str: Dtype string (e.g., "bfloat16")

    Returns:
        True if supported, False otherwise
    """
    supported = HARDWARE_DTYPE_SUPPORT.get(arch.lower(), ["float32"])
    return dtype_str in supported


def device_supports_fp64(device: str, vendor: str, architecture: str) -> bool:
    """Whether the compiled branch computes and stores float64 / complex128 on `device` — the host's
    from this module's own table (`HARDWARE_DTYPE_SUPPORT["cpu"]`), an accelerator's from its vendor
    profile: the device's (`precision.supports_fp64`) AND ATen's kernels' there
    (`precision.kernels_carry_fp64.compiled`, a capability) AND the compiled branch's storage policy
    (`precision.stores_fp64.compiled`), all required, a missing key refused. False on Apple GPUs:
    MPS refuses a float64 tensor outright, and the vendor's own code narrows to float32 there
    (diffusers' RoPE, 2026-10-08). The Triton branch reads its own keys the same way
    (`triton.dtype.triton_stores_fp64`)."""
    if str(device).split(":")[0] == "cpu":
        return architecture_supports_dtype("cpu", "float64")
    from neurobrix.core.config import loader
    precision = loader.get_vendor_config(vendor, architecture).get("precision") or {}
    carry = (precision.get("kernels_carry_fp64") or {}).get("compiled")
    stores = (precision.get("stores_fp64") or {}).get("compiled")
    if not all(isinstance(v, bool) for v in (precision.get("supports_fp64"), carry, stores)):
        raise ValueError(f"ZERO FALLBACK: the {vendor}/{architecture} profile does not declare "
                         f"precision.supports_fp64, precision.kernels_carry_fp64.compiled and "
                         f"precision.stores_fp64.compiled (found {precision!r}); a device's fp64 "
                         f"is never assumed")
    return precision["supports_fp64"] and carry and stores


def profile_device_supports_fp64(profile) -> bool:
    """`device_supports_fp64` of a hardware profile (a PrismProfile) for the compiled branch: every
    device's, refused if they disagree (one plan prices one answer). A profile with NO device
    (`config/hardware/cpu-only-x86.yml`, `devices: []`) runs every component on the host
    (`cpu_execution`), so it answers the host's own (`device_supports_fp64("cpu", ...)`); no profile
    at all is refused. The mirror of `triton.dtype.profile_triton_stores_fp64`."""
    if profile is None:
        raise ValueError("ZERO FALLBACK: no hardware profile; the compiled branch's fp64 is read "
                         "from its vendor profile, never assumed")
    if not getattr(profile, "devices", None):
        return device_supports_fp64("cpu", None, None)
    answers = {}
    for dev in profile.devices:
        key = (getattr(dev.brand, "value", dev.brand), dev.architecture)
        answers[key] = device_supports_fp64("gpu", *key)
    if len(set(answers.values())) > 1:
        raise ValueError(f"ZERO FALLBACK: the profile's devices disagree on the compiled branch's "
                         f"fp64 ({answers!r}); one plan prices one answer, so a mixed profile is "
                         f"refused rather than decided by its first device")
    return next(iter(answers.values()))
