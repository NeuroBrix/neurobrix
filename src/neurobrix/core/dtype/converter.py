# core/dtype/converter.py
"""
Dtype Converter - Safe Dtype Conversion

Consolidates dtype conversion logic previously duplicated across:
- core/runtime/weight_loader.py
- core/io/loader.py

Prism is the master for dtype decisions. bf16→fp16 is allowed when Prism
has verified all weight values fit within ±65504 (fp16 range).
"""

import torch
from typing import Dict

from neurobrix.core.dtype.config import get_dtype_bytes


def safe_dtype_convert(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert tensor dtype with safety clamping for bf16→fp16.

    Prism validates bf16→fp16 safety upstream via _scan_bf16_fp16_safety().
    This function adds clamping as a defense-in-depth measure to prevent
    inf/NaN from overflow in case any values exceed fp16 range (±65504).

    Args:
        tensor: Source tensor
        target_dtype: Target dtype (decided by Prism)

    Returns:
        Converted tensor
    """
    if tensor.dtype == target_dtype:
        return tensor

    if not tensor.is_floating_point():
        return tensor.to(target_dtype)

    # bf16→fp16: clamp to fp16 range to prevent overflow → inf → NaN
    if tensor.dtype == torch.bfloat16 and target_dtype == torch.float16:
        return tensor.clamp(-65504.0, 65504.0).to(torch.float16)

    return tensor.to(target_dtype)


def safe_dtype_convert_dict(
    tensors: Dict[str, torch.Tensor],
    target_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """
    Convert all tensors in a dict to target dtype.

    Args:
        tensors: Dict of name -> tensor
        target_dtype: Target dtype (decided by Prism)

    Returns:
        Dict with converted tensors
    """
    return {
        name: safe_dtype_convert(tensor, target_dtype)
        for name, tensor in tensors.items()
    }


def calculate_dtype_multiplier(source_dtype: str, target_dtype: str) -> float:
    """
    Calculate memory multiplier for dtype conversion.

    Used by Prism for memory calculations during allocation planning.

    Args:
        source_dtype: Source dtype string (e.g., "float16")
        target_dtype: Target dtype string (e.g., "float32")

    Returns:
        Multiplier (e.g., 2.0 for fp16->fp32)
    """
    source_bytes = get_dtype_bytes(source_dtype)
    target_bytes = get_dtype_bytes(target_dtype)
    return target_bytes / source_bytes


def resolve_safe_fallback(
    requested: str,
    supported: list,
    log_prefix: str = "[Dtype]",
) -> str:
    """
    Resolve safe fallback dtype when requested is not supported.

    Prism decides the target dtype. This is a legacy fallback for non-Prism paths.
    bf16→fp16 is handled by Prism's safety scan upstream.

    Args:
        requested: Requested dtype string
        supported: List of supported dtype strings
        log_prefix: Log message prefix

    Returns:
        Safe dtype string
    """
    if requested in supported:
        return requested

    if requested == "bfloat16":
        if "float16" in supported:
            return "float16"
        return "float32"

    if requested == "float16":
        return "float32"

    return "float32"
