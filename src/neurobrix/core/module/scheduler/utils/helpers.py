"""
Shared helper functions for scheduler implementations.

These functions are extracted from individual schedulers to avoid duplication.
All schedulers share these common utilities for step index and thresholding.
"""

import torch
from typing import Union


def init_step_index(
    timesteps: torch.Tensor,
    timestep: Union[int, torch.Tensor]
) -> int:
    """
    Initialize step index from timestep.

    Finds the index in timesteps tensor that corresponds to the given timestep.
    If exact match not found, uses closest match.

    Args:
        timesteps: Tensor of timestep values from set_timesteps()
        timestep: Current timestep (int or 0-d tensor)

    Returns:
        Step index (int)

    Used by: EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
             DDIMScheduler, DPMSolverPPScheduler
    """
    if isinstance(timestep, torch.Tensor):
        timestep_val = timestep.item()
    else:
        timestep_val = timestep

    step_indices = (timesteps == timestep_val).nonzero()
    if step_indices.numel() == 0:
        # Exact match not found - use closest
        step_indices = (timesteps - timestep_val).abs().argmin()
        return int(step_indices.item())
    else:
        return int(step_indices[0].item())


def threshold_sample(
    sample: torch.Tensor,
    dynamic_thresholding_ratio: float = 0.995,
    sample_max_value: float = 1.0
) -> torch.Tensor:
    """
    Dynamic thresholding from Imagen paper.

    Clips sample values based on dynamic quantile thresholds.
    Helps prevent over-saturation during sampling.

    Args:
        sample: Input tensor [batch, channels, ...]
        dynamic_thresholding_ratio: Quantile for threshold computation (default 0.995)
        sample_max_value: Maximum value for the threshold (default 1.0)

    Returns:
        Thresholded sample tensor with same shape as input

    Used by: DDIMScheduler, DPMSolverPPScheduler
    """
    dtype = sample.dtype
    batch_size, channels = sample.shape[:2]
    # Save spatial dimensions before flattening
    spatial_dims = sample.shape[2:]
    sample = sample.reshape(batch_size, channels, -1).float()

    abs_sample = sample.abs()
    s = torch.quantile(abs_sample, dynamic_thresholding_ratio, dim=-1)
    s = torch.clamp(s, min=1.0, max=sample_max_value)
    s = s.unsqueeze(-1)

    sample = torch.clamp(sample, -s, s) / s
    # Restore original spatial dimensions
    sample = sample.reshape(batch_size, channels, *spatial_dims)
    return sample.to(dtype)
