# core/scheduler/utils/timestep_utils.py
"""
Timestep Spacing Utilities - ZERO FALLBACK.

Different models use different timestep spacing strategies.
All parameters REQUIRED where model-specific.
"""

import torch
import numpy as np
from typing import Optional


def linspace_timesteps(
    num_inference_steps: int,
    num_train_timesteps: int,
) -> torch.Tensor:
    """
    Linear spacing with proper diffusers-compatible formula.

    Args:
        num_inference_steps: Number of inference steps (REQUIRED)
        num_train_timesteps: Number of training timesteps (REQUIRED)

    Returns:
        Timesteps tensor
    """
    # Matches diffusers: linspace(0, T-1, n+1).round()[::-1][:-1]
    timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps + 1)
    timesteps = timesteps.round()[::-1][:-1].copy().astype(np.int64)
    return torch.from_numpy(timesteps)


def leading_timesteps(
    num_inference_steps: int,
    num_train_timesteps: int,
) -> torch.Tensor:
    """
    Leading spacing.

    Args:
        num_inference_steps: Number of inference steps (REQUIRED)
        num_train_timesteps: Number of training timesteps (REQUIRED)

    Returns:
        Timesteps tensor
    """
    step_ratio = num_train_timesteps // num_inference_steps
    timesteps = (torch.arange(0, num_inference_steps) * step_ratio).flip(0)
    return timesteps.long()


def trailing_timesteps(
    num_inference_steps: int,
    num_train_timesteps: int,
) -> torch.Tensor:
    """
    Trailing spacing (diffusers default for DPM++).

    Args:
        num_inference_steps: Number of inference steps (REQUIRED)
        num_train_timesteps: Number of training timesteps (REQUIRED)

    Returns:
        Timesteps tensor
    """
    # This is the formula diffusers uses for DPMSolverMultistepScheduler
    timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps + 1)
    timesteps = timesteps.round()[::-1][:-1].copy().astype(np.int64)
    return torch.from_numpy(timesteps)


def karras_sigmas(
    num_inference_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
) -> torch.Tensor:
    """
    Karras sigma schedule.

    From "Elucidating the Design Space of Diffusion-Based Generative Models"

    ZERO FALLBACK: All parameters REQUIRED - differ between models.

    Args:
        num_inference_steps: Number of inference steps (REQUIRED)
        sigma_min: Minimum sigma value (REQUIRED)
        sigma_max: Maximum sigma value (REQUIRED)
        rho: Rho parameter (REQUIRED)

    Returns:
        Sigmas tensor
    """
    ramp = torch.linspace(0, 1, num_inference_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


def get_timesteps(
    spacing: str,
    num_inference_steps: int,
    num_train_timesteps: int,
) -> torch.Tensor:
    """
    Get timesteps by spacing type.

    ZERO FALLBACK: All parameters REQUIRED.

    Args:
        spacing: Spacing type name (REQUIRED)
        num_inference_steps: Number of inference steps (REQUIRED)
        num_train_timesteps: Number of training timesteps (REQUIRED)

    Returns:
        Timesteps tensor

    Raises:
        ValueError: If spacing type unknown
    """
    if spacing == "linspace":
        return linspace_timesteps(num_inference_steps, num_train_timesteps)
    elif spacing == "leading":
        return leading_timesteps(num_inference_steps, num_train_timesteps)
    elif spacing == "trailing":
        return trailing_timesteps(num_inference_steps, num_train_timesteps)
    else:
        raise ValueError(
            f"ZERO FALLBACK: Unknown timestep spacing '{spacing}'.\n"
            f"Supported: linspace, leading, trailing"
        )


def flow_timesteps(
    num_inference_steps: int,
    shift: float,
) -> torch.Tensor:
    """
    Flow matching timesteps (Flux, SD3).

    ZERO FALLBACK: shift parameter REQUIRED - differs between models:
    - Flux: shift=3.0
    - SD3: shift=1.0

    Args:
        num_inference_steps: Number of inference steps (REQUIRED)
        shift: Shift parameter (REQUIRED - CRITICAL for correct output)

    Returns:
        Timesteps in [0, 1] range with shift applied
    """
    timesteps = torch.linspace(1, 0, num_inference_steps + 1)[:-1]

    if shift != 1.0:
        # Apply shift for better sampling
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)

    return timesteps
