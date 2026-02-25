# core/scheduler/utils/noise_schedules.py
"""
Noise Schedule Utilities - ZERO FALLBACK.

Implements various beta/alpha schedules used by diffusion models.
All parameters REQUIRED - no default values.
"""

import torch
import math
from typing import Tuple


def linear_beta_schedule(
    num_timesteps: int,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    """
    Linear beta schedule (DDPM original).

    Args:
        num_timesteps: Number of diffusion steps (REQUIRED)
        beta_start: Starting beta value (REQUIRED)
        beta_end: Ending beta value (REQUIRED)

    Returns:
        Beta values tensor
    """
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)


def scaled_linear_beta_schedule(
    num_timesteps: int,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    """
    Scaled linear beta schedule (Stable Diffusion).

    Args:
        num_timesteps: Number of diffusion steps (REQUIRED)
        beta_start: Starting beta value (REQUIRED)
        beta_end: Ending beta value (REQUIRED)

    Returns:
        Beta values tensor
    """
    return torch.linspace(
        beta_start ** 0.5,
        beta_end ** 0.5,
        num_timesteps,
        dtype=torch.float32
    ) ** 2


def cosine_beta_schedule(
    num_timesteps: int,
    s: float = 0.008,  # Safe default: paper constant, not model-specific
) -> torch.Tensor:
    """
    Cosine beta schedule (Improved DDPM).

    As proposed in "Improved Denoising Diffusion Probabilistic Models"

    Args:
        num_timesteps: Number of diffusion steps (REQUIRED)
        s: Offset parameter (safe default from paper)

    Returns:
        Beta values tensor
    """
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps, dtype=torch.float32) / num_timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(
    num_timesteps: int,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    """
    Sigmoid beta schedule.

    Args:
        num_timesteps: Number of diffusion steps (REQUIRED)
        beta_start: Starting beta value (REQUIRED)
        beta_end: Ending beta value (REQUIRED)

    Returns:
        Beta values tensor
    """
    betas = torch.linspace(-6, 6, num_timesteps, dtype=torch.float32)
    betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    return betas


def get_beta_schedule(
    schedule_type: str,
    num_timesteps: int,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    """
    Get beta schedule by name.

    ZERO FALLBACK: All parameters REQUIRED, no defaults.

    Args:
        schedule_type: Schedule type name (REQUIRED)
        num_timesteps: Number of timesteps (REQUIRED)
        beta_start: Starting beta (REQUIRED)
        beta_end: Ending beta (REQUIRED)

    Returns:
        Beta values tensor

    Raises:
        ValueError: If schedule type unknown
    """
    schedules = {
        "linear": linear_beta_schedule,
        "scaled_linear": scaled_linear_beta_schedule,
        "squaredcos_cap_v2": cosine_beta_schedule,
        "cosine": cosine_beta_schedule,
        "sigmoid": sigmoid_beta_schedule,
    }

    if schedule_type not in schedules:
        raise ValueError(
            f"ZERO FALLBACK: Unknown beta schedule '{schedule_type}'.\n"
            f"Supported: {list(schedules.keys())}"
        )

    if schedule_type in ("linear", "scaled_linear", "sigmoid"):
        return schedules[schedule_type](num_timesteps, beta_start, beta_end)
    else:
        # Cosine schedule uses paper constant, doesn't need beta_start/end
        return schedules[schedule_type](num_timesteps)


def betas_to_alphas(betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert betas to alphas and alphas_cumprod.

    Args:
        betas: Beta values tensor

    Returns:
        Tuple of (alphas, alphas_cumprod)
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alphas_cumprod


def alphas_to_sigmas(alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    Convert alphas_cumprod to sigmas (Karras formulation).

    Args:
        alphas_cumprod: Cumulative product of alphas

    Returns:
        Sigma values tensor
    """
    return ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5


def sigmas_to_alphas(sigmas: torch.Tensor) -> torch.Tensor:
    """
    Convert sigmas to alphas.

    Args:
        sigmas: Sigma values tensor

    Returns:
        Alpha values tensor
    """
    return 1.0 / (1.0 + sigmas ** 2)


def rescale_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    """
    Rescale betas to have zero terminal SNR.

    Based on https://arxiv.org/abs/2305.08891

    Args:
        betas: Original beta values

    Returns:
        Rescaled beta values
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Rescale
    alphas_cumprod_sqrt = alphas_cumprod.sqrt()
    alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
    alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()

    alphas_cumprod_sqrt = (
        (alphas_cumprod_sqrt - alphas_cumprod_sqrt_T)
        / (alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T)
    )

    alphas_cumprod = alphas_cumprod_sqrt ** 2

    # Derive betas from rescaled alphas
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = torch.cat([alphas_cumprod[0:1], alphas])
    betas = 1 - alphas

    return betas
