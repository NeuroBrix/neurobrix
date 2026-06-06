"""Noise-schedule math for the triton schedulers — pure numpy (zero torch).

Mirror of core/module/scheduler/utils/noise_schedules.py with torch -> numpy.
These run ONCE at scheduler construction / set_timesteps on tiny 1-D arrays
(betas/alphas/sigmas of length num_train_timesteps), never on the GPU latent —
so numpy on CPU is correct and torch-free.
"""
import math
import numpy as np


def get_beta_schedule(beta_schedule: str, num_timesteps: int,
                      beta_start: float, beta_end: float) -> np.ndarray:
    if beta_schedule == "linear":
        return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    if beta_schedule in ("scaled_linear",):
        return np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                           num_timesteps, dtype=np.float64) ** 2
    if beta_schedule in ("cosine", "squaredcos_cap_v2"):
        s = 0.008
        steps = num_timesteps + 1
        t = np.linspace(0, num_timesteps, steps, dtype=np.float64) / num_timesteps
        acp = np.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        acp = acp / acp[0]
        betas = 1 - (acp[1:] / acp[:-1])
        return np.clip(betas, 0, 0.999)
    raise RuntimeError(f"ZERO FALLBACK: unknown beta_schedule '{beta_schedule}'")


def betas_to_alphas_cumprod(betas: np.ndarray) -> np.ndarray:
    alphas = 1.0 - betas
    return np.cumprod(alphas, axis=0)


def alphas_cumprod_to_sigmas(alphas_cumprod: np.ndarray) -> np.ndarray:
    # sigma = sqrt((1 - acp) / acp)  (standard diffusion karras-free sigma)
    return np.sqrt((1.0 - alphas_cumprod) / alphas_cumprod)


def get_timesteps_linspace(num_inference_steps: int, num_train_timesteps: int) -> np.ndarray:
    # diffusers "linspace": linspace(0, T-1, n+1).round()[::-1][:-1]
    ts = np.linspace(0, num_train_timesteps - 1, num_inference_steps + 1)
    return ts.round()[::-1][:-1].copy().astype(np.int64)
