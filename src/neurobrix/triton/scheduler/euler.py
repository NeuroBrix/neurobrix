"""Triton Euler schedulers — zero-torch mirror of
core/module/scheduler/diffusion/euler.py.

EulerDiscrete (first-order, optional Karras sigmas) and EulerAncestral (adds
stochastic noise per step). The sigma/timestep schedule is CPU numpy (init-time,
1000-entry arrays); every per-step op on the latent is NBXTensor × Python-float
scalar — no torch. Mirrors the core algorithm op-for-op so the two engines agree
to fp16 on identical inputs (validated by a unit test vs the core scheduler).

EulerAncestral draws N(0,1) noise via the NBXTensor randn wrapper, which honours
the shared `NBX_FORCE_RAND_SEED` pin for cross-mode determinism.
"""
import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor
from .noise_schedules import (
    get_beta_schedule, betas_to_alphas_cumprod, alphas_cumprod_to_sigmas,
    get_timesteps_np, karras_sigmas,
)


def _to_nbx_timesteps(ts_np: np.ndarray):
    """Expose timesteps as a list of NBXTensor [1] (tensor-like for loop/CFG)."""
    return [NBXTensor.from_numpy(np.array([float(t)], dtype=np.float32))
            for t in ts_np]


class _EulerBase:
    """Shared schedule + scaling for the Euler family (NBXTensor, zero torch)."""

    def __init__(self, config: dict):
        self.config = config
        self.num_train_timesteps = int(config.get("num_train_timesteps", 1000))
        self.beta_start = float(config.get("beta_start", 0.0001))
        self.beta_end = float(config.get("beta_end", 0.02))
        self.beta_schedule = config.get("beta_schedule", "linear")
        self.prediction_type = config.get("prediction_type", "epsilon")
        self.timestep_spacing = config.get("timestep_spacing", "linspace")
        self.use_karras_sigmas = bool(config.get("use_karras_sigmas", False))
        self.sigma_min = float(config.get("sigma_min", 0.002))
        self.sigma_max = float(config.get("sigma_max", 80.0))

        betas = get_beta_schedule(self.beta_schedule, self.num_train_timesteps,
                                  self.beta_start, self.beta_end)
        self.alphas_cumprod = betas_to_alphas_cumprod(betas)          # [T] f64
        self._all_sigmas = alphas_cumprod_to_sigmas(self.alphas_cumprod)

        self.num_inference_steps = None
        self._ts_np = None       # np.int64 [N]
        self.timesteps = None    # list[NBXTensor [1]]
        self.sigmas = None       # np.float64 [N+1] (zero appended)
        self._step_index = None

    def set_timesteps(self, num_inference_steps: int, device=None, **kwargs):
        self.num_inference_steps = num_inference_steps
        if self.use_karras_sigmas:
            sig = karras_sigmas(num_inference_steps, self.sigma_min,
                                self.sigma_max, rho=7.0)
            ts = np.array([int(np.abs(self._all_sigmas - s).argmin()) for s in sig],
                          dtype=np.int64)
        else:
            ts = get_timesteps_np(self.timestep_spacing, num_inference_steps,
                                  self.num_train_timesteps)
            idx = np.clip(ts.astype(np.int64), 0, self.num_train_timesteps - 1)
            sig = alphas_cumprod_to_sigmas(self.alphas_cumprod[idx])
        self._ts_np = ts.astype(np.int64)
        self.sigmas = np.concatenate([sig.astype(np.float64), np.zeros(1)])
        self.timesteps = _to_nbx_timesteps(self._ts_np)
        self._step_index = None

    def _init_step_index(self, timestep):
        t = int(timestep.item()) if isinstance(timestep, NBXTensor) else int(timestep)
        # nearest timestep in the schedule (mirror core init_step_index)
        self._step_index = int(np.abs(self._ts_np - t).argmin())

    @property
    def step_index(self):
        return self._step_index

    def scale_model_input(self, sample: NBXTensor, timestep) -> NBXTensor:
        if self._step_index is None:
            self._init_step_index(timestep)
        sigma = float(self.sigmas[self._step_index])
        return sample / ((sigma ** 2 + 1.0) ** 0.5)

    @property
    def init_noise_sigma(self) -> float:
        if self.sigmas is not None and len(self.sigmas) > 0:
            return float(self.sigmas[0])
        return 1.0

    def _denoised_and_derivative(self, model_output: NBXTensor,
                                 sample: NBXTensor, sigma: float):
        """Return (denoised, d) per prediction_type — NBXTensor × scalar."""
        if self.prediction_type == "epsilon":
            denoised = sample - model_output * sigma
            d = model_output  # (sample - denoised) / sigma == model_output
        elif self.prediction_type == "v_prediction":
            alpha = 1.0 / ((sigma ** 2 + 1.0) ** 0.5)
            denoised = sample * alpha - model_output * (sigma * alpha)
            d = (sample - denoised) * (1.0 / max(sigma, 1e-8))
        else:  # "sample"
            denoised = model_output
            d = (sample - denoised) * (1.0 / max(sigma, 1e-8))
        return denoised, d

    @staticmethod
    def _maybe_chunk_variance(model_output: NBXTensor, sample: NBXTensor):
        # Variance prediction → model_output has 2× the sample channels; keep first half.
        if (model_output.ndim >= 2 and sample.ndim >= 2
                and model_output.shape[1] == sample.shape[1] * 2):
            return model_output.narrow(1, 0, sample.shape[1]).contiguous()
        return model_output


class TritonEulerDiscreteScheduler(_EulerBase):
    def step(self, model_output: NBXTensor, timestep, sample: NBXTensor,
             return_dict: bool = True, **kwargs):
        if self.num_inference_steps is None:
            raise RuntimeError("ZERO FALLBACK: set_timesteps() before step()")
        model_output = self._maybe_chunk_variance(model_output, sample)
        if self._step_index is None:
            self._init_step_index(timestep)
        sigma = float(self.sigmas[self._step_index])
        sigma_next = float(self.sigmas[self._step_index + 1])
        denoised, d = self._denoised_and_derivative(model_output, sample, sigma)
        prev = sample + d * (sigma_next - sigma)
        self._step_index += 1
        return {"prev_sample": prev, "denoised": denoised} if return_dict else prev


class TritonEulerAncestralDiscreteScheduler(_EulerBase):
    def step(self, model_output: NBXTensor, timestep, sample: NBXTensor,
             return_dict: bool = True, **kwargs):
        if self.num_inference_steps is None:
            raise RuntimeError("ZERO FALLBACK: set_timesteps() before step()")
        from neurobrix.kernels.wrappers import randn_like_wrapper
        model_output = self._maybe_chunk_variance(model_output, sample)
        if self._step_index is None:
            self._init_step_index(timestep)
        sigma = float(self.sigmas[self._step_index])
        sigma_next = float(self.sigmas[self._step_index + 1])
        denoised, _ = self._denoised_and_derivative(model_output, sample, sigma)
        # Ancestral split of sigma_next into deterministic + stochastic parts.
        sigma_up = (max(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2)
                        / max(sigma ** 2, 1e-12), 0.0)) ** 0.5
        sigma_down = (max(sigma_next ** 2 - sigma_up ** 2, 0.0)) ** 0.5
        d = (sample - denoised) * (1.0 / max(sigma, 1e-8))
        prev = sample + d * (sigma_down - sigma)
        if sigma_next > 0:
            noise = randn_like_wrapper(sample)
            prev = prev + noise * sigma_up
        self._step_index += 1
        return {"prev_sample": prev, "denoised": denoised} if return_dict else prev
