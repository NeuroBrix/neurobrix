"""Triton DDIM scheduler — zero-torch mirror of
core/module/scheduler/diffusion/ddim.py.

Denoising Diffusion Implicit Models (Song et al. 2020): deterministic at eta=0,
stochastic at eta>0. The alpha_cumprod / timestep schedule is CPU numpy
(init-time); every per-step op on the latent is NBXTensor × Python-float scalar
— no torch. Mirrors the core algorithm op-for-op (validated by a unit test vs the
core scheduler). eta>0 noise is drawn via the NBXTensor randn wrapper (honours
the `NBX_FORCE_RAND_SEED` pin).
"""
import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor
from .noise_schedules import get_beta_schedule, betas_to_alphas_cumprod, get_timesteps_np


class TritonDDIMScheduler:
    def __init__(self, config: dict):
        self.config = config
        self.num_train_timesteps = int(config.get("num_train_timesteps", 1000))
        self.beta_start = float(config.get("beta_start", 0.0001))
        self.beta_end = float(config.get("beta_end", 0.02))
        self.beta_schedule = config.get("beta_schedule", "linear")
        self.prediction_type = config.get("prediction_type", "epsilon")
        self.timestep_spacing = config.get("timestep_spacing", "leading")
        self.clip_sample = bool(config.get("clip_sample", True))
        self.clip_sample_range = float(config.get("clip_sample_range", 1.0))
        self.eta = float(config.get("eta", 0.0))
        self.thresholding = bool(config.get("thresholding", False))

        betas = get_beta_schedule(self.beta_schedule, self.num_train_timesteps,
                                  self.beta_start, self.beta_end)
        self.alphas_cumprod = betas_to_alphas_cumprod(betas)   # [T] np.float64
        self.final_alpha_cumprod = 1.0

        self.num_inference_steps = None
        self._ts_np = None       # np.int64 [N]
        self.timesteps = None    # list[NBXTensor [1]]
        self._step_index = None

    def set_timesteps(self, num_inference_steps: int, device=None, **kwargs):
        self.num_inference_steps = num_inference_steps
        self._ts_np = get_timesteps_np(self.timestep_spacing, num_inference_steps,
                                       self.num_train_timesteps).astype(np.int64)
        self.timesteps = [NBXTensor.from_numpy(np.array([float(t)], dtype=np.float32))
                          for t in self._ts_np]
        self._step_index = None

    def _init_step_index(self, timestep):
        t = int(timestep.item()) if isinstance(timestep, NBXTensor) else int(timestep)
        self._step_index = int(np.abs(self._ts_np - t).argmin())

    @property
    def step_index(self):
        return self._step_index

    def scale_model_input(self, sample: NBXTensor, timestep) -> NBXTensor:
        return sample  # DDIM does not scale the input

    @property
    def init_noise_sigma(self) -> float:
        return 1.0

    def step(self, model_output: NBXTensor, timestep, sample: NBXTensor,
             return_dict: bool = True, **kwargs):
        if self.num_inference_steps is None:
            raise RuntimeError("ZERO FALLBACK: set_timesteps() before step()")
        if self.thresholding:
            raise NotImplementedError(
                "ZERO FALLBACK: triton DDIM dynamic thresholding unwired "
                "(needs a percentile reduction kernel) — follow-up chantier. "
                "clip_sample is supported.")
        # Variance prediction → keep first half of the channels.
        if (model_output.ndim >= 2 and sample.ndim >= 2
                and model_output.shape[1] == sample.shape[1] * 2):
            model_output = model_output.narrow(1, 0, sample.shape[1]).contiguous()
        if self._step_index is None:
            self._init_step_index(timestep)

        t = int(timestep.item()) if isinstance(timestep, NBXTensor) else int(timestep)
        if self._step_index == len(self._ts_np) - 1:
            prev_t = 0
        else:
            prev_t = int(self._ts_np[self._step_index + 1])

        a_t = float(self.alphas_cumprod[int(t)])
        a_prev = float(self.alphas_cumprod[int(prev_t)]) if prev_t > 0 else self.final_alpha_cumprod
        sqrt_a_t = a_t ** 0.5
        sqrt_1m_a_t = (1.0 - a_t) ** 0.5
        sqrt_a_prev = a_prev ** 0.5

        if self.prediction_type == "epsilon":
            pred_x0 = (sample - model_output * sqrt_1m_a_t) * (1.0 / max(sqrt_a_t, 1e-8))
            pred_eps = model_output
        elif self.prediction_type == "v_prediction":
            pred_x0 = sample * sqrt_a_t - model_output * sqrt_1m_a_t
            pred_eps = model_output * sqrt_a_t + sample * sqrt_1m_a_t
        else:  # "sample"
            pred_x0 = model_output
            pred_eps = (sample - pred_x0 * sqrt_a_t) * (1.0 / max(sqrt_1m_a_t, 1e-8))

        if self.clip_sample:
            from neurobrix.kernels.wrappers import clamp
            pred_x0 = clamp(pred_x0, -self.clip_sample_range, self.clip_sample_range)

        variance = max((1.0 - a_prev) / max(1.0 - a_t, 1e-12)
                       * (1.0 - a_t / max(a_prev, 1e-12)), 0.0)
        sigma = self.eta * (variance ** 0.5)
        dir_coeff = max(1.0 - a_prev - sigma ** 2, 0.0) ** 0.5
        prev = pred_x0 * sqrt_a_prev + pred_eps * dir_coeff
        if self.eta > 0 and sigma > 0:
            from neurobrix.kernels.wrappers import randn_like_wrapper
            prev = prev + randn_like_wrapper(sample) * sigma

        self._step_index += 1
        return ({"prev_sample": prev, "pred_original_sample": pred_x0}
                if return_dict else prev)
