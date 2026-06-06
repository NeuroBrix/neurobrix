"""Triton DPM-Solver++ — zero-torch mirror of
core/module/scheduler/diffusion/dpm_solver_pp.py.

Same algorithm, byte-for-byte coefficient math, but: the noise schedule is numpy
(CPU, init-time), every per-step coefficient (sigma_t, alpha_t, lambda, h, r0,
exp/log) is a Python float, and the only tensor operands (sample, model_output)
are NBXTensor — whose +,-,*,/ dispatch to triton kernels. No torch anywhere.

Coefficients are computed in fp64/float (numpy/math) exactly like diffusers'
fp32 lambda/sigma path; the NBXTensor latent keeps its own dtype.
"""
import math
import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
from .noise_schedules import (
    get_beta_schedule, betas_to_alphas_cumprod, alphas_cumprod_to_sigmas,
    get_timesteps_linspace,
)


class TritonDPMSolverPPScheduler:
    """Zero-torch DPM-Solver++ multistep scheduler (NBXTensor latents)."""

    def __init__(self, config: dict):
        self.config = config
        self.num_train_timesteps = int(config.get("num_train_timesteps", 1000))
        self.beta_start = float(config.get("beta_start", 0.0001))
        self.beta_end = float(config.get("beta_end", 0.02))
        self.beta_schedule = config.get("beta_schedule", "linear")
        self.prediction_type = config.get("prediction_type", "epsilon")
        self.solver_order = int(config.get("solver_order", 2))
        self.algorithm_type = config.get("algorithm_type", "dpmsolver++")
        self.solver_type = config.get("solver_type", "midpoint")
        self.final_sigmas_type = config.get("final_sigmas_type", "zero")
        self.lower_order_final = bool(config.get("lower_order_final", True))
        self.use_flow_sigmas = bool(config.get("use_flow_sigmas", False))
        self.flow_shift = float(config.get("flow_shift", 1.0))

        # CPU schedule (numpy, fp64) — computed once.
        betas = get_beta_schedule(self.beta_schedule, self.num_train_timesteps,
                                  self.beta_start, self.beta_end)
        self.alphas_cumprod = betas_to_alphas_cumprod(betas)  # [T] np.float64

        self.num_inference_steps = None
        self.timesteps = None           # np.int64 [N]
        self.sigmas = None              # np.float64 [N+1] (zero appended)
        self._step_index = None
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0

    # ---- schedule ----
    def set_timesteps(self, num_inference_steps: int, device=None, **kwargs):
        self.num_inference_steps = num_inference_steps
        if self.use_flow_sigmas:
            alphas = np.linspace(1, 1 / self.num_train_timesteps,
                                 num_inference_steps + 1, dtype=np.float64)
            sigmas = 1.0 - alphas
            sigmas = np.flip(
                self.flow_shift * sigmas / (1 + (self.flow_shift - 1) * sigmas)
            )[:-1].copy()
            timesteps = (sigmas * self.num_train_timesteps).copy()
            self.timesteps = timesteps.astype(np.int64)
            self.sigmas = sigmas.astype(np.float64)
        else:
            self.timesteps = get_timesteps_linspace(num_inference_steps,
                                                    self.num_train_timesteps)
            idx = np.clip(self.timesteps, 0, self.num_train_timesteps - 1)
            self.sigmas = alphas_cumprod_to_sigmas(self.alphas_cumprod[idx])
        # append zero sigma for the final step
        self.sigmas = np.concatenate([self.sigmas, np.zeros(1, dtype=np.float64)])
        self._step_index = None
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0

    def _init_step_index(self, timestep):
        ts = int(timestep.item()) if isinstance(timestep, NBXTensor) else int(timestep)
        diffs = np.abs(self.timesteps - ts)
        self._step_index = int(diffs.argmin())

    @property
    def step_index(self):
        return self._step_index

    def _sigma_to_alpha_sigma_t(self, sigma: float):
        if self.use_flow_sigmas:
            return (1.0 - sigma), sigma
        alpha_t = 1.0 / (sigma ** 2 + 1.0) ** 0.5
        return alpha_t, sigma * alpha_t

    # ---- model-output conversion (NBXTensor) ----
    def convert_model_output(self, model_output: NBXTensor, sample: NBXTensor) -> NBXTensor:
        sigma = float(self.sigmas[self._step_index])
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        pt = self.prediction_type
        if pt == "epsilon":
            return (sample - model_output * sigma_t) * (1.0 / alpha_t)
        if pt == "v_prediction":
            return sample * alpha_t - model_output * sigma_t
        if pt == "sample":
            return model_output
        if pt in ("flow", "flow_prediction"):
            return sample - model_output * sigma_t
        raise RuntimeError(f"ZERO FALLBACK: unknown prediction_type '{pt}'")

    # ---- updates (scalar coeffs + NBXTensor) ----
    def _first_order(self, model_output: NBXTensor, sample: NBXTensor) -> NBXTensor:
        sigma_t_raw = float(self.sigmas[self._step_index + 1])
        sigma_s_raw = float(self.sigmas[self._step_index])
        if sigma_t_raw == 0:
            return model_output
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t_raw)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s_raw)
        h = (math.log(alpha_t) - math.log(sigma_t)) - (math.log(alpha_s) - math.log(sigma_s))
        if self.algorithm_type == "dpmsolver++":
            return sample * (sigma_t / sigma_s) - model_output * (alpha_t * (math.exp(-h) - 1.0))
        return sample * (alpha_t / alpha_s) - model_output * (sigma_t * (math.exp(h) - 1.0))

    def _second_order(self, model_output_list, sample: NBXTensor) -> NBXTensor:
        sigma_t_raw = float(self.sigmas[self._step_index + 1])
        sigma_s0_raw = float(self.sigmas[self._step_index])
        sigma_s1_raw = float(self.sigmas[self._step_index - 1])
        if sigma_t_raw == 0:
            return model_output_list[-1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t_raw)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0_raw)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1_raw)
        lambda_t = math.log(alpha_t) - math.log(sigma_t)
        lambda_s0 = math.log(alpha_s0) - math.log(sigma_s0)
        lambda_s1 = math.log(alpha_s1) - math.log(sigma_s1)
        m0 = model_output_list[-1]
        m1 = model_output_list[-2]
        h = lambda_t - lambda_s0
        h_0 = lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0 = m0
        D1 = (m0 - m1) * (1.0 / r0)
        if self.algorithm_type == "dpmsolver++":
            coef = alpha_t * (math.exp(-h) - 1.0)
            base = sample * (sigma_t / sigma_s0)
            if self.solver_type == "midpoint":
                return base - D0 * coef - D1 * (0.5 * coef)
            if self.solver_type == "heun":
                return base - D0 * coef - D1 * (alpha_t * ((math.exp(-h) - 1.0) / h + 1.0))
            raise RuntimeError(f"ZERO FALLBACK: unknown solver_type '{self.solver_type}'")
        coef = sigma_t * (math.exp(h) - 1.0)
        base = sample * (alpha_t / alpha_s0)
        if self.solver_type == "midpoint":
            return base - D0 * coef - D1 * (0.5 * coef)
        if self.solver_type == "heun":
            return base - D0 * coef - D1 * (sigma_t * ((math.exp(h) - 1.0) / h - 1.0))
        raise RuntimeError(f"ZERO FALLBACK: unknown solver_type '{self.solver_type}'")

    # ---- step ----
    def step(self, model_output: NBXTensor, timestep, sample: NBXTensor,
             return_dict: bool = True, **kwargs):
        if self.num_inference_steps is None:
            raise RuntimeError("ZERO FALLBACK: set_timesteps() before step()")
        # variance prediction (2x channels) → keep first half
        if model_output.shape[1] == sample.shape[1] * 2:
            model_output = model_output[:, :sample.shape[1]]
        if self._step_index is None:
            self._init_step_index(timestep)

        n = len(self.timesteps)
        lower_order_final = (self._step_index == n - 1) and (
            self.final_sigmas_type == "zero"
            or (self.lower_order_final and n < 15))
        lower_order_second = ((self._step_index == n - 2)
                              and self.lower_order_final and n < 15)

        model_output = self.convert_model_output(model_output, sample)
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        sample = sample.to(NBXDtype.float32)
        if self.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev = self._first_order(model_output, sample)
        else:
            valid = [o for o in self.model_outputs if o is not None]
            prev = self._second_order(valid, sample)

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        prev = prev.to(model_output.nbx_dtype)
        self._step_index += 1
        return {"prev_sample": prev} if return_dict else prev

    def scale_model_input(self, sample: NBXTensor, timestep) -> NBXTensor:
        return sample

    @property
    def init_noise_sigma(self) -> float:
        return 1.0
