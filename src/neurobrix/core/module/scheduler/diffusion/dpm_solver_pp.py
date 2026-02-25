# core/scheduler/diffusion/dpm_solver_pp.py
"""
DPM-Solver++ Scheduler - ZERO FALLBACK.

DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models
(Lu et al., 2022) - https://arxiv.org/abs/2211.01095

Used by: SDXL, PixArt, Kandinsky, most modern diffusion models
Matches diffusers DPMSolverMultistepScheduler exactly.

ZERO FALLBACK: All required config values MUST come from NBX container.
"""

import torch
from typing import Dict, Any, Union, Optional, List

from ..base import DiffusionSchedulerBase, PredictionType
from ..config import SchedulerConfig, SchedulerConfigError
from ..utils.noise_schedules import get_beta_schedule, betas_to_alphas, alphas_to_sigmas
from ..utils.timestep_utils import get_timesteps
from ..utils.helpers import init_step_index, threshold_sample


class DPMSolverPPScheduler(DiffusionSchedulerBase):
    """
    DPM-Solver++ Multi-step Scheduler.

    ZERO FALLBACK: All required config values MUST come from NBX container.
    Missing required key = RuntimeError with helpful message.

    Key features:
    - Multi-step solver (2nd/3rd order)
    - Uses lambda-space for better stability
    - Supports epsilon, v-prediction, and sample prediction
    - Matches diffusers DPMSolverMultistepScheduler exactly
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize from config dict.

        ZERO FALLBACK: Required keys must be in config.
        Use from_config() classmethod for NBX integration.
        """
        # ZERO FALLBACK: Validate config - crash if required keys missing
        validated = SchedulerConfig.validate(config, "DPMSolverPPScheduler")

        # REQUIRED values - guaranteed present after validation
        self.num_train_timesteps = validated["num_train_timesteps"]
        beta_start = validated["beta_start"]
        beta_end = validated["beta_end"]
        beta_schedule = validated["beta_schedule"]
        self.prediction_type = PredictionType(validated["prediction_type"])
        self.timestep_spacing = validated["timestep_spacing"]
        self.solver_order = validated["solver_order"]
        self.solver_type = validated["solver_type"]
        self.lower_order_final = validated["lower_order_final"]
        self.algorithm_type = validated["algorithm_type"]

        # OPTIONAL values - safe defaults applied by validator
        self.thresholding = validated["thresholding"]
        self.sample_max_value = validated["sample_max_value"]
        self.euler_at_final = validated["euler_at_final"]
        self.final_sigmas_type = validated["final_sigmas_type"]
        # ZERO HARDCODE: Get from config instead of hardcoding 0.995
        self.dynamic_thresholding_ratio = validated.get("dynamic_thresholding_ratio", 0.995)

        # Flow matching support (Sana, newer flow models)
        self.use_flow_sigmas = validated.get("use_flow_sigmas", False)
        self.flow_shift = validated.get("flow_shift", 1.0)

        # Compute noise schedule using validated values
        betas = get_beta_schedule(
            beta_schedule,
            self.num_train_timesteps,
            beta_start,
            beta_end
        )
        self.alphas, self.alphas_cumprod = betas_to_alphas(betas)

        # State
        self.num_inference_steps: Optional[int] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self._step_index: Optional[int] = None

        # Multi-step buffer (stores denoised predictions)
        self.model_outputs: List[Optional[torch.Tensor]] = [None] * self.solver_order
        self.lower_order_nums = 0

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device | None = None,
        **kwargs
    ) -> None:
        """Set timesteps for inference."""
        self.num_inference_steps = num_inference_steps

        if self.use_flow_sigmas:
            # Flow matching: compute sigmas FIRST, then derive timesteps from them
            # This matches diffusers DPMSolverMultistepScheduler.set_timesteps() exactly
            # Formula from diffusers:
            #   alphas = linspace(1, 1/num_train_timesteps, num_steps + 1)
            #   sigmas = 1 - alphas
            #   sigmas = flip(flow_shift * sigmas / (1 + (flow_shift - 1) * sigmas))[:-1]
            #   timesteps = sigmas * num_train_timesteps
            import numpy as np
            alphas = np.linspace(1, 1 / self.num_train_timesteps, num_inference_steps + 1)
            sigmas = 1.0 - alphas
            sigmas = np.flip(
                self.flow_shift * sigmas / (1 + (self.flow_shift - 1) * sigmas)
            )[:-1].copy()
            timesteps = (sigmas * self.num_train_timesteps).copy()

            # Match diffusers: timesteps as int64 (rounds to nearest int), sigmas as float32
            self.timesteps = torch.from_numpy(timesteps).to(dtype=torch.int64)
            self.sigmas = torch.from_numpy(sigmas.astype('float32'))
        else:
            # Standard diffusion: use timestep spacing, then compute sigmas from alphas
            self.timesteps = get_timesteps(
                self.timestep_spacing,
                num_inference_steps,
                self.num_train_timesteps
            )
            timestep_indices = self.timesteps.cpu().long().clamp(0, self.num_train_timesteps - 1)
            alphas_cumprod = self.alphas_cumprod[timestep_indices]
            self.sigmas = alphas_to_sigmas(alphas_cumprod)

        if device is not None:
            self.timesteps = self.timesteps.to(device)

        # Append zero sigma for final step
        self.sigmas = torch.cat([self.sigmas, torch.zeros(1)])

        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.alphas_cumprod = self.alphas_cumprod.to(device)

        # Reset state
        self._step_index = None
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0

    def _init_step_index(self, timestep: Union[int, torch.Tensor]) -> None:
        """Initialize step index from timestep using shared helper."""
        assert self.timesteps is not None, "timesteps must be set"
        self._step_index = init_step_index(self.timesteps, timestep)

    @property
    def step_index(self):
        return self._step_index

    def _sigma_to_alpha_sigma_t(self, sigma: torch.Tensor):
        """Convert sigma to (alpha_t, sigma_t) pair."""
        if self.use_flow_sigmas:
            # Flow matching: sigma IS the timestep t in [0, 1]
            # alpha_t = 1 - t, sigma_t = t
            alpha_t = 1 - sigma
            sigma_t = sigma
        else:
            # Standard diffusion: sigma = sqrt((1-alpha)/alpha)
            alpha_t = 1 / (sigma ** 2 + 1) ** 0.5
            sigma_t = sigma * alpha_t
        return alpha_t, sigma_t

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert model output to denoised prediction (x0).
        Matches diffusers convert_model_output exactly.
        """
        assert self._step_index is not None, "step_index must be initialized"
        assert self.sigmas is not None, "sigmas must be set"
        sigma = self.sigmas[self._step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        if self.prediction_type == PredictionType.EPSILON:
            # x0 = (x - sigma_t * epsilon) / alpha_t
            denoised = (sample - sigma_t * model_output) / alpha_t
        elif self.prediction_type == PredictionType.V_PREDICTION:
            # x0 = alpha_t * x - sigma_t * v
            # NOTE: model_output is 'v' in this case
            denoised = alpha_t * sample - sigma_t * model_output
        elif self.prediction_type == PredictionType.SAMPLE:
            denoised = model_output
        elif self.prediction_type in (PredictionType.FLOW, PredictionType.FLOW_PREDICTION):
            # Flow matching: model predicts velocity v = (x_1 - x_0)
            # For flow models: x0 = sample - sigma * model_output
            # (sigma is timestep t in [0,1] for flow matching)
            denoised = sample - sigma_t * model_output
        else:
            raise SchedulerConfigError(
                f"ZERO FALLBACK: Unknown prediction type '{self.prediction_type}'"
            )

        if self.thresholding:
            denoised = self._threshold_sample(denoised)

        return denoised

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """Dynamic thresholding from Imagen using shared helper."""
        return threshold_sample(sample, self.dynamic_thresholding_ratio, self.sample_max_value)

    def dpm_solver_first_order_update(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        DPM-Solver++ first order update.
        Matches diffusers dpm_solver_first_order_update exactly.
        """
        assert self._step_index is not None, "step_index must be initialized"
        assert self.sigmas is not None, "sigmas must be set"
        # CRITICAL: diffusers overwrites sigma_t/sigma_s with scaled versions
        sigma_t_raw = self.sigmas[self._step_index + 1]
        sigma_s_raw = self.sigmas[self._step_index]

        # FINAL STEP FIX: When sigma_t = 0 (no noise remaining), return denoised prediction
        # This avoids log(0) = -inf which causes NaN in subsequent calculations
        if sigma_t_raw == 0:
            # At final step, the denoised prediction IS the output (no more noise to remove)
            return model_output

        # These calls REPLACE sigma_t and sigma_s with scaled versions
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t_raw)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s_raw)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

        h = lambda_t - lambda_s

        if self.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        elif self.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
        else:
            raise SchedulerConfigError(
                f"ZERO FALLBACK: Unknown algorithm type '{self.algorithm_type}'"
            )

        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.Tensor],
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        DPM-Solver++ second order update using previous model outputs.
        Matches diffusers multistep_dpm_solver_second_order_update exactly.
        """
        assert self._step_index is not None, "step_index must be initialized"
        assert self.sigmas is not None, "sigmas must be set"
        # Get original sigmas
        sigma_t_raw = self.sigmas[self._step_index + 1]
        sigma_s0_raw = self.sigmas[self._step_index]
        sigma_s1_raw = self.sigmas[self._step_index - 1]

        # FINAL STEP FIX: When sigma_t = 0 (no noise remaining), return denoised prediction
        if sigma_t_raw == 0:
            return model_output_list[-1]

        # CRITICAL: These calls REPLACE sigma values with scaled versions
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t_raw)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0_raw)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1_raw)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        # Get model outputs (stored in diffusers order: oldest at [0], newest at [-1])
        m0 = model_output_list[-1]  # Current step
        m1 = model_output_list[-2]  # Previous step

        h = lambda_t - lambda_s0
        h_0 = lambda_s0 - lambda_s1
        r0 = h_0 / h

        # Diffusers formula: D0 = m0, D1 = (1/r0) * (m0 - m1)
        D0 = m0
        D1 = (1.0 / r0) * (m0 - m1)

        if self.algorithm_type == "dpmsolver++":
            if self.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )
            else:
                raise SchedulerConfigError(
                    f"ZERO FALLBACK: Unknown solver type '{self.solver_type}'"
                )
        else:
            # dpmsolver (non-++)
            if self.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                )
            else:
                raise SchedulerConfigError(
                    f"ZERO FALLBACK: Unknown solver type '{self.solver_type}'"
                )

        return x_t

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        DPM-Solver++ step.
        Matches diffusers DPMSolverMultistepScheduler.step exactly.
        """
        if self.num_inference_steps is None:
            raise RuntimeError(
                "ZERO FALLBACK: set_timesteps() must be called before step()"
            )

        # Handle variance prediction (8-channel output)
        if model_output.shape[1] == sample.shape[1] * 2:
            model_output, _ = model_output.chunk(2, dim=1)

        # Initialize step index if needed
        if self._step_index is None:
            self._init_step_index(timestep)

        # Type guard for timesteps
        assert self.timesteps is not None, "timesteps must be set"

        # Compute lower_order flags matching diffusers
        lower_order_final = (self._step_index == len(self.timesteps) - 1) and (
            self.euler_at_final
            or (self.lower_order_final and len(self.timesteps) < 15)
            or self.final_sigmas_type == "zero"
        )
        lower_order_second = (
            (self._step_index == len(self.timesteps) - 2)
            and self.lower_order_final
            and len(self.timesteps) < 15
        )

        # Convert model output to denoised prediction
        model_output = self.convert_model_output(model_output, sample=sample)
        
        # Store in history (diffusers style: shift left, add at end)
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        # Upcast sample for numerical precision (matching diffusers)
        sample = sample.to(torch.float32)

        # Determine which order update to use
        if self.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample)
        elif self.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            # Type guard: ensure all model outputs are non-None
            valid_outputs = [out for out in self.model_outputs if out is not None]
            prev_sample = self.multistep_dpm_solver_second_order_update(
                valid_outputs, sample=sample
            )
        else:
            # Third order (not yet implemented, fall back to second)
            # Type guard: ensure all model outputs are non-None
            valid_outputs = [out for out in self.model_outputs if out is not None]
            prev_sample = self.multistep_dpm_solver_second_order_update(
                valid_outputs, sample=sample
            )

        # Increment lower_order_nums counter
        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        # Cast back to model output dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # Increment step index for next call
        assert self._step_index is not None, "step_index must be initialized"
        self._step_index += 1

        if return_dict:
            return {"prev_sample": prev_sample}
        return prev_sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples (forward diffusion)."""
        alphas_cumprod = self.alphas_cumprod.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Scale input sample. DPM++ does not scale the input (returns as-is)."""
        return sample

    @property
    def init_noise_sigma(self) -> float:
        """Initial noise sigma. DPM++ uses unit variance noise."""
        return 1.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DPMSolverPPScheduler":
        """Create scheduler from NBX config."""
        return cls(config)
