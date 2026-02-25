# core/scheduler/consistency/lcm.py
"""
LCM (Latent Consistency Model) Scheduler.

Enables 1-8 step generation with distilled models.

Song et al., 2023 - Latent Consistency Models

ZERO FALLBACK: All required values from NBX config.
"""

import torch
from typing import Dict, Any, Union, Optional

from ..base import DiffusionSchedulerBase, PredictionType
from ..config import SchedulerConfig
from ..utils.noise_schedules import get_beta_schedule, betas_to_alphas


class LCMScheduler(DiffusionSchedulerBase):
    """
    Latent Consistency Model Scheduler.

    Enables few-step (1-8) generation with LCM-distilled models.
    Uses consistency distillation training to enable direct prediction.

    ZERO FALLBACK: All required values MUST come from NBX config.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize from config dict.

        ZERO FALLBACK: Required keys must be in config.
        """
        # ZERO FALLBACK validation
        validated = SchedulerConfig.validate(config, "LCMScheduler")

        # REQUIRED values
        self.num_train_timesteps = validated["num_train_timesteps"]
        beta_start = validated["beta_start"]
        beta_end = validated["beta_end"]
        beta_schedule = validated["beta_schedule"]
        self.prediction_type = PredictionType(validated["prediction_type"])

        # LCM-specific optional values
        self.original_inference_steps = validated.get("original_inference_steps", 50)
        self.clip_sample = validated.get("clip_sample", False)
        self.clip_sample_range = validated.get("clip_sample_range", 1.0)
        self.thresholding = validated["thresholding"]
        self.sample_max_value = validated["sample_max_value"]
        # ZERO HARDCODE: Get from config instead of hardcoding 0.995
        self.dynamic_thresholding_ratio = validated.get("dynamic_thresholding_ratio", 0.995)

        # Compute noise schedule
        betas = get_beta_schedule(
            beta_schedule,
            self.num_train_timesteps,
            beta_start,
            beta_end
        )
        self.alphas, self.alphas_cumprod = betas_to_alphas(betas)
        self.final_alpha_cumprod = torch.tensor(1.0)

        # State
        self.num_inference_steps: Optional[int] = None
        self.timesteps: Optional[torch.Tensor] = None
        self._step_index: Optional[int] = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[torch.device] = None,
        strength: float = 1.0,
        **kwargs
    ) -> None:
        """
        Set timesteps for LCM inference.

        LCM uses specific timestep selection based on original_inference_steps.
        """
        self.num_inference_steps = num_inference_steps

        # LCM timestep formula
        # Selects evenly spaced timesteps from the original schedule
        c = self.num_train_timesteps // self.original_inference_steps
        lcm_origin_timesteps = (
            torch.arange(1, num_inference_steps + 1) *
            (self.original_inference_steps // num_inference_steps) * c - 1
        )

        # Handle strength (for img2img)
        if strength < 1.0:
            skipped_steps = int(num_inference_steps * (1 - strength))
            lcm_origin_timesteps = lcm_origin_timesteps[skipped_steps:]

        # Reverse to go from high noise to low noise
        timesteps = lcm_origin_timesteps.flip(0).long()

        if device is not None:
            timesteps = timesteps.to(device)
            self.alphas_cumprod = self.alphas_cumprod.to(device)

        self.timesteps = timesteps

        self._step_index = None

    def _init_step_index(self, timestep: Union[int, torch.Tensor]) -> None:
        """Initialize step index from timestep."""
        assert self.timesteps is not None
        if isinstance(timestep, torch.Tensor):
            timestep_val = timestep.item()
        else:
            timestep_val = timestep

        step_indices = (self.timesteps == timestep_val).nonzero()
        if step_indices.numel() == 0:
            step_indices = (self.timesteps - timestep_val).abs().argmin()
            self._step_index = int(step_indices.item())
        else:
            self._step_index = int(step_indices[0].item())

    @property
    def step_index(self):
        return self._step_index

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """Dynamic thresholding."""
        dtype = sample.dtype
        batch_size, channels = sample.shape[:2]
        sample = sample.reshape(batch_size, channels, -1).float()

        abs_sample = sample.abs()
        s = torch.quantile(abs_sample, self.dynamic_thresholding_ratio, dim=-1)
        s = torch.clamp(s, min=1.0, max=self.sample_max_value)
        s = s.unsqueeze(-1)

        sample = torch.clamp(sample, -s, s) / s
        sample = sample.reshape(batch_size, channels, *sample.shape[2:])
        return sample.to(dtype)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        LCM step.

        LCM directly predicts the clean sample, similar to consistency models.
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

        if isinstance(timestep, torch.Tensor):
            timestep_val = timestep.item()
        else:
            timestep_val = timestep

        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[int(timestep_val)]

        # Previous timestep
        assert self._step_index is not None
        assert self.timesteps is not None
        if self._step_index == len(self.timesteps) - 1:
            prev_timestep = 0
        else:
            prev_timestep = int(self.timesteps[self._step_index + 1].item())

        alpha_prod_t_prev = (
            self.alphas_cumprod[int(prev_timestep)]
            if prev_timestep > 0
            else self.final_alpha_cumprod
        )

        # Ensure on same device
        if alpha_prod_t.device != sample.device:
            alpha_prod_t = alpha_prod_t.to(sample.device)
            alpha_prod_t_prev = alpha_prod_t_prev.to(sample.device)

        # Compute sqrt values
        sqrt_alpha_prod_t = alpha_prod_t ** 0.5
        sqrt_one_minus_alpha_prod_t = (1 - alpha_prod_t) ** 0.5

        # LCM predicts x0 from noise prediction
        if self.prediction_type == PredictionType.EPSILON:
            pred_x0 = (sample - sqrt_one_minus_alpha_prod_t * model_output) / sqrt_alpha_prod_t.clamp(min=1e-8)
        elif self.prediction_type == PredictionType.V_PREDICTION:
            pred_x0 = sqrt_alpha_prod_t * sample - sqrt_one_minus_alpha_prod_t * model_output
        else:
            pred_x0 = model_output

        # Thresholding / clipping
        if self.thresholding:
            pred_x0 = self._threshold_sample(pred_x0)
        elif self.clip_sample:
            pred_x0 = pred_x0.clamp(-self.clip_sample_range, self.clip_sample_range)

        # Get noise for next step
        sqrt_alpha_prod_t_prev = alpha_prod_t_prev ** 0.5
        sqrt_one_minus_alpha_prod_t_prev = (1 - alpha_prod_t_prev) ** 0.5

        # LCM formula: directly go to next timestep's noise level
        # x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1-alpha_{t-1}) * epsilon
        # For LCM, we derive epsilon from the prediction
        if self.prediction_type == PredictionType.EPSILON:
            pred_epsilon = model_output
        else:
            pred_epsilon = (sample - sqrt_alpha_prod_t * pred_x0) / sqrt_one_minus_alpha_prod_t.clamp(min=1e-8)

        prev_sample = sqrt_alpha_prod_t_prev * pred_x0 + sqrt_one_minus_alpha_prod_t_prev * pred_epsilon

        # Increment step index
        assert self._step_index is not None
        self._step_index += 1

        if return_dict:
            return {"prev_sample": prev_sample, "denoised": pred_x0}
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
        """LCM doesn't scale input."""
        return sample

    @property
    def init_noise_sigma(self) -> float:
        """Initial noise sigma."""
        return 1.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LCMScheduler":
        """Create scheduler from NBX config."""
        return cls(config)
