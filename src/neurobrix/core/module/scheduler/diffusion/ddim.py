# core/scheduler/diffusion/ddim.py
"""
DDIM Scheduler - Denoising Diffusion Implicit Models.

Song et al., 2020 - https://arxiv.org/abs/2010.02502

Deterministic sampling with eta=0, stochastic with eta>0.
Simple, stable, reference implementation.

ZERO FALLBACK: All required values from NBX config.
"""

import torch
from typing import Dict, Any, Union, Optional

from ..base import DiffusionSchedulerBase, PredictionType
from ..config import SchedulerConfig
from ..utils.noise_schedules import get_beta_schedule, betas_to_alphas
from ..utils.timestep_utils import get_timesteps
from ..utils.helpers import init_step_index, threshold_sample


class DDIMScheduler(DiffusionSchedulerBase):
    """
    DDIM Scheduler (Song et al., 2020).

    Deterministic sampling with eta=0.
    Simple, stable, reference implementation.

    ZERO FALLBACK: All required values MUST come from NBX config.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize from config dict.

        ZERO FALLBACK: Required keys must be in config.
        """
        # ZERO FALLBACK validation
        validated = SchedulerConfig.validate(config, "DDIMScheduler")

        # REQUIRED values - guaranteed present after validation
        self.num_train_timesteps = validated["num_train_timesteps"]
        beta_start = validated["beta_start"]
        beta_end = validated["beta_end"]
        beta_schedule = validated["beta_schedule"]
        self.prediction_type = PredictionType(validated["prediction_type"])
        self.timestep_spacing = validated["timestep_spacing"]

        # OPTIONAL values (safe defaults from validator)
        self.clip_sample = validated.get("clip_sample", True)
        self.clip_sample_range = validated.get("clip_sample_range", 1.0)
        self.eta = validated.get("eta", 0.0)  # 0 = deterministic DDIM
        self.thresholding = validated["thresholding"]
        self.sample_max_value = validated["sample_max_value"]
        self.steps_offset = validated["steps_offset"]
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
        **kwargs
    ) -> None:
        """Set timesteps for inference."""
        self.num_inference_steps = num_inference_steps
        self.timesteps = get_timesteps(
            self.timestep_spacing,
            num_inference_steps,
            self.num_train_timesteps
        )

        if device is not None:
            self.timesteps = self.timesteps.to(device)
            self.alphas_cumprod = self.alphas_cumprod.to(device)

        self._step_index = None

    def _init_step_index(self, timestep: Union[int, torch.Tensor]) -> None:
        """Initialize step index from timestep using shared helper."""
        assert self.timesteps is not None
        self._step_index = init_step_index(self.timesteps, timestep)

    @property
    def step_index(self):
        return self._step_index

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """Dynamic thresholding from Imagen using shared helper."""
        return threshold_sample(sample, self.dynamic_thresholding_ratio, self.sample_max_value)

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
        DDIM step.

        Matches diffusers DDIMScheduler.step() behavior.
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

        # Previous timestep
        assert self._step_index is not None
        assert self.timesteps is not None
        if self._step_index == len(self.timesteps) - 1:
            prev_timestep = 0
        else:
            prev_timestep = int(self.timesteps[self._step_index + 1].item())

        # Alpha values
        alpha_prod_t = self.alphas_cumprod[int(timestep_val)]
        alpha_prod_t_prev = (
            self.alphas_cumprod[int(prev_timestep)]
            if prev_timestep > 0
            else self.final_alpha_cumprod
        )

        # Ensure on same device
        if alpha_prod_t.device != sample.device:
            alpha_prod_t = alpha_prod_t.to(sample.device)
            alpha_prod_t_prev = alpha_prod_t_prev.to(sample.device)

        # Sqrt values
        sqrt_alpha_prod_t = alpha_prod_t ** 0.5
        sqrt_one_minus_alpha_prod_t = (1 - alpha_prod_t) ** 0.5
        sqrt_alpha_prod_t_prev = alpha_prod_t_prev ** 0.5
        sqrt_one_minus_alpha_prod_t_prev = (1 - alpha_prod_t_prev) ** 0.5

        # Predict x0 based on prediction type
        if self.prediction_type == PredictionType.EPSILON:
            pred_x0 = (sample - sqrt_one_minus_alpha_prod_t * model_output) / sqrt_alpha_prod_t.clamp(min=1e-8)
            pred_epsilon = model_output
        elif self.prediction_type == PredictionType.V_PREDICTION:
            pred_x0 = sqrt_alpha_prod_t * sample - sqrt_one_minus_alpha_prod_t * model_output
            pred_epsilon = sqrt_alpha_prod_t * model_output + sqrt_one_minus_alpha_prod_t * sample
        else:  # SAMPLE
            pred_x0 = model_output
            pred_epsilon = (sample - sqrt_alpha_prod_t * pred_x0) / sqrt_one_minus_alpha_prod_t.clamp(min=1e-8)

        # Thresholding
        if self.thresholding:
            pred_x0 = self._threshold_sample(pred_x0)
        elif self.clip_sample:
            pred_x0 = pred_x0.clamp(-self.clip_sample_range, self.clip_sample_range)

        # DDIM formula
        # sigma = eta * sqrt((1-alpha_prev)/(1-alpha) * (1 - alpha/alpha_prev))
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        variance = variance.clamp(min=0)
        sigma = self.eta * (variance ** 0.5)

        # Direction pointing to x_t
        pred_direction = (1 - alpha_prod_t_prev - sigma ** 2).clamp(min=0) ** 0.5 * pred_epsilon

        # x_{t-1}
        prev_sample = sqrt_alpha_prod_t_prev * pred_x0 + pred_direction

        # Add noise if eta > 0
        if self.eta > 0:
            noise = torch.randn(
                sample.shape,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype
            )
            prev_sample = prev_sample + sigma * noise

        # Increment step index
        assert self._step_index is not None
        self._step_index += 1

        if return_dict:
            return {"prev_sample": prev_sample, "pred_original_sample": pred_x0}
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
        """DDIM doesn't scale input."""
        return sample

    @property
    def init_noise_sigma(self) -> float:
        """Initial noise sigma."""
        return 1.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DDIMScheduler":
        """Create scheduler from NBX config."""
        return cls(config)
