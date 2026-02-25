# core/scheduler/flow/flow_euler.py
"""
Flow Euler Scheduler for Flux and SD3.

Implements flow matching ODE solver using Euler method.

CRITICAL: shift parameter differs between models:
- Flux: shift=3.0
- SD3: shift=1.0

ZERO FALLBACK: shift MUST come from NBX config.
"""

import torch
from typing import Dict, Any, Union, Optional

from ..base import FlowSchedulerBase
from ..config import FlowSchedulerConfig, SchedulerConfigError


class FlowEulerScheduler(FlowSchedulerBase):
    """
    Flow Matching Euler Scheduler.

    For flow matching: dx/dt = v(x, t)
    Integration: x_{t-dt} = x_t - dt * v

    Used by: Flux, SD3, Rectified Flow models.

    ZERO FALLBACK: shift parameter MUST come from NBX config.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize from config dict.

        ZERO FALLBACK: Required keys must be in config.
        """
        # ZERO FALLBACK validation for flow scheduler
        validated = FlowSchedulerConfig.validate(config, "FlowEulerScheduler")

        # REQUIRED values - CRITICAL for correct output
        self.num_train_timesteps = validated["num_train_timesteps"]
        self.shift = validated["shift"]  # CRITICAL: 3.0 for Flux, 1.0 for SD3

        # OPTIONAL values
        self.base_image_seq_len = validated.get("base_image_seq_len", 256)
        self.max_image_seq_len = validated.get("max_image_seq_len", 4096)
        self.base_shift = validated.get("base_shift", 0.5)
        self.max_shift = validated.get("max_shift", 1.15)
        self.invert_sigmas = validated.get("invert_sigmas", False)

        # State
        self.num_inference_steps: Optional[int] = None
        self.timesteps: torch.Tensor = torch.empty(0)
        self.sigmas = None
        self._step_index = 0

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> None:
        """
        Set timesteps for inference.

        Flow models use timesteps from 1 → 0.
        """
        self.num_inference_steps = num_inference_steps

        # Dynamic shift based on image size (for Flux)
        image_seq_len = kwargs.get("image_seq_len", None)
        if image_seq_len is not None:
            # Compute dynamic shift
            mu = self._calculate_mu(image_seq_len)
        else:
            mu = self.shift

        # Flow timesteps: 1 → 0
        timesteps = torch.linspace(1, 0, num_inference_steps + 1)[:-1]

        # Apply shift (time-shifting for better denoising)
        if mu != 1.0:
            timesteps = self._apply_shift(timesteps, mu)

        self.timesteps = timesteps
        self.sigmas = timesteps.clone()  # For flow, sigma = t

        if device is not None:
            self.timesteps = self.timesteps.to(device)
            self.sigmas = self.sigmas.to(device)

        self._step_index = 0

    def _calculate_mu(self, image_seq_len: int) -> float:
        """Calculate dynamic shift (mu) based on image sequence length."""
        # Linear interpolation between base_shift and max_shift
        m = (self.max_shift - self.base_shift) / (self.max_image_seq_len - self.base_image_seq_len)
        b = self.base_shift - m * self.base_image_seq_len
        mu = image_seq_len * m + b
        return mu

    def _apply_shift(self, timesteps: torch.Tensor, mu: float) -> torch.Tensor:
        """Apply time shift transformation."""
        # shift * t / (1 + (shift - 1) * t)
        return mu * timesteps / (1 + (mu - 1) * timesteps)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Flow Euler step.

        For flow matching: x_{t-1} = x_t + dt * v
        where dt = t_{i+1} - t_i (negative since going 1→0)
        """
        if self.num_inference_steps is None:
            raise RuntimeError(
                "ZERO FALLBACK: set_timesteps() must be called before step()"
            )

        if isinstance(timestep, torch.Tensor):
            t = timestep.item()
        else:
            t = timestep

        # Get next timestep
        if self._step_index < len(self.timesteps) - 1:
            t_next = self.timesteps[self._step_index + 1].item()
        else:
            t_next = 0.0

        # dt is negative (going from 1 to 0)
        dt = t_next - t

        # Euler step: x_{t-1} = x_t + dt * v
        # Since dt is negative and v points from noise to data,
        # this moves us from noise towards data
        prev_sample = sample + dt * model_output

        self._step_index += 1

        if return_dict:
            return {"prev_sample": prev_sample}
        return prev_sample

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """
        Scale model input.

        Flow models typically don't scale input.
        """
        return sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples.

        For flow matching: x_t = (1-t) * x_0 + t * noise
        """
        if isinstance(timesteps, torch.Tensor) and timesteps.dim() == 0:
            t = timesteps.item()
        elif isinstance(timesteps, (int, float)):
            t = timesteps
        else:
            t = timesteps[0].item()

        # Ensure t is a tensor for broadcasting
        t = torch.tensor(t, device=original_samples.device, dtype=original_samples.dtype)
        while t.dim() < original_samples.dim():
            t = t.unsqueeze(-1)

        # Flow interpolation: x_t = (1-t) * x_0 + t * noise
        noisy_samples = (1 - t) * original_samples + t * noise
        return noisy_samples

    @property
    def init_noise_sigma(self) -> float:
        """Initial noise sigma (1.0 for flow matching)."""
        return 1.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FlowEulerScheduler":
        """Create scheduler from NBX config."""
        return cls(config)


class RectifiedFlowScheduler(FlowSchedulerBase):
    """
    Rectified Flow Scheduler.

    Direct path ODE from noise to data.
    Used by: Rectified Flow models, InstaFlow.

    ZERO FALLBACK: All required values from NBX config.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize from config dict."""
        validated = FlowSchedulerConfig.validate(config, "RectifiedFlowScheduler")

        self.num_train_timesteps = validated["num_train_timesteps"]
        self.shift = validated.get("shift", 1.0)

        self.num_inference_steps: Optional[int] = None
        self.timesteps: torch.Tensor = torch.empty(0)
        self._step_index = 0

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> None:
        """Set timesteps for inference."""
        self.num_inference_steps = num_inference_steps

        # Linear timesteps from 1 to 0
        self.timesteps = torch.linspace(1, 0, num_inference_steps + 1)[:-1]

        if device is not None:
            self.timesteps = self.timesteps.to(device)

        self._step_index = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Rectified flow step."""
        if self.num_inference_steps is None:
            raise RuntimeError(
                "ZERO FALLBACK: set_timesteps() must be called before step()"
            )

        if isinstance(timestep, torch.Tensor):
            t = timestep.item()
        else:
            t = timestep

        # Get dt
        if self._step_index < len(self.timesteps) - 1:
            t_next = self.timesteps[self._step_index + 1].item()
        else:
            t_next = 0.0

        dt = t_next - t

        # Euler step
        prev_sample = sample + dt * model_output

        self._step_index += 1

        if return_dict:
            return {"prev_sample": prev_sample}
        return prev_sample

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Scale model input (no scaling for rectified flow)."""
        return sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise using linear interpolation."""
        if isinstance(timesteps, torch.Tensor) and timesteps.dim() == 0:
            t = timesteps.item()
        elif isinstance(timesteps, (int, float)):
            t = timesteps
        else:
            t = timesteps[0].item()

        t = torch.tensor(t, device=original_samples.device, dtype=original_samples.dtype)
        while t.dim() < original_samples.dim():
            t = t.unsqueeze(-1)

        return (1 - t) * original_samples + t * noise

    @property
    def init_noise_sigma(self) -> float:
        """Initial noise sigma."""
        return 1.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RectifiedFlowScheduler":
        """Create scheduler from NBX config."""
        return cls(config)
