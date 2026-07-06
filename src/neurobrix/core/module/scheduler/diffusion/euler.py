# core/scheduler/diffusion/euler.py
"""
Euler Discrete Scheduler.

Simple first-order Euler method for diffusion models.
Supports Karras sigmas for improved sampling quality.

ZERO FALLBACK: All required values from NBX config.
"""

import torch
from typing import Dict, Any, Union, Optional

from ..base import DiffusionSchedulerBase, PredictionType
from ..config import SchedulerConfig
from ..utils.noise_schedules import get_beta_schedule, betas_to_alphas, alphas_to_sigmas
from ..utils.timestep_utils import get_timesteps, karras_sigmas
from ..utils.helpers import init_step_index


class EulerDiscreteScheduler(DiffusionSchedulerBase):
    """
    Euler Discrete Scheduler with optional Karras sigmas.

    Simple first-order Euler method. Fast and stable.

    ZERO FALLBACK: All required values MUST come from NBX config.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize from config dict.

        ZERO FALLBACK: Required keys must be in config.
        """
        # ZERO FALLBACK validation
        validated = SchedulerConfig.validate(config, "EulerDiscreteScheduler")

        # REQUIRED values
        self.num_train_timesteps = validated["num_train_timesteps"]
        beta_start = validated["beta_start"]
        beta_end = validated["beta_end"]
        beta_schedule = validated["beta_schedule"]
        self.prediction_type = PredictionType(validated["prediction_type"])
        self.timestep_spacing = validated["timestep_spacing"]

        # OPTIONAL values
        self.use_karras_sigmas = validated["use_karras_sigmas"]
        self.sigma_min = validated.get("sigma_min", 0.002)
        self.sigma_max = validated.get("sigma_max", 80.0)
        self.steps_offset = validated["steps_offset"]

        # Compute noise schedule
        betas = get_beta_schedule(
            beta_schedule,
            self.num_train_timesteps,
            beta_start,
            beta_end
        )
        self.alphas, self.alphas_cumprod = betas_to_alphas(betas)

        # Pre-compute sigmas for all timesteps
        self._all_sigmas = alphas_to_sigmas(self.alphas_cumprod)

        # State (timesteps type inherited from base class DiffusionSchedulerBase)
        self.num_inference_steps: Optional[int] = None
        self.timesteps = None  # Type from base: Optional[torch.Tensor]
        self.sigmas: Optional[torch.Tensor] = None
        self._step_index: Optional[int] = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> None:
        """Set timesteps for inference."""
        self.num_inference_steps = num_inference_steps

        if self.use_karras_sigmas:
            # Karras sigmas
            self.sigmas = karras_sigmas(
                num_inference_steps,
                self.sigma_min,
                self.sigma_max,
                rho=7.0  # Standard Karras rho
            )
            # Derive timesteps from sigmas
            timesteps = []
            for sigma in self.sigmas:
                idx = (self._all_sigmas - sigma).abs().argmin()
                timesteps.append(idx.item())
            self.timesteps = torch.tensor(timesteps)
        elif self.timestep_spacing == "linspace":
            # Euler-family linspace — VENDOR PARITY (diffusers EulerDiscrete /
            # EulerAncestral): FLOAT timesteps linspace(0, T-1, n) reversed,
            # INCLUDING t=0, with sigmas INTERPOLATED at the fractional
            # positions. The shared get_timesteps keeps the DPM/DDIM integer
            # n+1-drop-last convention — the two families genuinely differ in
            # diffusers, and conflating them was the Allegro #30 root cause:
            # the ladder never reached t=0 (last DiT call at t=125, sigma
            # 0.43 -> 0 in ONE Euler jump) and every intermediate sigma sat
            # 15-40% off the trained schedule -> structured residual noise
            # (the native "scanline"), at every step count and both engines.
            import numpy as _np
            _ts = _np.linspace(0, self.num_train_timesteps - 1,
                               num_inference_steps,
                               dtype=_np.float32)[::-1].copy()
            _all_sigmas = alphas_to_sigmas(self.alphas_cumprod).numpy()
            _sig = _np.interp(
                _ts, _np.arange(self.num_train_timesteps, dtype=_np.float32),
                _all_sigmas).astype(_np.float32)
            self.timesteps = torch.from_numpy(_ts)
            self.sigmas = torch.from_numpy(_sig)
        else:
            # Standard timesteps
            self.timesteps = get_timesteps(
                self.timestep_spacing,
                num_inference_steps,
                self.num_train_timesteps
            )
            # Compute sigmas from timesteps
            timestep_indices = self.timesteps.long().clamp(0, self.num_train_timesteps - 1)
            alphas_cumprod = self.alphas_cumprod[timestep_indices]
            self.sigmas = alphas_to_sigmas(alphas_cumprod)

        # Append zero sigma for final step
        self.sigmas = torch.cat([self.sigmas, torch.zeros(1)])

        if device is not None:
            assert self.timesteps is not None, "timesteps must be set before device transfer"
            self.timesteps = self.timesteps.to(device)
            self.sigmas = self.sigmas.to(device)
            self.alphas_cumprod = self.alphas_cumprod.to(device)

        self._step_index = None

    def _init_step_index(self, timestep: Union[int, torch.Tensor]) -> None:
        """Initialize step index from timestep using shared helper."""
        assert self.timesteps is not None, "timesteps must be initialized via set_timesteps()"
        self._step_index = init_step_index(self.timesteps, timestep)

    @property
    def step_index(self):
        return self._step_index

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
        Euler step.

        Matches diffusers EulerDiscreteScheduler.step() behavior.
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

        assert self._step_index is not None, "step_index must be initialized"
        assert self.sigmas is not None, "sigmas must be initialized via set_timesteps()"
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]

        # Upcast to fp32 for the step arithmetic — VENDOR PARITY (diffusers
        # "Upcast to avoid precision issues when computing prev_sample").
        # At low sigma, (sample - denoised) / sigma is a catastrophic
        # cancellation amplified by 1/sigma: in fp16 the rounding of
        # sigma*model_output inside `denoised` re-emerges divided by sigma
        # (error ~ eps_fp16 * |sample| / sigma — tens of percent of the
        # derivative in the final steps). prev_sample is cast back to the
        # model dtype at the end, exactly like the vendor.
        _out_dtype = model_output.dtype
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)

        # Convert model output to denoised prediction
        if self.prediction_type == PredictionType.EPSILON:
            # x0 = x - sigma * epsilon
            denoised = sample - sigma * model_output
            # derivative: (x - denoised) / sigma
            d = model_output
        elif self.prediction_type == PredictionType.V_PREDICTION:
            # For v-prediction: x0 = alpha * x - sigma * v
            # sigma = sqrt((1-alpha)/alpha), so alpha = 1/(1+sigma^2)
            alpha = 1.0 / (sigma ** 2 + 1).sqrt()
            denoised = alpha * sample - sigma * alpha * model_output
            d = (sample - denoised) / sigma.clamp(min=1e-8)
        else:  # SAMPLE prediction
            denoised = model_output
            d = (sample - denoised) / sigma.clamp(min=1e-8)

        # Euler step: x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * d
        dt = sigma_next - sigma
        prev_sample = (sample + dt * d).to(_out_dtype)
        denoised = denoised.to(_out_dtype)

        # Increment step index
        assert self._step_index is not None, "step_index must be initialized"
        self._step_index += 1

        if return_dict:
            return {"prev_sample": prev_sample, "denoised": denoised}
        return prev_sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples (forward diffusion)."""
        # Get sigmas for timesteps
        assert self.sigmas is not None, "sigmas must be initialized via set_timesteps()"
        sigmas = self.sigmas[:-1][timesteps].to(original_samples.device)

        while len(sigmas.shape) < len(original_samples.shape):
            sigmas = sigmas.unsqueeze(-1)

        # For Euler: x_t = x_0 + sigma * noise
        noisy_samples = original_samples + sigmas * noise
        return noisy_samples

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Scale model input.

        Euler scheduler scales by 1/sqrt(sigma^2 + 1).
        """
        if self._step_index is None:
            self._init_step_index(timestep)

        assert self._step_index is not None, "step_index must be initialized"
        assert self.sigmas is not None, "sigmas must be initialized via set_timesteps()"
        sigma = self.sigmas[self._step_index]
        sample = sample / ((sigma ** 2 + 1) ** 0.5)
        return sample

    @property
    def init_noise_sigma(self) -> float:
        """Initial noise sigma."""
        if self.sigmas is not None and len(self.sigmas) > 0:
            return self.sigmas[0].item()
        return 1.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EulerDiscreteScheduler":
        """Create scheduler from NBX config."""
        return cls(config)


class EulerAncestralDiscreteScheduler(DiffusionSchedulerBase):
    """
    Euler Ancestral Discrete Scheduler.

    Adds stochastic noise at each step for more diverse outputs.

    ZERO FALLBACK: All required values MUST come from NBX config.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize from config dict."""
        validated = SchedulerConfig.validate(config, "EulerAncestralDiscreteScheduler")

        # REQUIRED values
        self.num_train_timesteps = validated["num_train_timesteps"]
        beta_start = validated["beta_start"]
        beta_end = validated["beta_end"]
        beta_schedule = validated["beta_schedule"]
        self.prediction_type = PredictionType(validated["prediction_type"])
        self.timestep_spacing = validated["timestep_spacing"]

        # Compute noise schedule
        betas = get_beta_schedule(
            beta_schedule,
            self.num_train_timesteps,
            beta_start,
            beta_end
        )
        self.alphas, self.alphas_cumprod = betas_to_alphas(betas)
        self._all_sigmas = alphas_to_sigmas(self.alphas_cumprod)

        # State (timesteps type inherited from base class DiffusionSchedulerBase)
        self.num_inference_steps: Optional[int] = None
        self.timesteps = None  # Type from base: Optional[torch.Tensor]
        self.sigmas: Optional[torch.Tensor] = None
        self._step_index: Optional[int] = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> None:
        """Set timesteps for inference."""
        self.num_inference_steps = num_inference_steps
        if self.timestep_spacing == "linspace":
            # Euler-family linspace — VENDOR PARITY (see the EulerDiscrete
            # set_timesteps comment; the Allegro #30 root cause): FLOAT
            # timesteps down to t=0 inclusive, sigmas interpolated at the
            # fractional positions.
            import numpy as _np
            _ts = _np.linspace(0, self.num_train_timesteps - 1,
                               num_inference_steps,
                               dtype=_np.float32)[::-1].copy()
            _all_sigmas = alphas_to_sigmas(self.alphas_cumprod).numpy()
            _sig = _np.interp(
                _ts, _np.arange(self.num_train_timesteps, dtype=_np.float32),
                _all_sigmas).astype(_np.float32)
            self.timesteps = torch.from_numpy(_ts)
            self.sigmas = torch.from_numpy(_sig)
        else:
            self.timesteps = get_timesteps(
                self.timestep_spacing,
                num_inference_steps,
                self.num_train_timesteps
            )
            timestep_indices = self.timesteps.long().clamp(0, self.num_train_timesteps - 1)
            alphas_cumprod = self.alphas_cumprod[timestep_indices]
            self.sigmas = alphas_to_sigmas(alphas_cumprod)
        self.sigmas = torch.cat([self.sigmas, torch.zeros(1)])

        if device is not None:
            assert self.timesteps is not None, "timesteps must be set before device transfer"
            self.timesteps = self.timesteps.to(device)
            self.sigmas = self.sigmas.to(device)
            self.alphas_cumprod = self.alphas_cumprod.to(device)

        self._step_index = None

    def _init_step_index(self, timestep: Union[int, torch.Tensor]) -> None:
        """Initialize step index from timestep using shared helper."""
        assert self.timesteps is not None, "timesteps must be initialized via set_timesteps()"
        self._step_index = init_step_index(self.timesteps, timestep)

    @property
    def step_index(self):
        return self._step_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Euler Ancestral step with stochastic noise injection."""
        if self.num_inference_steps is None:
            raise RuntimeError(
                "ZERO FALLBACK: set_timesteps() must be called before step()"
            )

        if model_output.shape[1] == sample.shape[1] * 2:
            model_output, _ = model_output.chunk(2, dim=1)

        if self._step_index is None:
            self._init_step_index(timestep)

        assert self._step_index is not None, "step_index must be initialized"
        assert self.sigmas is not None, "sigmas must be initialized via set_timesteps()"
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]

        # Upcast to fp32 for the step arithmetic — VENDOR PARITY (diffusers
        # EulerAncestralDiscreteScheduler.step: "Upcast to avoid precision
        # issues when computing prev_sample"). The fp16 derivative
        # (sample - denoised) / sigma cancels catastrophically at low sigma
        # (error ~ eps_fp16 * |sample| / sigma), which compounds over the
        # many low-t steps of a long schedule (Allegro native 20-step).
        # prev_sample is cast back to the model dtype at the end (vendor).
        _out_dtype = model_output.dtype
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)

        # Convert model output to denoised
        if self.prediction_type == PredictionType.EPSILON:
            denoised = sample - sigma * model_output
        elif self.prediction_type == PredictionType.V_PREDICTION:
            alpha = 1.0 / (sigma ** 2 + 1).sqrt()
            denoised = alpha * sample - sigma * alpha * model_output
        else:
            denoised = model_output

        # Ancestral sampling
        sigma_up = (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2).clamp(min=0) ** 0.5
        sigma_down = (sigma_next ** 2 - sigma_up ** 2).clamp(min=0) ** 0.5

        # Derivative
        d = (sample - denoised) / sigma.clamp(min=1e-8)

        # Euler step
        dt = sigma_down - sigma
        prev_sample = sample + dt * d

        # Add noise (generated at the MODEL dtype, exactly like the vendor
        # randn_tensor(dtype=model_output.dtype) — keeps the noise stream
        # bit-comparable with vendor sampling for a given generator)
        if sigma_next > 0:
            # Diagnostic ancestral-stream pinning (NBX_ANCESTRAL_SEED):
            # draw from a DEDICATED generator seeded like the vendor
            # pipeline's, burning one init-shaped draw first (the vendor
            # generator's first draw is prepare_latents' init). Makes our
            # ancestral sequence bit-equal to the vendor run's, so same-init
            # trajectories are exactly comparable end-to-end. Gated,
            # diagnosis-only.
            import os as _osa
            _aseed = _osa.environ.get("NBX_ANCESTRAL_SEED")
            if _aseed and generator is None:
                if getattr(self, "_diag_gen", None) is None:
                    self._diag_gen = torch.Generator(device=sample.device)
                    self._diag_gen.manual_seed(int(_aseed))
                    torch.randn(sample.shape, generator=self._diag_gen,
                                device=sample.device, dtype=_out_dtype)  # burn init draw
                generator = self._diag_gen
            # Draw on the generator's device then move — diffusers
            # randn_tensor semantics (a torch.Generator is device-bound; on a
            # multi-device plan the sample may live elsewhere).
            _rand_device = generator.device if generator is not None else sample.device
            noise = torch.randn(
                sample.shape,
                generator=generator,
                device=_rand_device,
                dtype=_out_dtype
            ).to(sample.device)
            prev_sample = prev_sample + sigma_up * noise.to(torch.float32)

        prev_sample = prev_sample.to(_out_dtype)
        denoised = denoised.to(_out_dtype)

        assert self._step_index is not None, "step_index must be initialized"
        self._step_index += 1

        if return_dict:
            return {"prev_sample": prev_sample, "denoised": denoised}
        return prev_sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples."""
        assert self.sigmas is not None, "sigmas must be initialized via set_timesteps()"
        sigmas = self.sigmas[:-1][timesteps].to(original_samples.device)
        while len(sigmas.shape) < len(original_samples.shape):
            sigmas = sigmas.unsqueeze(-1)
        return original_samples + sigmas * noise

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Scale model input."""
        if self._step_index is None:
            self._init_step_index(timestep)
        assert self._step_index is not None, "step_index must be initialized"
        assert self.sigmas is not None, "sigmas must be initialized via set_timesteps()"
        sigma = self.sigmas[self._step_index]
        return sample / ((sigma ** 2 + 1) ** 0.5)

    @property
    def init_noise_sigma(self) -> float:
        """Initial noise sigma."""
        if self.sigmas is not None and len(self.sigmas) > 0:
            return self.sigmas[0].item()
        return 1.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EulerAncestralDiscreteScheduler":
        """Create scheduler from NBX config."""
        return cls(config)
