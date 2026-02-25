"""
NeuroBrix Universal Scheduler System.

Provides unified interfaces for all scheduler types:
- DiffusionSchedulerBase: Image/Video/Audio diffusion models
- FlowSchedulerBase: Flow matching models (Flux, SD3)
- SamplerBase: LLM token sampling

Enterprise Grade: ZERO DIFFUSERS DEPENDENCY at runtime.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
from enum import Enum
import torch


class PredictionType(Enum):
    """Model output prediction type."""
    EPSILON = "epsilon"           # Predicts noise
    V_PREDICTION = "v_prediction" # Predicts velocity
    SAMPLE = "sample"             # Predicts clean sample (x0)
    FLOW = "flow"                 # Predicts flow vector (for flow matching)
    FLOW_PREDICTION = "flow_prediction"  # Flow prediction (Sana, newer flow models)


class TimestepSpacing(Enum):
    """Timestep spacing strategy."""
    LINSPACE = "linspace"         # Linear spacing
    LEADING = "leading"           # Leading spacing
    TRAILING = "trailing"         # Trailing spacing (diffusers default)
    KARRAS = "karras"             # Karras sigmas


class BetaSchedule(Enum):
    """Beta schedule type."""
    LINEAR = "linear"
    SCALED_LINEAR = "scaled_linear"
    SQUAREDCOS_CAP_V2 = "squaredcos_cap_v2"  # Cosine schedule
    SIGMOID = "sigmoid"


# =============================================================================
# LEGACY INTERFACE (for backward compatibility)
# =============================================================================

class Scheduler(ABC):
    """
    Abstract base class for all schedulers (legacy interface).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scheduler with config.

        Args:
            config: Configuration dictionary (from scheduler_config.json)
        """
        self.config = config
        self.timesteps = None
        self.sigmas = None
        self.num_inference_steps = None

    @abstractmethod
    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Sets the discrete timesteps used for the diffusion chain.

        Args:
            num_inference_steps: Number of diffusion steps used to generate samples.
            device: Device to place timesteps on.
        """
        pass

    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict the sample at the previous timestep by reversing the SDE.

        Args:
            model_output: Direct output from learned diffusion model.
            timestep: Current discrete timestep in the diffusion chain.
            sample: Current instance of sample being created by diffusion process.
            return_dict: Whether to return a dict or tuple.

        Returns:
            The predicted sample at the previous timestep (x_t-1).
        """
        pass

    @abstractmethod
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to the original samples.
        """
        pass

    @abstractmethod
    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input.
        """
        pass


# =============================================================================
# DIFFUSION SCHEDULER BASE
# =============================================================================

class DiffusionSchedulerBase(ABC):
    """
    Base class for diffusion schedulers.

    Used by: Image diffusion, Video diffusion, Audio diffusion
    Models: SD, SDXL, PixArt, Kandinsky, SVD, AnimateDiff, AudioLDM, etc.
    """

    # Configuration
    num_train_timesteps: int
    prediction_type: PredictionType

    # State
    timesteps: Optional[torch.Tensor]
    num_inference_steps: Optional[int]

    @abstractmethod
    def set_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device,
        **kwargs
    ) -> None:
        """Configure the timestep schedule."""
        pass

    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Compute one denoising step."""
        pass

    @abstractmethod
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples (forward diffusion process)."""
        pass

    @abstractmethod
    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Scale the model input if required."""
        pass

    @property
    @abstractmethod
    def init_noise_sigma(self) -> float:
        """Initial noise sigma for sampling."""
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "DiffusionSchedulerBase":
        """Create scheduler from NBX config."""
        pass


# =============================================================================
# FLOW SCHEDULER BASE
# =============================================================================

class FlowSchedulerBase(ABC):
    """
    Base class for flow matching schedulers.

    Used by: Flux, Stable Diffusion 3, Rectified Flow models
    """

    num_train_timesteps: int
    timesteps: torch.Tensor

    @abstractmethod
    def set_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device,
        **kwargs
    ) -> None:
        """Configure timestep schedule (typically 0->1 for flow)."""
        pass

    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Compute one flow step."""
        pass

    @abstractmethod
    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Scale input for flow models."""
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "FlowSchedulerBase":
        """Create scheduler from NBX config."""
        pass


# =============================================================================
# LLM SAMPLER BASE (imported from autoregressive module)
# =============================================================================
# SamplerBase is semantically for LLM token sampling, not diffusion scheduling.
# The canonical definition lives in core.module.autoregressive.samplers.
# Re-exported here for backward compatibility only.

from neurobrix.core.module.autoregressive.samplers import SamplerBase, SamplerConfig
