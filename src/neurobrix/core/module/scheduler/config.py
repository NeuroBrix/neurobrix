# core/scheduler/config.py
"""
Scheduler Configuration Validation.

ZERO FALLBACK: All required values MUST come from NBX container.
Missing required key = RuntimeError (not silent default).
"""

from typing import Dict, Any, Set


class SchedulerConfigError(RuntimeError):
    """Raised when scheduler config violates ZERO FALLBACK principle."""
    pass


class SchedulerConfig:
    """
    Validates and processes scheduler configuration from NBX container.

    ZERO FALLBACK PRINCIPLE:
    - Required keys MUST be present in config
    - Missing required key = explicit crash with helpful message
    - Only truly optional features can have defaults
    """

    # Keys that MUST be in NBX config - no defaults allowed
    REQUIRED_KEYS: Set[str] = {
        "num_train_timesteps",
        "beta_start",
        "beta_end",
        "beta_schedule",
        "prediction_type",
        "algorithm_type",
        "solver_type",
        "solver_order",
        "timestep_spacing",
        "lower_order_final",
    }

    # Keys that can have SAFE defaults (only features that can be disabled)
    # These defaults DISABLE functionality, they don't change core behavior
    OPTIONAL_SAFE_DEFAULTS: Dict[str, Any] = {
        # Feature toggles (False = disabled)
        "thresholding": False,
        "euler_at_final": False,
        "use_karras_sigmas": False,
        "use_lu_lambdas": False,
        "rescale_betas_zero_snr": False,

        # Derived values (only used when features enabled)
        "sample_max_value": 1.0,
        "final_sigmas_type": "sigma_min",
        "dynamic_thresholding_ratio": 0.995,
        "lambda_min_clipped": float("-inf"),
        "variance_type": None,
        "trained_betas": None,
        "steps_offset": 0,
    }

    @classmethod
    def validate(cls, config: Dict[str, Any], scheduler_type: str = "unknown") -> Dict[str, Any]:
        """
        Validate scheduler config and return cleaned version.

        Args:
            config: Raw config from NBX container
            scheduler_type: Scheduler type for error messages

        Returns:
            Validated and cleaned config dict

        Raises:
            SchedulerConfigError: If required keys are missing
        """
        # 1. Clean HuggingFace-specific keys
        clean_config = {
            k: v for k, v in config.items()
            if not k.startswith("_")  # Remove _class_name, _diffusers_version, etc.
        }

        # 2. Check required keys
        missing = cls.REQUIRED_KEYS - set(clean_config.keys())
        if missing:
            available = sorted(clean_config.keys())
            raise SchedulerConfigError(
                f"\n{'='*70}\n"
                f"ZERO FALLBACK VIOLATION: Scheduler config incomplete\n"
                f"{'='*70}\n"
                f"Scheduler type: {scheduler_type}\n"
                f"Missing REQUIRED keys: {sorted(missing)}\n"
                f"Available keys: {available}\n"
                f"\n"
                f"These values MUST be embedded in the NBX container.\n"
                f"Check modules/scheduler/scheduler_config.json in the .nbx file.\n"
                f"{'='*70}"
            )

        # 3. Apply safe defaults for optional features
        for key, default in cls.OPTIONAL_SAFE_DEFAULTS.items():
            if key not in clean_config:
                clean_config[key] = default

        return clean_config

    @classmethod
    def require(cls, config: Dict[str, Any], key: str) -> Any:
        """
        Get required key from config or raise explicit error.

        Use this for keys that might be dynamically required.
        """
        if key not in config:
            raise SchedulerConfigError(
                f"ZERO FALLBACK: Config missing required key '{key}'.\n"
                f"This value must be in NBX container."
            )
        return config[key]


class FlowSchedulerConfig:
    """Config validation for flow matching schedulers (Flux, SD3)."""

    REQUIRED_KEYS: Set[str] = {
        "num_train_timesteps",
        "shift",  # Critical for Flux (3.0) vs SD3 (1.0)
    }

    OPTIONAL_SAFE_DEFAULTS: Dict[str, Any] = {
        "base_image_seq_len": 256,
        "max_image_seq_len": 4096,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "invert_sigmas": False,
    }

    @classmethod
    def validate(cls, config: Dict[str, Any], scheduler_type: str = "unknown") -> Dict[str, Any]:
        """Validate flow scheduler config."""
        clean_config = {k: v for k, v in config.items() if not k.startswith("_")}

        missing = cls.REQUIRED_KEYS - set(clean_config.keys())
        if missing:
            available = sorted(clean_config.keys())
            raise SchedulerConfigError(
                f"\n{'='*70}\n"
                f"ZERO FALLBACK VIOLATION: Flow scheduler config incomplete\n"
                f"{'='*70}\n"
                f"Scheduler type: {scheduler_type}\n"
                f"Missing REQUIRED keys: {sorted(missing)}\n"
                f"Available keys: {available}\n"
                f"\n"
                f"CRITICAL: 'shift' parameter differs between models:\n"
                f"  - Flux: shift=3.0\n"
                f"  - SD3: shift=1.0\n"
                f"This MUST come from NBX container.\n"
                f"{'='*70}"
            )

        for key, default in cls.OPTIONAL_SAFE_DEFAULTS.items():
            if key not in clean_config:
                clean_config[key] = default

        return clean_config


class SamplerConfig:
    """Config validation for LLM samplers."""

    # Samplers can have more flexible defaults since they're user preferences
    REQUIRED_KEYS: Set[str] = set()  # All optional for samplers

    DEFAULTS: Dict[str, Any] = {
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "min_tokens_to_keep": 1,
    }

    @classmethod
    def validate(cls, config: Dict[str, Any], scheduler_type: str = "unknown") -> Dict[str, Any]:
        """Validate sampler config with flexible defaults."""
        clean_config = {k: v for k, v in config.items() if not k.startswith("_")}

        for key, default in cls.DEFAULTS.items():
            if key not in clean_config:
                clean_config[key] = default

        return clean_config
