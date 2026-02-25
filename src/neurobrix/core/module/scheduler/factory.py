# core/module/scheduler/factory.py
"""
NeuroBrix Scheduler Factory - ZERO FALLBACK.

Complete registry of DIFFUSION scheduler implementations ONLY.
Unknown scheduler type = crash, not silent fallback.

IMPORTANT: This factory is for DIFFUSION schedulers only.
Autoregressive generation uses core.module.autoregressive.factory.
"""

from typing import Dict, Any, Type, Union, Optional
from .base import Scheduler, DiffusionSchedulerBase, FlowSchedulerBase
from .config import SchedulerConfigError


class SchedulerFactory:
    """
    Factory to instantiate DIFFUSION schedulers from NBX config.

    ZERO FALLBACK: Unknown scheduler type = crash, not silent fallback.

    Supports:
    - Diffusion schedulers (DPM++, DDIM, Euler, etc.)
    - Flow schedulers (FlowEuler for Flux/SD3)
    - Consistency schedulers (LCM)

    NOT for autoregressive generation - use AutoregressiveFactory instead.
    """

    # Registry of supported schedulers
    # Maps config class name / alias to (module_path, class_name)
    _REGISTRY: Dict[str, tuple] = {
        # =========================================================================
        # DPM-Solver family
        # =========================================================================
        "DPMSolverMultistepScheduler": ("diffusion.dpm_solver_pp", "DPMSolverPPScheduler"),
        "DPMSolverSinglestepScheduler": ("diffusion.dpm_solver_pp", "DPMSolverPPScheduler"),
        "dpmsolver++": ("diffusion.dpm_solver_pp", "DPMSolverPPScheduler"),
        "dpm++": ("diffusion.dpm_solver_pp", "DPMSolverPPScheduler"),
        "dpm++_2m": ("diffusion.dpm_solver_pp", "DPMSolverPPScheduler"),
        "dpm++_2m_karras": ("diffusion.dpm_solver_pp", "DPMSolverPPScheduler"),

        # =========================================================================
        # DDIM family
        # =========================================================================
        "DDIMScheduler": ("diffusion.ddim", "DDIMScheduler"),
        "ddim": ("diffusion.ddim", "DDIMScheduler"),

        # =========================================================================
        # Euler family
        # =========================================================================
        "EulerDiscreteScheduler": ("diffusion.euler", "EulerDiscreteScheduler"),
        "euler": ("diffusion.euler", "EulerDiscreteScheduler"),
        "euler_discrete": ("diffusion.euler", "EulerDiscreteScheduler"),

        "EulerAncestralDiscreteScheduler": ("diffusion.euler", "EulerAncestralDiscreteScheduler"),
        "euler_a": ("diffusion.euler", "EulerAncestralDiscreteScheduler"),
        "euler_ancestral": ("diffusion.euler", "EulerAncestralDiscreteScheduler"),

        # =========================================================================
        # Flow family (Flux, SD3)
        # =========================================================================
        "FlowMatchEulerDiscreteScheduler": ("flow.flow_euler", "FlowEulerScheduler"),
        "FlowEulerScheduler": ("flow.flow_euler", "FlowEulerScheduler"),
        "flow_euler": ("flow.flow_euler", "FlowEulerScheduler"),
        "flux": ("flow.flow_euler", "FlowEulerScheduler"),

        "RectifiedFlowScheduler": ("flow.flow_euler", "RectifiedFlowScheduler"),
        "rectified_flow": ("flow.flow_euler", "RectifiedFlowScheduler"),

        # =========================================================================
        # Consistency family (LCM, TCD)
        # =========================================================================
        "LCMScheduler": ("consistency.lcm", "LCMScheduler"),
        "lcm": ("consistency.lcm", "LCMScheduler"),
        "latent_consistency": ("consistency.lcm", "LCMScheduler"),

        # =========================================================================
        # NOTE: LLM samplers and autoregressive generators have been moved to
        # core.module.autoregressive - they are NOT schedulers!
        # Use AutoregressiveFactory for token-by-token generation.
        # =========================================================================
    }

    # Cache for imported classes
    _CLASS_CACHE: Dict[str, Type] = {}

    @classmethod
    def _get_class(cls, module_path: str, class_name: str) -> Type:
        """Lazily import and return scheduler class."""
        cache_key = f"{module_path}.{class_name}"

        if cache_key in cls._CLASS_CACHE:
            return cls._CLASS_CACHE[cache_key]

        # Import the module
        import importlib
        full_module = f"neurobrix.core.module.scheduler.{module_path}"

        try:
            module = importlib.import_module(full_module)
            scheduler_class = getattr(module, class_name)
            cls._CLASS_CACHE[cache_key] = scheduler_class
            return scheduler_class
        except ImportError as e:
            raise SchedulerConfigError(
                f"ZERO FALLBACK: Failed to import scheduler module.\n"
                f"Module: {full_module}\n"
                f"Class: {class_name}\n"
                f"Error: {e}"
            )
        except AttributeError:
            raise SchedulerConfigError(
                f"ZERO FALLBACK: Scheduler class not found in module.\n"
                f"Module: {full_module}\n"
                f"Class: {class_name}"
            )

    @classmethod
    def create(
        cls,
        config: Dict[str, Any]
    ) -> Union[Scheduler, DiffusionSchedulerBase, FlowSchedulerBase]:
        """
        Create a scheduler instance from NBX config.

        ZERO FALLBACK: Crashes if scheduler type unknown or missing.

        Args:
            config: Scheduler configuration from NBX container

        Returns:
            Scheduler instance

        Raises:
            SchedulerConfigError: If scheduler type missing or unknown
        """
        # 1. Get scheduler type - REQUIRED
        scheduler_type = config.get("_class_name")

        if not scheduler_type:
            raise SchedulerConfigError(
                f"\n{'='*70}\n"
                f"ZERO FALLBACK VIOLATION: Scheduler config missing '_class_name'\n"
                f"{'='*70}\n"
                f"Config keys: {sorted(config.keys())}\n"
                f"\n"
                f"The NBX container MUST specify which scheduler to use.\n"
                f"Check modules/scheduler/scheduler_config.json in the .nbx file.\n"
                f"{'='*70}"
            )

        # 2. Look up implementation - REQUIRED
        if scheduler_type not in cls._REGISTRY:
            available = sorted(set(cls._REGISTRY.keys()))
            raise SchedulerConfigError(
                f"\n{'='*70}\n"
                f"ZERO FALLBACK VIOLATION: Unknown scheduler type\n"
                f"{'='*70}\n"
                f"Requested: '{scheduler_type}'\n"
                f"Available: {available}\n"
                f"\n"
                f"To add support for this scheduler:\n"
                f"1. Implement the scheduler in core/module/scheduler/\n"
                f"2. Register it in SchedulerFactory._REGISTRY\n"
                f"{'='*70}"
            )

        # 3. Get class and create instance
        module_path, class_name = cls._REGISTRY[scheduler_type]
        scheduler_class = cls._get_class(module_path, class_name)

        return scheduler_class.from_config(config)

    @classmethod
    def create_by_name(
        cls,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Union[Scheduler, DiffusionSchedulerBase, FlowSchedulerBase]:
        """
        Create scheduler by name (convenience method).

        Args:
            name: Scheduler name/alias (e.g., "ddim", "euler", "top_p")
            config: Optional config dict

        Returns:
            Scheduler instance
        """
        if config is None:
            config = {}
        config["_class_name"] = name
        return cls.create(config)

    @classmethod
    def list_available(cls) -> list:
        """List all available scheduler types (unique class names only)."""
        # Return unique scheduler types (not aliases)
        unique_types = set()
        for key, (module, classname) in cls._REGISTRY.items():
            unique_types.add(classname)
        return sorted(unique_types)

    @classmethod
    def list_aliases(cls) -> Dict[str, str]:
        """List all aliases and their target schedulers."""
        return {
            key: f"{module}.{classname}"
            for key, (module, classname) in cls._REGISTRY.items()
        }

    @classmethod
    def get_scheduler_info(cls, scheduler_type: str) -> Dict[str, Any]:
        """Get information about a scheduler type."""
        if scheduler_type not in cls._REGISTRY:
            return {"error": f"Unknown scheduler: {scheduler_type}"}

        module_path, class_name = cls._REGISTRY[scheduler_type]
        return {
            "name": scheduler_type,
            "class": class_name,
            "module": f"neurobrix.core.module.scheduler.{module_path}",
            "aliases": [
                k for k, v in cls._REGISTRY.items()
                if v == (module_path, class_name)
            ],
        }
