# core/module/scheduler/__init__.py
"""
Diffusion Scheduler Module.

Iterative denoising schedulers for diffusion models ONLY.
NOT for autoregressive generation.

Scheduler Types:
- diffusion/: DDPM, DDIM, DPM-Solver, Euler, etc.
- consistency/: Consistency models
- flow/: Flow matching schedulers
"""

from neurobrix.core.module.scheduler.factory import SchedulerFactory
from neurobrix.core.module.scheduler.base import Scheduler, DiffusionSchedulerBase, FlowSchedulerBase
from neurobrix.core.module.scheduler.config import SchedulerConfig

# Alias for backward compatibility
BaseScheduler = Scheduler

__all__ = [
    "SchedulerFactory",
    "Scheduler",
    "DiffusionSchedulerBase",
    "FlowSchedulerBase",
    "SchedulerConfig",
    # Backward compatibility
    "BaseScheduler",
]
