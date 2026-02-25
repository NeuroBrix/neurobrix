# core/scheduler/consistency/__init__.py
"""
Consistency Model Schedulers.

For: LCM (Latent Consistency Model), TCD, etc.
Enables few-step (1-8) generation.
"""

from .lcm import LCMScheduler

__all__ = ["LCMScheduler"]
