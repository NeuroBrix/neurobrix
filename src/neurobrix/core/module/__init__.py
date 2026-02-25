# core/module/__init__.py
"""
NeuroBrix Module Package.

Provides execution strategy modules for different generation paradigms:
- scheduler: Diffusion denoising (iterative_process flow)
- autoregressive: Token-by-token generation (autoregressive flow)
- tokenizer: Text tokenization utilities

CRITICAL: Autoregressive is NOT a scheduler. They are fundamentally different:
- Scheduler: Drives iterative denoising (20-50 steps, noise -> image)
- Autoregressive: Token-by-token LLM generation (N tokens, prompt -> tokens)
"""

from neurobrix.core.module.scheduler.factory import SchedulerFactory
from neurobrix.core.module.autoregressive.factory import AutoregressiveFactory

__all__ = [
    "SchedulerFactory",
    "AutoregressiveFactory",
]
