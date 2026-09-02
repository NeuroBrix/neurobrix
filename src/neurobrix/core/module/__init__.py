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

# PEP 562 lazy exports (D-CORE-MODULE-INIT-TORCH, 2026-09-02): the two
# factories import torch (scheduler/base.py), and every
# `from neurobrix.core.module.audio.<x> import …` executed by a TRITON flow
# runs this init first — the shared numpy DSP (`mel_dsp`, `stt_longform`)
# is torch-free but the package it lives in was not, and the static R33
# grep cannot see a package init. The factories resolve on first attribute
# access; a plain submodule import no longer touches torch.
__all__ = [
    "SchedulerFactory",
    "AutoregressiveFactory",
]

_LAZY = {
    "SchedulerFactory": "neurobrix.core.module.scheduler.factory",
    "AutoregressiveFactory": "neurobrix.core.module.autoregressive.factory",
}


def __getattr__(name):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
