# core/module/autoregressive/__init__.py
"""
Autoregressive Generation Module.

Token-by-token generation for LLMs and VQ image generation.
NOT a scheduler - fundamentally different execution paradigm.

Classes:
- AutoregressiveGenerator: Generic autoregressive token generation
- VQImageGenerator: VQ codebook token generation for images
- GenerationState: State container for autoregressive generation

Samplers:
- CombinedSampler: Full-featured sampler with all strategies
- TopPSampler: Nucleus sampling
- TopKSampler: Top-K sampling
- TemperatureSampler: Temperature-scaled sampling
- GreedySampler: Deterministic greedy decoding
"""

from neurobrix.core.module.autoregressive.generator import (
    AutoregressiveGenerator,
    VQImageGenerator,
    GenerationState,
)
from neurobrix.core.module.autoregressive.factory import AutoregressiveFactory
from neurobrix.core.module.autoregressive.samplers import (
    SamplerBase,
    SamplerConfig,
    CombinedSampler,
    TopPSampler,
    TopKSampler,
    TemperatureSampler,
    GreedySampler,
    RepetitionPenaltySampler,
    sample_top_k,
    sample_top_p,
)

__all__ = [
    # Generators
    "AutoregressiveGenerator",
    "VQImageGenerator",
    "GenerationState",
    # Factory
    "AutoregressiveFactory",
    # Samplers
    "SamplerBase",
    "SamplerConfig",
    "CombinedSampler",
    "TopPSampler",
    "TopKSampler",
    "TemperatureSampler",
    "GreedySampler",
    "RepetitionPenaltySampler",
    "sample_top_k",
    "sample_top_p",
]
