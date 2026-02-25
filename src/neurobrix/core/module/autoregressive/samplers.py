# core/module/autoregressive/samplers.py
"""
LLM Sampling Strategies.

Implements various token sampling methods for autoregressive language models:
- Greedy: Always pick highest probability token
- Top-K: Sample from top k tokens
- Top-P (Nucleus): Sample from smallest set with cumulative prob >= p
- Temperature: Scale logits before sampling

ZERO FALLBACK: Uses safe defaults for LLM sampling since they're user preferences.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Set


# =============================================================================
# Base Classes (moved from scheduler for proper separation)
# =============================================================================

class SamplerBase(ABC):
    """
    Base class for LLM sampling strategies.

    Used by: All autoregressive language models
    """

    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Sample next token(s) from logits."""
        pass

    @abstractmethod
    def process_logits(
        self,
        logits: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Process/filter logits before sampling."""
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "SamplerBase":
        """Create sampler from config dict."""
        pass


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


# =============================================================================
# Sampler Implementations
# =============================================================================

class GreedySampler(SamplerBase):
    """
    Greedy decoding - always select highest probability token.

    Deterministic and fast. Good for factual/deterministic tasks.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize (no config needed for greedy)."""
        self.config = config or {}

    def process_logits(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Greedy doesn't modify logits."""
        return logits

    def __call__(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Select highest probability token."""
        return torch.argmax(logits, dim=-1, keepdim=True)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GreedySampler":
        """Create sampler from config."""
        return cls(config)


class TemperatureSampler(SamplerBase):
    """
    Temperature-scaled sampling.

    Higher temperature = more random, lower = more deterministic.
    Temperature of 1.0 = no scaling.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize with temperature from config."""
        validated = SamplerConfig.validate(config)
        self.temperature = validated["temperature"]

    def process_logits(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Scale logits by temperature."""
        if self.temperature != 1.0 and self.temperature > 0:
            logits = logits / self.temperature
        return logits

    def __call__(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample from temperature-scaled distribution."""
        processed = self.process_logits(logits, **kwargs)
        probs = torch.softmax(processed, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TemperatureSampler":
        """Create sampler from config."""
        return cls(config)


class TopKSampler(SamplerBase):
    """
    Top-K sampling - sample from top k tokens.

    Limits vocabulary to top k most likely tokens.
    Good balance of quality and diversity.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize with top_k and temperature from config."""
        validated = SamplerConfig.validate(config)
        self.top_k = validated["top_k"]
        self.temperature = validated["temperature"]
        self.min_tokens_to_keep = validated["min_tokens_to_keep"]

    def process_logits(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Filter to top-k tokens."""
        # Temperature scaling
        if self.temperature != 1.0 and self.temperature > 0:
            logits = logits / self.temperature

        # Top-k filtering
        if self.top_k > 0:
            top_k = min(max(self.top_k, self.min_tokens_to_keep), logits.size(-1))
            # Get the kth largest value
            top_k_values = torch.topk(logits, top_k, dim=-1)
            kth_value = top_k_values.values[..., -1:]
            # Mask out tokens below kth value
            indices_to_remove = logits < kth_value
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits

    def __call__(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample from top-k filtered distribution."""
        processed = self.process_logits(logits, **kwargs)
        probs = torch.softmax(processed, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TopKSampler":
        """Create sampler from config."""
        return cls(config)


class TopPSampler(SamplerBase):
    """
    Top-P (Nucleus) Sampling.

    Samples from smallest set of tokens whose cumulative probability >= p.
    Dynamically adjusts vocabulary size based on distribution.
    Most commonly used LLM sampling strategy.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize with top_p, top_k, and temperature from config."""
        validated = SamplerConfig.validate(config)
        self.top_p = validated["top_p"]
        self.top_k = validated["top_k"]
        self.temperature = validated["temperature"]
        self.min_tokens_to_keep = validated["min_tokens_to_keep"]

    def process_logits(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Filter using top-p (nucleus) sampling."""
        # Temperature scaling first
        if self.temperature != 1.0 and self.temperature > 0:
            logits = logits / self.temperature

        # Top-k filter (applied before top-p if specified)
        if self.top_k > 0:
            top_k = min(max(self.top_k, self.min_tokens_to_keep), logits.size(-1))
            kth_value = torch.topk(logits, top_k, dim=-1).values[..., -1:]
            indices_to_remove = logits < kth_value
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Top-p filter
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p

            # Keep at least min_tokens_to_keep tokens
            sorted_indices_to_remove[..., :self.min_tokens_to_keep] = False

            # Shift to include the first token that exceeded the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Scatter back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1,
                index=sorted_indices,
                src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits

    def __call__(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample from nucleus-filtered distribution."""
        processed = self.process_logits(logits, **kwargs)
        probs = torch.softmax(processed, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TopPSampler":
        """Create sampler from config."""
        return cls(config)


class RepetitionPenaltySampler(SamplerBase):
    """
    Repetition Penalty Sampler.

    Penalizes tokens that have appeared in the context.
    Reduces repetition in generated text.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize with repetition_penalty from config."""
        validated = SamplerConfig.validate(config)
        self.repetition_penalty = validated["repetition_penalty"]
        self.temperature = validated["temperature"]

    def process_logits(
        self,
        logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        if input_ids is not None and self.repetition_penalty != 1.0:
            # Get unique tokens from input
            for batch_idx in range(logits.size(0)):
                for token_id in set(input_ids[batch_idx].tolist()):
                    if token_id < logits.size(-1):
                        if logits[batch_idx, token_id] > 0:
                            logits[batch_idx, token_id] /= self.repetition_penalty
                        else:
                            logits[batch_idx, token_id] *= self.repetition_penalty

        # Temperature scaling
        if self.temperature != 1.0 and self.temperature > 0:
            logits = logits / self.temperature

        return logits

    def __call__(
        self,
        logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Sample with repetition penalty applied."""
        processed = self.process_logits(logits, input_ids=input_ids, **kwargs)
        probs = torch.softmax(processed, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RepetitionPenaltySampler":
        """Create sampler from config."""
        return cls(config)


class CombinedSampler(SamplerBase):
    """
    Combined Sampler with all filtering strategies.

    Applies temperature, top-k, top-p, and repetition penalty.
    Most flexible sampler for production use.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize with all sampling parameters."""
        validated = SamplerConfig.validate(config)
        self.temperature = validated["temperature"]
        self.top_k = validated["top_k"]
        self.top_p = validated["top_p"]
        self.repetition_penalty = validated["repetition_penalty"]
        self.min_tokens_to_keep = validated["min_tokens_to_keep"]

    def process_logits(
        self,
        logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Apply all filtering strategies."""
        # 1. Repetition penalty
        if input_ids is not None and self.repetition_penalty != 1.0:
            for batch_idx in range(logits.size(0)):
                for token_id in set(input_ids[batch_idx].tolist()):
                    if token_id < logits.size(-1):
                        if logits[batch_idx, token_id] > 0:
                            logits[batch_idx, token_id] /= self.repetition_penalty
                        else:
                            logits[batch_idx, token_id] *= self.repetition_penalty

        # 2. Temperature scaling
        if self.temperature != 1.0 and self.temperature > 0:
            logits = logits / self.temperature

        # 3. Top-k filtering
        if self.top_k > 0:
            top_k = min(max(self.top_k, self.min_tokens_to_keep), logits.size(-1))
            kth_value = torch.topk(logits, top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth_value, float('-inf'))

        # 4. Top-p filtering
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., :self.min_tokens_to_keep] = False
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1,
                index=sorted_indices,
                src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits

    def __call__(
        self,
        logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Sample with all filters applied."""
        processed = self.process_logits(logits, input_ids=input_ids, **kwargs)
        probs = torch.softmax(processed, dim=-1)

        # Greedy decoding when temperature <= 0
        if self.temperature <= 0:
            result = probs.argmax(dim=-1, keepdim=True)
        else:
            result = torch.multinomial(probs, num_samples=1)
        return result

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CombinedSampler":
        """Create sampler from config."""
        return cls(config)


# =============================================================================
# Convenience functions
# =============================================================================

def sample_top_k(logits: torch.Tensor, k: int, temperature: float = 1.0) -> torch.Tensor:
    """Quick top-k sampling function."""
    sampler = TopKSampler({"top_k": k, "temperature": temperature})
    return sampler(logits)


def sample_top_p(logits: torch.Tensor, p: float, temperature: float = 1.0) -> torch.Tensor:
    """Quick top-p (nucleus) sampling function."""
    sampler = TopPSampler({"top_p": p, "temperature": temperature})
    return sampler(logits)
