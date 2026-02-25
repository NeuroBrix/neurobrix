# core/module/autoregressive/generator.py
"""
NeuroBrix Autoregressive Generation Module.

Drives token-by-token generation for LLM-based models.
NOT a scheduler - this is a fundamentally different execution paradigm.

IMPORTANT DISTINCTION:
- Scheduler: Drives iterative denoising (20-50 steps, noise -> image)
- Generator: Token-by-token LLM generation (N tokens, prompt -> tokens)

VENDOR-LESS: No model vendor library imports.
DATA-DRIVEN: All parameters from runtime/defaults.json.

Usage Pattern:
    generator = AutoregressiveGenerator.from_config(config)
    generator.set_generation_params(max_tokens=576, ...)

    for step_idx in generator:
        # Execute LM forward pass via GraphExecutor
        hidden = lm_executor.run({"hidden_states": current_input})

        # Get logits via head GraphExecutor
        logits = head_executor.run(hidden)["logits"]

        # Generator samples next token
        next_token, is_done = generator.step(logits, step_idx)

        if is_done:
            break

        # Prepare next input via embed/aligner GraphExecutors
        embed = embed_executor.run({"input_ids": next_token})
        aligned = aligner_executor.run(embed)
        current_input = aligned["output"]

    # Get all generated tokens
    all_tokens = generator.get_generated_tokens()

    # Decode via VQ decoder GraphExecutor
    image = decoder_executor.run({"input_ids": all_tokens})
"""

import torch
from typing import Dict, Any, List, Optional, Tuple, Iterator
from dataclasses import dataclass, field

from .samplers import SamplerBase, CombinedSampler


@dataclass
class GenerationState:
    """
    State container for autoregressive generation.

    Tracks:
    - Generated tokens so far
    - Step index
    - Whether generation is complete
    """
    generated_tokens: List[int] = field(default_factory=list)
    step_idx: int = 0
    is_done: bool = False
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


class AutoregressiveGenerator:
    """
    Generator for autoregressive token generation.

    NOT a scheduler. Drives the generation loop:
    1. Provides iterator for step-by-step generation
    2. Samples tokens from logits using configured sampler
    3. Tracks generated tokens and completion state

    DATA-DRIVEN: All parameters from config, no hardcoding.
    VENDOR-LESS: No janus/transformers imports.
    """

    # Configuration
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    eos_token_id: Optional[int]
    pad_token_id: Optional[int]
    vocab_size: int

    # State
    _state: GenerationState
    _sampler: SamplerBase

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize generator from config.

        Args:
            config: Dict with generation parameters:
                - max_tokens: Maximum tokens to generate
                - temperature: Sampling temperature
                - top_p: Nucleus sampling threshold
                - top_k: Top-K sampling size (0 = disabled)
                - vocab_size: Size of token vocabulary
                - eos_token_id: End of sequence token (optional)
                - pad_token_id: Padding token (optional)
        """
        self.config = config
        self._class_name = config.get("_class_name", "AutoregressiveGenerator")

        # Generation parameters - DATA-DRIVEN from config
        # Direct access for temperature/top_k/top_p — if missing, it's a builder bug (ZERO FALLBACK)
        self.max_tokens = config["max_tokens"]
        self.temperature = config["temperature"]
        self.top_p = config["top_p"]
        self.top_k = config["top_k"]
        self.repetition_penalty = config["repetition_penalty"]
        self.vocab_size = config.get("vocab_size", 16384)
        self.eos_token_id = config.get("eos_token_id")
        self.pad_token_id = config.get("pad_token_id")

        # Prompt token IDs for repetition penalty context.
        # HuggingFace penalizes tokens from BOTH prompt and generated text.
        # Without prompt context, repetition penalty only sees generated tokens,
        # causing degraded output quality with sampling (temp > 0).
        self._prompt_ids: List[int] = []

        # Initialize state
        self._state = GenerationState()

        # Initialize sampler based on config
        self._sampler = self._create_sampler()

    def _create_sampler(self) -> SamplerBase:
        """
        Create token sampler based on config.

        DATA-DRIVEN sampler selection:
        - top_p < 1.0 -> TopPSampler (nucleus)
        - temperature != 1.0 -> TemperatureSampler
        - Otherwise -> CombinedSampler (handles all cases)
        """
        sampler_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_tokens_to_keep": 1,
            "repetition_penalty": self.repetition_penalty,
        }

        # Use CombinedSampler for flexibility
        return CombinedSampler(sampler_config)

    def set_generation_params(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Override generation parameters.

        Allows runtime override of config values.
        """
        needs_resample = False
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature
            needs_resample = True
        if top_p is not None:
            self.top_p = top_p
            needs_resample = True
        if top_k is not None:
            self.top_k = top_k
            needs_resample = True
        if repetition_penalty is not None:
            self.repetition_penalty = repetition_penalty
            needs_resample = True
        if needs_resample:
            self._sampler = self._create_sampler()
        if device is not None:
            self._state.device = device

    def set_prompt_ids(self, prompt_ids: List[int]) -> None:
        """Set prompt token IDs for repetition penalty context."""
        self._prompt_ids = prompt_ids

    def reset(self) -> None:
        """Reset generator state for new generation."""
        self._state = GenerationState(device=self._state.device)

    def __iter__(self) -> Iterator[int]:
        """
        Iterator for generation loop.

        Yields step indices from 0 to max_tokens-1.
        Use is_done property or return value from step() to check completion.
        """
        self.reset()
        for step_idx in range(self.max_tokens):
            if self._state.is_done:
                break
            self._state.step_idx = step_idx
            yield step_idx

    def step(
        self,
        logits: torch.Tensor,
        step_idx: int
    ) -> Tuple[torch.Tensor, bool]:
        """
        Sample next token from logits.

        Args:
            logits: Token logits from gen_head [batch, vocab_size]
            step_idx: Current step index

        Returns:
            (next_token, is_done): Token ID tensor and completion flag

        The sampler handles temperature, top-k, top-p filtering.
        """
        # Validate logits shape
        if logits.dim() == 3:
            # [batch, seq, vocab] -> take last position
            logits = logits[:, -1, :]
        elif logits.dim() != 2:
            raise ValueError(
                f"Expected logits shape [batch, vocab] or [batch, seq, vocab], "
                f"got {list(logits.shape)}"
            )

        # Build full token context for repetition penalty (prompt + generated).
        # HuggingFace penalizes tokens from the ENTIRE sequence, not just generated.
        all_context = self._prompt_ids + self._state.generated_tokens
        context_ids = (
            torch.tensor([all_context], device=logits.device) if all_context else None
        )

        # Sample using configured sampler
        # Sampler handles temperature, top-k, top-p, repetition penalty
        next_token = self._sampler(logits, input_ids=context_ids)

        # Track generated token
        token_id = int(next_token.squeeze().item())
        self._state.generated_tokens.append(token_id)
        self._state.step_idx = step_idx

        # Check for EOS
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            self._state.is_done = True

        # Check for max tokens
        if len(self._state.generated_tokens) >= self.max_tokens:
            self._state.is_done = True

        return next_token, self._state.is_done

    def get_generated_tokens(self) -> torch.Tensor:
        """
        Get all generated tokens as tensor.

        Returns:
            Tensor [num_tokens] with generated token IDs
        """
        return torch.tensor(
            self._state.generated_tokens,
            dtype=torch.long,
            device=self._state.device
        )

    @property
    def is_done(self) -> bool:
        """Check if generation is complete."""
        return self._state.is_done

    @property
    def num_generated(self) -> int:
        """Number of tokens generated so far."""
        return len(self._state.generated_tokens)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AutoregressiveGenerator":
        """Create generator from NBX config."""
        return cls(config)


class VQImageGenerator(AutoregressiveGenerator):
    """
    Specialized generator for VQ-based image generation.

    Extends AutoregressiveGenerator with:
    - Image size -> token count calculation
    - Codebook configuration
    - Image token grid management

    Used by models like Janus, LlamaGen, etc.
    """

    # VQ-specific configuration
    image_size: int
    patch_size: int
    codebook_size: int
    codebook_dim: int

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VQ image generator.

        Args:
            config: Dict with:
                - image_size: Target image size in pixels
                - patch_size: VQ patch size (typically 16)
                - codebook_size: Number of VQ codes (vocab_size)
                - codebook_dim: Dimension of VQ embeddings
                - Plus all AutoregressiveGenerator params
        """
        # VQ-specific params - DATA-DRIVEN from config
        self.image_size = config.get("image_size", 384)
        self.patch_size = config.get("patch_size", 16)
        self.codebook_size = config.get("codebook_size", 16384)
        self.codebook_dim = config.get("codebook_dim", 8)

        # Calculate token count from image size
        num_patches = self.image_size // self.patch_size
        max_tokens = num_patches * num_patches
        config["max_tokens"] = max_tokens
        config["vocab_size"] = self.codebook_size

        # Initialize base generator
        super().__init__(config)

    def get_image_grid_shape(self) -> Tuple[int, int, int, int]:
        """
        Get shape for VQ decoder.

        Returns:
            (batch, codebook_dim, height_patches, width_patches)
        """
        num_patches = self.image_size // self.patch_size
        return (1, self.codebook_dim, num_patches, num_patches)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VQImageGenerator":
        """Create generator from NBX config."""
        return cls(config)
