"""Triton Autoregressive Generator — zero torch dependency.

Ported from core/module/autoregressive/generator.py.
Drives token-by-token generation using NBXTensor + Triton samplers.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator

from .samplers import CombinedSampler, create_sampler


@dataclass
class GenerationState:
    """State container for autoregressive generation."""
    generated_tokens: List[int] = field(default_factory=list)
    step_idx: int = 0
    is_done: bool = False
    device_idx: int = 0


class TritonGenerator:
    """Token-by-token generation loop with Triton samplers.

    Ported from AutoregressiveGenerator. Same interface, NBXTensor throughout.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_tokens: int = config["max_tokens"]
        self.temperature: float = config["temperature"]
        self.top_p: float = config["top_p"]
        self.top_k: int = config["top_k"]
        self.repetition_penalty: float = config["repetition_penalty"]
        self.vocab_size: int = config.get("vocab_size", 16384)
        self.eos_token_id: Optional[int] = config.get("eos_token_id")
        self.pad_token_id: Optional[int] = config.get("pad_token_id")
        self._prompt_ids: List[int] = []
        self._state = GenerationState()
        self._sampler = self._create_sampler()

    def _create_sampler(self) -> CombinedSampler:
        return create_sampler({
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_tokens_to_keep": 1,
            "repetition_penalty": self.repetition_penalty,
        })

    def set_generation_params(self, max_tokens: Optional[int] = None,
                              temperature: Optional[float] = None,
                              top_p: Optional[float] = None,
                              top_k: Optional[int] = None,
                              repetition_penalty: Optional[float] = None,
                              device_idx: Optional[int] = None):
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
        if device_idx is not None:
            self._state.device_idx = device_idx

    def set_prompt_ids(self, prompt_ids: List[int]):
        self._prompt_ids = prompt_ids

    def reset(self):
        self._state = GenerationState(device_idx=self._state.device_idx)

    def __iter__(self) -> Iterator[int]:
        self.reset()
        for step_idx in range(self.max_tokens):
            if self._state.is_done:
                break
            self._state.step_idx = step_idx
            yield step_idx

    def step(self, logits: NBXTensor, step_idx: int) -> Tuple[NBXTensor, bool]:
        """Sample next token from logits.

        Args:
            logits: Token logits [batch, vocab] or [batch, seq, vocab] as NBXTensor.
            step_idx: Current step index.

        Returns:
            (next_token_tensor, is_done)
        """
        if logits.ndim == 3:
            # [batch, seq, vocab] → [batch, vocab] — take last position
            logits = logits.select(1, logits.shape[1] - 1)
        elif logits.ndim != 2:
            raise ValueError(f"Expected logits [batch, vocab] or [batch, seq, vocab], "
                             f"got {logits.shape}")

        # Build context IDs for repetition penalty
        all_context = self._prompt_ids + self._state.generated_tokens
        if all_context:
            ctx_np = np.array([all_context], dtype=np.int64)
            DeviceAllocator.set_device(self._state.device_idx)
            context_ids = NBXTensor.from_numpy(ctx_np)
        else:
            context_ids = None

        next_token = self._sampler(logits, input_ids=context_ids)

        # Read token ID to CPU (single int)
        token_id = int(next_token.item())
        self._state.generated_tokens.append(token_id)
        self._state.step_idx = step_idx

        if self.eos_token_id is not None and token_id == self.eos_token_id:
            self._state.is_done = True
        if len(self._state.generated_tokens) >= self.max_tokens:
            self._state.is_done = True

        _prog = __import__("os").environ.get("NBX_DECODE_PROGRESS")
        if _prog:  # gated decode-trajectory dump (autoregressive); buffer-immune
            with open(_prog, "a") as _pf:
                _pf.write(f"step={step_idx} n={len(self._state.generated_tokens)} "
                          f"last={token_id} done={self._state.is_done}\n")
                _pf.flush()

        return next_token, self._state.is_done

    def get_generated_tokens(self) -> List[int]:
        """Return generated token IDs as Python list."""
        return list(self._state.generated_tokens)

    @property
    def is_done(self) -> bool:
        return self._state.is_done

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TritonGenerator":
        return cls(config)
