"""Triton Sampling Strategies — zero torch dependency.

Ported from core/module/autoregressive/samplers.py.
All tensor ops use NBXTensor + Triton kernel wrappers from kernels/.

Implements: greedy, temperature, top-k, top-p, repetition penalty, combined.
"""

from typing import Any, Dict, Optional

import numpy as np

from nbxtensor.tensor import NBXTensor
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import wrappers as w
from nbxtensor.ops import attention_wrappers as _nbx_attention_wrappers  # the family the moved names live in (M2 split)
from nbxtensor.ops import elementwise_wrappers as _nbx_elementwise_wrappers  # the family the moved names live in (M2 split)
from nbxtensor.ops import index_wrappers as _nbx_index_wrappers  # the family the moved names live in (M2 split)
from nbxtensor.ops import random_wrappers as _nbx_random_wrappers  # the family the moved names live in (M2 split)
from nbxtensor.ops import reduction_wrappers as _nbx_reduction_wrappers  # the family the moved names live in (M2 split)
from nbxtensor.ops.token_filters import outside_nucleus, penalize_repeats


# =============================================================================
# Config (pure Python — shared with native mode)
# =============================================================================


class SamplerConfig:
    """Config validation for LLM samplers."""

    DEFAULTS: Dict[str, Any] = {
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "min_tokens_to_keep": 1,
    }

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        clean = {k: v for k, v in config.items() if not k.startswith("_")}
        for key, default in cls.DEFAULTS.items():
            if key not in clean:
                clean[key] = default
        return clean


# =============================================================================
# Samplers
# =============================================================================

class GreedySampler:
    """Greedy decoding — always select highest probability token."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def __call__(self, logits: NBXTensor, **kwargs) -> NBXTensor:
        return _nbx_reduction_wrappers.argmax_wrapper(logits, dim=-1, keepdim=True)


class TemperatureSampler:
    """Temperature-scaled sampling."""

    def __init__(self, config: Dict[str, Any]):
        validated = SamplerConfig.validate(config)
        self.temperature = validated["temperature"]

    def __call__(self, logits: NBXTensor, **kwargs) -> NBXTensor:
        if self.temperature != 1.0 and self.temperature > 0:
            logits = _nbx_elementwise_wrappers.div(logits, self.temperature)
        probs = _nbx_attention_wrappers.softmax(logits, dim=-1)
        return _nbx_random_wrappers.multinomial_wrapper(probs, num_samples=1)


class TopKSampler:
    """Top-K sampling — sample from top k tokens."""

    def __init__(self, config: Dict[str, Any]):
        validated = SamplerConfig.validate(config)
        self.top_k = validated["top_k"]
        self.temperature = validated["temperature"]
        self.min_tokens_to_keep = validated["min_tokens_to_keep"]

    def __call__(self, logits: NBXTensor, **kwargs) -> NBXTensor:
        if self.temperature != 1.0 and self.temperature > 0:
            logits = _nbx_elementwise_wrappers.div(logits, self.temperature)

        if self.top_k > 0:
            top_k = min(max(self.top_k, self.min_tokens_to_keep), logits.shape[-1])
            values, _indices = _nbx_elementwise_wrappers.topk_wrapper(logits, top_k, dim=-1)
            # kth value is the last in top-k (smallest of the top)
            kth = values.select(-1, top_k - 1).unsqueeze(-1)
            mask = _nbx_elementwise_wrappers.lt(logits, kth)
            logits = _nbx_index_wrappers.masked_fill(logits, mask, float('-inf'))

        probs = _nbx_attention_wrappers.softmax(logits, dim=-1)
        return _nbx_random_wrappers.multinomial_wrapper(probs, num_samples=1)


class TopPSampler:
    """Top-P (Nucleus) sampling — sample from smallest set with cumsum >= p."""

    def __init__(self, config: Dict[str, Any]):
        validated = SamplerConfig.validate(config)
        self.top_p = validated["top_p"]
        self.top_k = validated["top_k"]
        self.temperature = validated["temperature"]
        self.min_tokens_to_keep = validated["min_tokens_to_keep"]

    def __call__(self, logits: NBXTensor, **kwargs) -> NBXTensor:
        if self.temperature != 1.0 and self.temperature > 0:
            logits = _nbx_elementwise_wrappers.div(logits, self.temperature)

        # Top-k pre-filter
        if self.top_k > 0:
            top_k = min(max(self.top_k, self.min_tokens_to_keep), logits.shape[-1])
            values, _indices = _nbx_elementwise_wrappers.topk_wrapper(logits, top_k, dim=-1)
            kth = values.select(-1, top_k - 1).unsqueeze(-1)
            mask = _nbx_elementwise_wrappers.lt(logits, kth)
            logits = _nbx_index_wrappers.masked_fill(logits, mask, float('-inf'))

        # Top-p filter
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = _nbx_elementwise_wrappers.sort_wrapper(logits, dim=-1, descending=True)
            sorted_probs = _nbx_attention_wrappers.softmax(sorted_logits, dim=-1)
            cumulative_probs = _nbx_reduction_wrappers.cumsum_wrapper(sorted_probs, dim=-1)

            # Mask tokens with cumulative prob above threshold
            sorted_mask = _nbx_elementwise_wrappers.gt(cumulative_probs, self.top_p)

            # Keep at least min_tokens_to_keep: zero out first N positions in mask
            # Shift mask right by 1 to include the boundary token
            # These are small CPU operations on the mask tensor
            # For simplicity, use the sorted logits approach: mask and scatter back
            logits = _nbx_index_wrappers.masked_fill(logits, sorted_mask, float('-inf'))

        probs = _nbx_attention_wrappers.softmax(logits, dim=-1)
        return _nbx_random_wrappers.multinomial_wrapper(probs, num_samples=1)


class CombinedSampler:
    """Combined sampler with temperature, top-k, top-p, and repetition penalty.

    Most flexible sampler for production use.
    """

    def __init__(self, config: Dict[str, Any]):
        # Every decode's four parameters come resolved (sampling_contract.resolve_sampling); a missing one is a
        # caller defect named here, never the literal SamplerConfig.DEFAULTS would put in its place.
        from neurobrix.core.module.sampling_contract import SAMPLING_KEYS
        _missing = [k for k in SAMPLING_KEYS if config.get(k) is None]
        if _missing:
            raise RuntimeError(f"ZERO FALLBACK: CombinedSampler built without {', '.join(_missing)} "
                               f"(resolve them with core/module/sampling_contract.resolve_sampling)")
        validated = SamplerConfig.validate(config)
        self.temperature = validated["temperature"]
        self.top_k = validated["top_k"]
        self.top_p = validated["top_p"]
        self.repetition_penalty = validated["repetition_penalty"]
        self.min_tokens_to_keep = validated["min_tokens_to_keep"]

    def _sample_from_candidates(self, logits: NBXTensor) -> NBXTensor:
        """Sample from the top-k candidates, working at width k.

        The masking formulation keeps the vector at full vocabulary width
        and lets every later stage pay for entries that cannot be
        sampled. On a 151936-token vocabulary with top_k=20, that is a
        7600x overcharge on the sort, the softmax, the cumulative sum and
        the draw. `multinomial` is the worst of them: it samples by
        Gumbel-max, so it draws one uniform PER CATEGORY — 151936 random
        numbers to pick one token.

        Gathering the candidates first runs the same arithmetic over
        top_k entries. It also removes the sort outright: topk already
        returns its values sorted descending, which is the order the
        top-p filter needs.

        Distributionally this is exact. Gumbel-max over the masked full
        vector adds noise to every entry, but the masked ones hold -inf
        and -inf + g is still -inf, so only the k survivors can ever win
        — both forms are exact Gumbel-max sampling from the same
        categorical distribution. What changes is the REALISATION: k
        Gumbels are drawn instead of V, so a seeded run picks a different
        (equally valid) token. Nothing promises RNG realisations are
        stable across versions, but a recorded baseline will move.
        """
        top_k = min(max(self.top_k, self.min_tokens_to_keep), logits.shape[-1])
        cand_logits, cand_ids = _nbx_elementwise_wrappers.topk_wrapper(logits, top_k, dim=-1)

        if self.top_p < 1.0:
            probs = _nbx_attention_wrappers.softmax(cand_logits, dim=-1)
            cum = _nbx_reduction_wrappers.cumsum_wrapper(probs, dim=-1)
            mask = outside_nucleus(cum, self.top_p)
            cand_logits = _nbx_index_wrappers.masked_fill(cand_logits, mask, float('-inf'))

        probs = _nbx_attention_wrappers.softmax(cand_logits, dim=-1)
        # A position within the candidate set, not a vocabulary id.
        pos = _nbx_random_wrappers.multinomial_wrapper(probs, num_samples=1)
        return _nbx_index_wrappers.gather_wrapper(cand_ids, -1, pos)

    def __call__(self, logits: NBXTensor,
                 input_ids: Optional[NBXTensor] = None, **kwargs) -> NBXTensor:
        # 1. Repetition penalty (in place: the caller hands a row of its own)
        logits = penalize_repeats(logits, input_ids, self.repetition_penalty)

        # 2. Temperature scaling
        if self.temperature != 1.0 and self.temperature > 0:
            logits = _nbx_elementwise_wrappers.div(logits, self.temperature)

        # Greedy needs no candidate set and draws nothing.
        if self.temperature <= 0:
            return _nbx_reduction_wrappers.argmax_wrapper(logits, dim=-1, keepdim=True)

        # 3+4+5 at width k when a top-k cut bounds the candidate set.
        if self.top_k > 0:
            return self._sample_from_candidates(logits)

        # 4. Top-p filtering (the library's sort; values read in fp32)
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = _nbx_elementwise_wrappers.sort_wrapper(
                logits.to(NBXDtype.float32), dim=-1, descending=True)
            sorted_probs = _nbx_attention_wrappers.softmax(sorted_logits, dim=-1)
            cum_probs = _nbx_reduction_wrappers.cumsum_wrapper(sorted_probs, dim=-1)
            sorted_mask = outside_nucleus(cum_probs, self.top_p)
            indices_to_remove = _nbx_index_wrappers.scatter_wrapper(
                sorted_mask, -1, sorted_indices, sorted_mask)
            logits = _nbx_index_wrappers.masked_fill(logits, indices_to_remove, float('-inf'))

        # 5. Sample. Greedy already returned above.
        probs = _nbx_attention_wrappers.softmax(logits, dim=-1)
        return _nbx_random_wrappers.multinomial_wrapper(probs, num_samples=1)


# =============================================================================
# Factory
# =============================================================================


def sample_next_id(sampler: CombinedSampler, logits: NBXTensor, context_ids) -> int:
    """One token id from a decode step's logits ([1, S, V] or [1, V]) through `sampler`. The row is read in fp32 once
    whatever the head stored (the TritonGenerator's rule, as vLLM: never computed in half) and copied when it already
    is fp32 — the penalty scales in place and the logits stay the model's. `context_ids`: the ids the penalty reads."""
    row = logits[:, -1, :] if logits.ndim == 3 else logits
    row = row.to(NBXDtype.float32) if row._dtype != NBXDtype.float32 else row.clone()
    ids = None
    if context_ids:
        ids = NBXTensor.from_numpy(np.array([list(context_ids)], dtype=np.int64), device=row._device_idx)
    return int(sampler(row, input_ids=ids).item())

